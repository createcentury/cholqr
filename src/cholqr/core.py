import torch

__all__ = ["cholesky_qr", "orthonormal_basis"]


def cholesky_qr(Y: torch.Tensor, shift: float = 1e-6, passes: int = 3, out_dtype=None,
                robust: bool = False) -> torch.Tensor:
    """Orthonormal basis Q of range(Y) for a (batch of) tall-skinny matrix Y of shape (..., m, r), r <= m.

    Shifted CholeskyQR3 with three implementation choices that make it fast on GPUs:

    * every pass rescales the columns before forming the Gram matrix (scale invariant, no FP32 overflow);
    * the r x r triangular inverse is formed explicitly and applied as one batched GEMM (a triangular solve
      over m rows is slow on GPUs);
    * ``torch.linalg.cholesky_ex`` is used and a failed pass is skipped per matrix with ``torch.where``, so the
      routine never synchronizes with the host and compiles into a single graph under ``torch.compile``.

    Pass 1 uses the shift ``shift * tr(G)`` (a heuristic, not the a-priori shift of Fukaya et al., whose FP32 value is
    impractically large for large m*r); later passes use ``(u / r) * tr(G)`` with u the FP32 unit roundoff, restoring
    ``||Q^T Q - I||`` to O(u). Computation is in FP32; the result is cast to ``out_dtype`` (default: the dtype of Y).

    Range of validity (tested on 3072 x 210 inputs with geometric and power-law spectra):

    * default (FP32): reliable for cond(Y) up to about 1e6. Beyond ~5e6 the factorization breaks down
      (orthogonality is lost and passes are skipped). Inputs that come from bf16 sketches are usually fine further,
      because bf16 rounding caps their effective condition number.
    * ``robust=True``: the first pass runs in FP64 with a tiny shift, which handles cond(Y) up to at least 1e8 with
      ``||Q^T Q - I||_F ~ 1e-5``. It costs one FP64 Gram product (cheap on data-center GPUs, slow on consumer GPUs).

    Args:
        Y: tensor of shape (..., m, r) with r <= m.
        shift: relative shift of the first pass.
        passes: number of CholeskyQR passes (3 = shifted CholeskyQR3).
        out_dtype: dtype of the result.
        robust: run the first pass in FP64 (see above).
    Returns:
        Q of shape (..., m, r) with orthonormal columns spanning range(Y) (when Y has full column rank).
    """
    if Y.dim() < 2:
        raise ValueError("Y must have at least 2 dimensions (..., m, r)")
    m, r = Y.shape[-2], Y.shape[-1]
    if r > m:
        raise ValueError(f"cholesky_qr expects tall-skinny input (r <= m), got m={m}, r={r}")
    u = torch.finfo(torch.float32).eps
    Q = Y.double() if robust else Y.float()
    for i in range(passes):
        if robust and i == 1:
            Q = Q.float()
        eye = torch.eye(r, device=Q.device, dtype=Q.dtype)
        Q = Q / Q.norm(dim=-2, keepdim=True).clamp_min(1e-30)
        G = Q.mT @ Q
        tr = G.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[..., None, None]
        if i == 0:
            s = (1e-14 if robust else shift) * tr
        else:
            s = (u / r) * tr
        L, info = torch.linalg.cholesky_ex(G + (s + 1e-30) * eye)
        L_inv = torch.linalg.solve_triangular(L, eye.expand_as(L), upper=False)
        Q_next = Q @ L_inv.mT
        Q = torch.where((info == 0)[..., None, None], Q_next, Q)
    return Q.float().to(Y.dtype if out_dtype is None else out_dtype)


def orthonormal_basis(Y: torch.Tensor, **kwargs) -> torch.Tensor:
    """Alias of :func:`cholesky_qr` (drop-in for ``torch.linalg.qr(Y, mode="reduced")[0]`` when only Q is used)."""
    return cholesky_qr(Y, **kwargs)
