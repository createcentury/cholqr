import warnings
from typing import Optional

import torch

__all__ = ["cholesky_qr", "orthonormal_basis"]

_METHODS = ("scqr3", "cqr2")


def cholesky_qr(Y: torch.Tensor, method: str = "scqr3", robust: bool = False, return_r: bool = False,
                out_dtype=None, shift: float = 1e-6, passes: Optional[int] = None):
    """Thin QR of a (batch of) tall-skinny matrix Y of shape (..., m, r), r <= m, via Cholesky QR.

    Methods
    -------
    ``"scqr3"`` (default): shifted CholeskyQR3 (Fukaya, Kannan, Nakatsukasa, Yamamoto, Yanagisawa, 2020). A shifted
    first pass followed by two CholeskyQR passes. FP32 is reliable for cond(Y) up to about 1e6. With ``robust=True``
    the first pass runs in FP64, which handles cond(Y) up to at least 1e8.

    ``"cqr2"``: CholeskyQR2 (Yamamoto, Nakatsukasa, Yanagisawa, Fukaya, 2015). Two passes, about 1/3 cheaper. Use it
    **only for well-conditioned inputs**: in FP32 it breaks down beyond cond(Y) ~ u^{-1/2} ~ 4e3.

    Implementation notes (all methods): every pass rescales the columns before the Gram product (scale invariant, no
    FP32 overflow). The r x r triangular inverse is applied as one batched GEMM. ``cholesky_ex`` with a per-matrix
    ``torch.where`` fallback skips a failed pass without host synchronization, so the function compiles into a single
    graph.

    Args:
        Y: real tensor of shape (..., m, r) with r <= m.
        method: ``"scqr3"`` or ``"cqr2"``.
        robust: (scqr3 only) run the first pass in FP64.
        return_r: also return the upper-triangular factor R with Y = Q R.
        out_dtype: dtype of the results (default: dtype of Y).
        shift: relative shift of the first scqr3 pass (FP32 mode).
        passes: override the number of passes (default 3 for scqr3, 2 for cqr2).
    Returns:
        Q of shape (..., m, r), or (Q, R) with R of shape (..., r, r) if ``return_r``.
    """
    if Y.dim() < 2:
        raise ValueError("Y must have at least 2 dimensions (..., m, r)")
    if Y.is_complex():
        raise TypeError("cholesky_qr supports real tensors only (got a complex dtype)")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if robust and method != "scqr3":
        raise ValueError("robust=True is only available for method='scqr3'")
    if Y.is_cuda and torch.get_float32_matmul_precision() != "highest" and not torch.compiler.is_compiling():
        warnings.warn("cholqr: TF32 matmul is enabled (float32_matmul_precision != 'highest'). Gram products are then "
                      "computed at ~1e-3 precision: orthogonality degrades to ~1e-4 (scqr3) and cqr2 becomes unreliable "
                      "already near cond(Y) ~ 1e3. Use torch.set_float32_matmul_precision('highest') around the call.",
                      RuntimeWarning, stacklevel=2)
    m, r = Y.shape[-2], Y.shape[-1]
    if r > m:
        raise ValueError(f"cholesky_qr expects tall-skinny input (r <= m), got m={m}, r={r}")
    n_pass = passes if passes is not None else (3 if method == "scqr3" else 2)
    u = torch.finfo(torch.float32).eps
    Q = Y.double() if robust else Y.float()
    R = None
    for i in range(n_pass):
        if robust and i == 1:
            Q = Q.float()
            R = R.float() if R is not None else None
        eye = torch.eye(r, device=Q.device, dtype=Q.dtype)
        d = Q.norm(dim=-2, keepdim=True).clamp_min(1e-30)                 # (..., 1, r) column norms
        Q = Q / d
        G = Q.mT @ Q
        tr = G.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[..., None, None]
        if method == "scqr3" and i == 0:
            s = (1e-14 if robust else shift) * tr
        else:
            s = (u / r) * tr                                                  # O(u) safeguard
        L, info = torch.linalg.cholesky_ex(G + (s + 1e-30) * eye)
        L_inv = torch.linalg.solve_triangular(L, eye.expand_as(L), upper=False)
        ok = (info == 0)[..., None, None]
        Q = torch.where(ok, Q @ L_inv.mT, Q)
        if return_r:
            # Q_before = Q_scaled * d  and  Q_scaled = Q_after @ L^T (if the pass succeeded)
            D = torch.diag_embed(d.squeeze(-2))
            F = torch.where(ok, L.mT @ D, D)
            R = F if R is None else F @ R
    dt = Y.dtype if out_dtype is None else out_dtype
    Q = Q.float().to(dt)
    if not return_r:
        return Q
    return Q, R.float().to(dt)


def orthonormal_basis(Y: torch.Tensor, **kwargs) -> torch.Tensor:
    """Orthonormal basis of range(Y) (drop-in for ``torch.linalg.qr(Y, mode="reduced")[0]``)."""
    kwargs.pop("return_r", None)
    return cholesky_qr(Y, **kwargs)
