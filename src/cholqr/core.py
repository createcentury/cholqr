import math
import warnings
from typing import Optional

import torch

__all__ = ["cholesky_qr", "orthonormal_basis"]

_METHODS = ("scqr3", "cqr2", "rcqr", "auto")
_ON_BREAKDOWN = ("fallback", "warn", "raise", "ignore")
_BREAKDOWN_LMIN = 1e-2   # last-pass Cholesky diagonal below this = the last pass's input was numerically rank deficient
_AUTO_QR_MAX_R = 64   # single matrices with r <= this go to torch.linalg.qr (A100 measurements, see README)


_GRAM_CHUNK = 256
_GRAM_SPLITK_MIN_ROWS = 2048   # below this the plain GEMM is used (A100 A/B, see lowrank-muon dispatch results)


def _gram(Q: torch.Tensor) -> torch.Tensor:
    """Q^T Q for tall Q (..., m, r) with a manual split-K over m.

    cuBLAS often runs a small-output (r x r), long-reduction (m) GEMM without split-K, nearly serially
    (e.g. 1.13 ms for 4 x 6144 x 8 fp32 on an A100). Chunking m into blocks of _GRAM_CHUNK rows gives a batched
    GEMM plus a sum (about 14x faster there). Mathematically identical; only the summation order differs.
    """
    m = Q.shape[-2]
    if m < max(2 * _GRAM_CHUNK, _GRAM_SPLITK_MIN_ROWS):
        return Q.mT @ Q
    q = m // _GRAM_CHUNK
    head = Q[..., : q * _GRAM_CHUNK, :].reshape(*Q.shape[:-2], q, _GRAM_CHUNK, Q.shape[-1])
    G = (head.mT @ head).sum(-3)
    if m % _GRAM_CHUNK:
        tail = Q[..., q * _GRAM_CHUNK:, :]
        G = G + tail.mT @ tail
    return G


def cholesky_qr(Y: torch.Tensor, method: str = "scqr3", robust: bool = False, return_r: bool = False,
                out_dtype=None, shift: float = 1e-6, passes: Optional[int] = None, oversample: float = 1.25,
                generator: Optional[torch.Generator] = None, on_breakdown: str = "fallback"):
    """Thin QR of a (batch of) tall-skinny matrix Y of shape (..., m, r), r <= m, via Cholesky QR.

    Methods
    -------
    ``"scqr3"`` (default): shifted CholeskyQR3 (Fukaya, Kannan, Nakatsukasa, Yamamoto, Yanagisawa, 2020). A shifted
    first pass followed by two CholeskyQR passes. FP32 is reliable for cond(Y) up to about 1e6. With ``robust=True``
    the first pass runs in FP64, which handles cond(Y) up to at least 1e8.

    ``"cqr2"``: CholeskyQR2 (Yamamoto, Nakatsukasa, Yanagisawa, Fukaya, 2015). Two passes, about 1/3 cheaper. Use it
    **only for well-conditioned inputs**: in FP32 it breaks down beyond cond(Y) ~ u^{-1/2} ~ 4e3.

    ``"auto"``: dispatch on shape. A single matrix (no batch, or batch size 1) with r <= 64 goes to Householder
    ``torch.linalg.qr`` (in FP32, with R's diagonal made non-negative), because CholeskyQR's fixed launch overhead
    (~0.2-0.8 ms) dominates there. Everything else uses ``"scqr3"`` (honouring ``robust``).

    ``"rcqr"``: randomized (sketch-preconditioned) CholeskyQR, as in Balabanov (2022), Higgins et al. (2024) and the
    Dion optimizer. A Gaussian sketch S (k x m, k = oversample * r rounded up to a multiple of 128, as in Dion) gives
    R0 from a Householder QR of the small matrix S Y; Y R0^{-1} is then well conditioned and one CholeskyQR pass
    finishes. One sketch is shared by all matrices of a batch.

    Implementation notes (all methods): every pass rescales the columns before the Gram product (scale invariant, no
    FP32 overflow). The r x r triangular inverse is applied as one batched GEMM. ``cholesky_ex`` with a per-matrix
    ``torch.where`` fallback skips a failed pass without host synchronization, so the function compiles into a single
    graph.

    Breakdown detection (new in 0.5): a matrix is flagged when a Cholesky factorization fails in any pass (the pass was
    skipped, so the result is not orthonormal) or when the last pass's Cholesky factor has a diagonal entry below
    ``1e-2`` (the input to the last pass was numerically rank deficient, e.g. a zero column). Both tests reuse the r x r
    factors already computed; they add no Gram product. This is how FP32 ``scqr3`` fails beyond cond(Y) ~ 1e6-1e7:
    without the check the result silently loses orthogonality (||Q^T Q - I||_F ~ 20-90).

    Args:
        Y: real tensor of shape (..., m, r) with r <= m.
        method: ``"scqr3"``, ``"cqr2"``, ``"rcqr"`` or ``"auto"``.
        robust: (scqr3 only) run the first pass in FP64.
        return_r: also return the upper-triangular factor R with Y = Q R.
        out_dtype: dtype of the results (default: dtype of Y).
        shift: relative shift of the first scqr3 pass (FP32 mode).
        passes: override the number of passes (default 3 for scqr3, 2 for cqr2; for rcqr: CholeskyQR passes after the
            sketch step, default 1).
        oversample: (rcqr only) sketch size factor.
        generator: (rcqr only) torch.Generator for the sketch (graph-breaks under torch.compile).
        on_breakdown: what to do with flagged matrices. ``"fallback"`` (default): recompute only the flagged matrices,
            first with ``scqr3`` and an FP64 first pass (``robust=True``; skipped on devices without FP64), then, if
            still flagged, with Householder ``torch.linalg.qr``. ``"warn"``: warn and return the unrepaired result.
            ``"raise"``: raise ``torch.linalg.LinAlgError``. ``"ignore"``: return the result as computed (the 0.4
            behaviour). Every option except ``"ignore"`` costs one host synchronization (reading a flag) per call.
            While compiling (``torch.compile``) the check is skipped, as with ``"ignore"``, so the function still
            compiles into a single graph; check orthogonality yourself there or use ``robust=True``.
    Returns:
        Q of shape (..., m, r), or (Q, R) with R of shape (..., r, r) if ``return_r``.
    """
    if Y.dim() < 2:
        raise ValueError("Y must have at least 2 dimensions (..., m, r)")
    if Y.is_complex():
        raise TypeError("cholesky_qr supports real tensors only (got a complex dtype)")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if robust and method not in ("scqr3", "auto"):
        raise ValueError("robust=True is only available for method='scqr3' or 'auto'")
    if on_breakdown not in _ON_BREAKDOWN:
        raise ValueError(f"on_breakdown must be one of {_ON_BREAKDOWN}, got {on_breakdown!r}")
    # is_compiling() must be evaluated first: older Dynamo (e.g. torch 2.7) cannot trace
    # get_float32_matmul_precision() (returns a str) and fails under fullgraph=True.
    if not torch.compiler.is_compiling() and Y.is_cuda and torch.get_float32_matmul_precision() != "highest":
        warnings.warn("cholqr: TF32 matmul is enabled (float32_matmul_precision != 'highest'). Gram products are then "
                      "computed at ~1e-3 precision: orthogonality degrades to ~1e-4 (scqr3) and cqr2 becomes unreliable "
                      "already near cond(Y) ~ 1e3. Use torch.set_float32_matmul_precision('highest') around the call.",
                      RuntimeWarning, stacklevel=2)
    m, r = Y.shape[-2], Y.shape[-1]
    if r > m:
        raise ValueError(f"cholesky_qr expects tall-skinny input (r <= m), got m={m}, r={r}")
    if method == "auto":
        batch = math.prod(Y.shape[:-2])
        if batch <= 1 and r <= _AUTO_QR_MAX_R:
            return _householder(Y, return_r, out_dtype)
        method = "scqr3"
    Q, R, bad = _cqr(Y, method, robust, return_r, out_dtype, shift, passes, oversample, generator)
    if on_breakdown != "ignore" and not torch.compiler.is_compiling() and bool(bad.any()):
        n_bad, n_all = int(bad.sum()), bad.numel()
        if on_breakdown == "raise":
            raise torch.linalg.LinAlgError(
                f"cholqr: CholeskyQR broke down for {n_bad} of {n_all} matrices (input too ill-conditioned or rank "
                f"deficient for method={method!r}, robust={robust}); use robust=True or on_breakdown='fallback'")
        if on_breakdown == "warn":
            warnings.warn(f"cholqr: CholeskyQR broke down for {n_bad} of {n_all} matrices; the result is not "
                          "orthonormal for them (use robust=True or on_breakdown='fallback')", RuntimeWarning, stacklevel=2)
        else:
            Q, R = _repair(Y, Q, R, bad, method, robust, return_r, out_dtype, shift, passes, oversample, generator)
    return (Q, R) if return_r else Q


def _prescale(Q):
    """Overflow guard for the first FP32 column norm. Returns (Q / a, a): a = 1 (exact, so results are bit-identical to
    the plain norm) for columns whose norm is finite, and a = column max where Q.norm() overflows (FP32 entries above
    ~1e19). Branch-free, so it compiles into the graph."""
    d = Q.norm(dim=-2, keepdim=True)
    a = torch.where(torch.isfinite(d), torch.ones_like(d), Q.abs().amax(dim=-2, keepdim=True).clamp_min(1e-30))
    return Q / a, a


def _repair(Y, Q, R, bad, method, robust, return_r, out_dtype, shift, passes, oversample, generator):
    """Recompute the flagged matrices: scqr3 with an FP64 first pass, then Householder for any still flagged."""
    m, r = Y.shape[-2], Y.shape[-1]
    shape = Q.shape
    idx = bad.reshape(-1).nonzero().squeeze(1)
    Ys = Y.reshape(-1, m, r)[idx]
    can_fp64 = Ys.device.type != "mps"
    if can_fp64 and not (method == "scqr3" and robust):
        Qs, Rs, bad_s = _cqr(Ys, "scqr3", True, return_r, out_dtype, shift, None, oversample, generator)
        if bool(bad_s.any()):
            j = bad_s.nonzero().squeeze(1)
            Qh, Rh = _householder(Ys[j], True, out_dtype)
            Qs = Qs.index_copy(0, j, Qh)
            Rs = Rs.index_copy(0, j, Rh) if return_r else None
    else:
        Qs, Rs = _householder(Ys, True, out_dtype)
    Q = Q.reshape(-1, m, r).index_copy(0, idx, Qs.to(Q.dtype)).reshape(shape)
    if return_r:
        R = R.reshape(-1, r, r).index_copy(0, idx, Rs.to(R.dtype)).reshape(R.shape)
    return Q, R


def _cqr(Y, method, robust, return_r, out_dtype, shift, passes, oversample, generator):
    """Core passes. Returns (Q, R or None, bad) with bad of shape Y.shape[:-2] (breakdown flags)."""
    m, r = Y.shape[-2], Y.shape[-1]
    n_pass = passes if passes is not None else {"scqr3": 3, "cqr2": 2, "rcqr": 1}[method]
    u = torch.finfo(torch.float32).eps
    Q = Y.double() if robust else Y.float()
    R = None
    if method == "rcqr":
        k = min(m, max(r, math.ceil(oversample * r / 128) * 128))
        Q, a0 = _prescale(Q) if Q.dtype == torch.float32 else (Q, None)
        d0 = Q.norm(dim=-2, keepdim=True).clamp_min(1e-30)
        Q = Q / d0
        d0 = d0 if a0 is None else d0 * a0                                 # total column scale (for R)
        S = torch.randn(k, m, device=Q.device, dtype=Q.dtype, generator=generator) / math.sqrt(k)
        R0 = torch.linalg.qr(S @ Q, mode="r")[1]                            # (..., r, r)
        dg = R0.diagonal(dim1=-2, dim2=-1)
        tiny = 1e-30 + u * dg.abs().amax(dim=-1, keepdim=True)
        dg_safe = torch.where(dg.abs() < tiny, torch.where(dg < 0, -tiny, tiny), dg)   # rank-deficient guard
        R0 = R0 - torch.diag_embed(dg) + torch.diag_embed(dg_safe)
        Q = torch.linalg.solve_triangular(R0, Q, upper=True, left=False)
        if return_r:
            R = R0 @ torch.diag_embed(d0.squeeze(-2))
    bad = None
    eye = torch.eye(r, device=Q.device, dtype=Q.dtype)                    # built once (torch.eye is costly to launch)
    for i in range(n_pass):
        if robust and i == 1:
            Q = Q.float()
            R = R.float() if R is not None else None
            eye = eye.float()
        a = None
        if i == 0 and Q.dtype == torch.float32:
            Q, a = _prescale(Q)
        d = Q.norm(dim=-2, keepdim=True).clamp_min(1e-30)                 # (..., 1, r) column norms
        Q = Q / d
        if a is not None:
            d = d * a                                                     # total column scale (for R)
        G = _gram(Q)
        # columns have unit norm, so tr(G) = r (zero columns only make it smaller, i.e. the shift is conservative)
        if method == "scqr3" and i == 0:
            s = (1e-14 if robust else shift) * r
        else:
            s = u                                                             # O(u) safeguard, = (u / r) * tr(G)
        L, info = torch.linalg.cholesky_ex(G + (s + 1e-30) * eye)
        L_inv = torch.linalg.solve_triangular(L, eye.expand_as(L), upper=False)
        ok = (info == 0)[..., None, None]
        bad = (info != 0) if bad is None else (bad | (info != 0))
        if i == n_pass - 1:
            bad = bad | (L.diagonal(dim1=-2, dim2=-1).amin(dim=-1) < _BREAKDOWN_LMIN)
        Q = torch.where(ok, Q @ L_inv.mT, Q)
        if return_r:
            # Q_before = Q_scaled * d  and  Q_scaled = Q_after @ L^T (if the pass succeeded)
            D = torch.diag_embed(d.squeeze(-2))
            F = torch.where(ok, L.mT @ D, D)
            R = F if R is None else F @ R
    dt = Y.dtype if out_dtype is None else out_dtype
    Q = Q.float().to(dt)
    if return_r and R is None:                                           # passes=0
        R = torch.eye(r, device=Q.device, dtype=Q.dtype).expand(*Y.shape[:-2], r, r)
    if bad is None:                                                       # passes=0
        bad = torch.zeros(Y.shape[:-2], dtype=torch.bool, device=Y.device)
    return Q, (R.float().to(dt) if return_r else None), bad


def _householder(Y, return_r, out_dtype):
    """torch.linalg.qr in FP32 (half precision is not supported by it), normalized so that diag(R) >= 0."""
    Q, R = torch.linalg.qr(Y.float(), mode="reduced")
    sgn = torch.where(R.diagonal(dim1=-2, dim2=-1) < 0, -1.0, 1.0)       # (..., r)
    Q = Q * sgn[..., None, :]
    dt = Y.dtype if out_dtype is None else out_dtype
    if not return_r:
        return Q.to(dt)
    return Q.to(dt), (R * sgn[..., :, None]).to(dt)


def orthonormal_basis(Y: torch.Tensor, **kwargs) -> torch.Tensor:
    """Orthonormal basis of range(Y) (drop-in for ``torch.linalg.qr(Y, mode="reduced")[0]``)."""
    kwargs.pop("return_r", None)
    return cholesky_qr(Y, **kwargs)
