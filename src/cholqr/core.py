import functools
import math
import os
import warnings
from fractions import Fraction
from typing import Optional

import torch

__all__ = ["cholesky_qr", "orthonormal_basis", "orthogonality_bound"]


def _jit_tracing():
    """True under torch.jit.trace; False if the installed torch has no TorchScript tracer."""
    f = getattr(getattr(torch, "jit", None), "is_tracing", None)
    return bool(f()) if callable(f) else False


def _is_compiling() -> bool:
    """torch.compiler.is_compiling (torch >= 2.3); torch._dynamo.is_compiling on older versions."""
    f = getattr(torch.compiler, "is_compiling", None)
    if f is None:
        from torch import _dynamo
        f = _dynamo.is_compiling
    return f()


_METHODS = ("scqr3", "cqr2", "rcqr", "auto")
_ON_BREAKDOWN = ("fallback", "warn", "raise", "ignore")
# Last-pass test. A CholeskyQR pass on X leaves orthogonality ~ c u / Lmin^2 (Lmin = smallest diagonal entry of its
# Cholesky factor; c = 18-33 measured for FP32 Gram orders). Flagging Lmin < sqrt(64 u / 1e-4) = 0.195 targets a
# final orthogonality of 1e-4 with a 2x safety factor on c. Pre-registered check on unseen data: caught 155/155 (CPU) and
# 56/56 (Apple GPU) runs with orthogonality > 1e-4, false alarms 4/1931 and 0/760; an earlier draft value of 1e-2 caught 65/155.
# Validated for r <= 276. Wider (pre-registered, benchmarks/results/thresh_wide_2026-11/): missed 1/69 degraded runs at
# r = 512 and 6/84 at r = 1024 (orthogonality 1.05e-4 to 1.8e-4), 0 false alarms; the threshold is unchanged.
_BREAKDOWN_LMIN = math.sqrt(64 * 2.0 ** -24 / 1e-4)
_AUTO_QR_MAX_R = 64   # single matrices with r <= this go to torch.linalg.qr (A100 measurements, see README)


_GRAM_CHUNK = 256
_GRAM_SPLITK_MIN_ROWS = 2048   # below this the plain GEMM is used (A100 A/B measurements)


def _in_functorch(Y: torch.Tensor) -> bool:
    """True inside torch.func transforms (vmap, grad, ...), where data-dependent Python control flow is not allowed."""
    try:
        return bool(torch._C._functorch.is_functorch_wrapped_tensor(Y))
    except Exception:
        return False


def _tf32_active(Q: torch.Tensor) -> bool:
    """True if a float32 CUDA matmul on Q may run in TF32: an NVIDIA GPU with compute capability >= 8.0 (Ampere or newer)
    and TF32 allowed (new API torch.backends.cuda.matmul.fp32_precision == "tf32", else the legacy allow_tf32 flag)."""
    if not (Q.is_cuda and Q.dtype == torch.float32) or getattr(torch.version, "hip", None):
        return False
    try:
        if torch.cuda.get_device_capability(Q.device)[0] < 8:
            return False
        fp = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
        if isinstance(fp, str) and fp not in ("", "none"):
            return fp == "tf32"
        return bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        return False


def _tf32_split(A: torch.Tensor):
    """A = hi + lo with hi exactly representable in TF32 (10-bit mantissa, round half away from zero) and lo = A - hi,
    exact by Sterbenz (A/2 <= hi <= 2A). Where rounding up overflows (|A| >= 3.40199e38) or A is not finite, hi = A and
    lo = 0 (plain product for those entries)."""
    b = A.contiguous().view(torch.int32)
    hi = ((b + 0x1000) & ~0x1FFF).view(torch.float32)
    good = torch.isfinite(hi)
    hi = torch.where(good, hi, A)
    return hi, torch.where(good, A - hi, torch.zeros_like(A))


def _mm(A: torch.Tensor, B: torch.Tensor, tf32: bool) -> torch.Tensor:
    """A @ B; with TF32 active, the 3xTF32 scheme hi@hi + (hi@lo + lo@hi) restores ~FP32 accuracy (lo@lo, ~2^-22
    relative, is dropped). Three TF32 GEMMs are still faster than one FP32 GEMM on tensor-core GPUs."""
    if not tf32:
        return A @ B
    Ah, Al = _tf32_split(A)
    Bh, Bl = _tf32_split(B)
    return Ah @ Bh + (Ah @ Bl + Al @ Bh)


def _gram_mm(A: torch.Tensor, tf32: bool) -> torch.Tensor:
    """A^T A; with TF32 active, Ah^T Ah + (M + M^T), M = Ah^T Al: two TF32 GEMMs and an exactly symmetric result."""
    if not tf32:
        return A.mT @ A
    Ah, Al = _tf32_split(A)
    M = Ah.mT @ Al
    return Ah.mT @ Ah + (M + M.mT)


def _gram(Q: torch.Tensor) -> torch.Tensor:
    """Q^T Q for tall Q (..., m, r) with a manual split-K over m.

    cuBLAS often runs a small-output (r x r), long-reduction (m) GEMM without split-K, nearly serially
    (e.g. 1.13 ms for 4 x 6144 x 8 fp32 on an A100). Chunking m into blocks of _GRAM_CHUNK rows gives a batched
    GEMM plus a sum (about 14x faster there). Mathematically identical; only the summation order differs.
    """
    m = Q.shape[-2]
    tf32 = _tf32_active(Q)
    if m < max(2 * _GRAM_CHUNK, _GRAM_SPLITK_MIN_ROWS):
        return _gram_mm(Q, tf32)
    q = m // _GRAM_CHUNK
    head = Q[..., : q * _GRAM_CHUNK, :].reshape(*Q.shape[:-2], q, _GRAM_CHUNK, Q.shape[-1])
    G = _gram_mm(head, tf32).sum(-3)
    if m % _GRAM_CHUNK:
        tail = Q[..., q * _GRAM_CHUNK:, :]
        G = G + _gram_mm(tail, tf32)
    return G


def cholesky_qr(Y: torch.Tensor, method: str = "scqr3", robust: bool = False, return_r: bool = False,
                out_dtype=None, shift: float = 1e-6, passes: Optional[int] = None, oversample: float = 1.25,
                generator: Optional[torch.Generator] = None, on_breakdown: str = "fallback", verify: bool = False,
                tol: Optional[float] = None):
    """Thin QR of a (batch of) tall-skinny matrix Y of shape (..., m, r), r <= m, via Cholesky QR.

    Methods
    -------
    ``"scqr3"`` (default): shifted CholeskyQR3 (Fukaya, Kannan, Nakatsukasa, Yamamoto, Yanagisawa, 2020). A shifted
    first pass followed by two CholeskyQR passes. FP32 is reliable for cond(Y) up to about 1e6. With ``robust=True``
    the first pass runs in FP64, which handles cond(Y) up to at least 1e8.

    ``"cqr2"``: CholeskyQR2 (Yamamoto, Nakatsukasa, Yanagisawa, Fukaya, 2015). Two passes, about 1/3 cheaper. Use it
    **only for well-conditioned inputs** (cond(Y) <~ 1e3 recommended): in FP32 its first breakdown was measured from cond(Y) ~ 3e3 (clustered spectra; u^{-1/2} ~ 4e3) to 1e4-1e5 (geometric spectra).

    ``"auto"``: dispatch on shape. A single matrix (no batch, or batch size 1) with r <= 64 goes to Householder
    ``torch.linalg.qr`` (in FP32, with R's diagonal made non-negative), because CholeskyQR's fixed launch overhead
    (~0.2-0.8 ms) dominates there. Everything else uses ``"scqr3"`` (honouring ``robust``).

    ``"rcqr"``: randomized (sketch-preconditioned) CholeskyQR, as in Balabanov (2022), Higgins et al. (2025) and the
    Dion optimizer. A Gaussian sketch S (k x m, k = oversample * r rounded up to a multiple of 128, as in Dion) gives
    R0 from a Householder QR of the small matrix S Y; Y R0^{-1} is then well conditioned and one CholeskyQR pass
    finishes. One sketch is shared by all matrices of a batch.

    Implementation notes (all methods): every pass rescales the columns before the Gram product (scale invariant, no
    FP32 overflow). The r x r triangular inverse is applied as one batched GEMM. ``cholesky_ex`` with a per-matrix
    ``torch.where`` fallback skips a failed pass without host synchronization, so the function compiles into a single
    graph.

    What is checked: orthogonality of Q only. Neither the default check nor ``verify=True`` says anything about how well
    range(Q) matches the range of the exact input: information lost when Y was rounded to a low-precision format
    (bfloat16, float16, float8, or plain TF32 products), or when Y was ill-conditioned, is not detected. A Q that passes
    ``verify=True`` can still span the wrong subspace in those cases.

    Breakdown detection (new in 0.5): a matrix is flagged when a Cholesky factorization fails in any pass (the pass was
    skipped, so the result is not orthonormal) or when the last pass's Cholesky factor has a diagonal entry below
    ``0.195`` (the input to the last pass was so ill-conditioned that the final orthogonality can exceed ~1e-4, e.g. cqr2
    near its limit, or numerically rank deficient, e.g. a zero column). Both tests reuse the r x r
    factors already computed; they add no Gram product. This is how FP32 ``scqr3`` fails beyond cond(Y) ~ 1e6-1e7:
    without the check the result silently loses orthogonality (||Q^T Q - I||_F ~ 20-90).

    The ``0.195`` threshold was validated for r <= 276 columns. For wider matrices it misses a few degraded runs:
    in a pre-registered test it missed 1 of 69 at r = 512 and 6 of 84 at r = 1024 (orthogonality 1.05e-4 to 1.8e-4
    in all missed runs), with no false alarms. For wide matrices that need a guaranteed 1e-4, use ``verify=True``.

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
            While compiling (``torch.compile``) and inside ``torch.func`` transforms (``vmap``, ``grad``) the check is skipped, as with ``"ignore"``, so the function still
            compiles into a single graph; check orthogonality yourself there or use ``robust=True``.
        verify: certify the result (new in 0.5, opt-in). Every returned Q~ is accepted only if an FP64 bound proves
            ``||Q~^T Q~ - I||_F <= tol`` for the exact values of the returned tensor (see ``orthogonality_bound``;
            a theorem under the assumptions H0-H2: FP32-valued Q~, IEEE FP64 standard model, and FP64 GEMMs that sum
            products in any order with or without FMA but without Strassen-type algorithms or FP64 emulation; given
            these, independent of TF32 or of how Q~ was computed). With ``verify=True`` the certificate replaces the empirical breakdown flags: uncertified
            matrices are handled per ``on_breakdown``: ``"fallback"`` recomputes them (scqr3 with an FP64 first pass if
            the device has FP64, then Householder QR, in FP64 where available), re-verifies, and raises
            ``torch.linalg.LinAlgError`` if a matrix still cannot be certified; ``"warn"`` warns and returns them;
            ``"raise"`` raises; ``"ignore"`` is rejected (``ValueError``). Not available while compiling or inside
            ``torch.func`` transforms (raises ``RuntimeError``; call ``orthogonality_bound`` outside the transform).
            Costs one FP64 Gram product (on the CPU for MPS tensors) and one host synchronization per call. Covers
            orthogonality only, not the accuracy of the subspace or of R.
        tol: (verify only) Frobenius tolerance. Default ``None`` = 1e-4 for float32/float64 output; for Q returned in
            float16/bfloat16/float8 it must be given explicitly (``ValueError`` otherwise), because rounding Q alone
            moves the error by ~ 2 sqrt(r) u_d; use ``out_dtype=torch.float32`` or tol >~ 1e-4 + 2 sqrt(r) u_d + r u_d^2.
    Returns:
        Q of shape (..., m, r), or (Q, R) with R of shape (..., r, r) if ``return_r``.
    """
    if Y.dim() < 2:
        raise ValueError("Y must have at least 2 dimensions (..., m, r)")
    if Y.dtype.is_complex:
        raise TypeError("cholesky_qr supports real tensors only (got a complex dtype)")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if robust and method not in ("scqr3", "auto"):
        raise ValueError("robust=True is only available for method='scqr3' or 'auto'")
    if on_breakdown not in _ON_BREAKDOWN:
        raise ValueError(f"on_breakdown must be one of {_ON_BREAKDOWN}, got {on_breakdown!r}")
    # is_compiling() must be evaluated first: older Dynamo (e.g. torch 2.7) cannot trace
    # get_float32_matmul_precision() (returns a str) and fails under fullgraph=True.
    # TF32 ("high") is compensated inside cholqr by 3xTF32 products (see _mm). "medium" (bfloat16 internal precision)
    # is not compensated and is warned about.
    if not _is_compiling() and Y.is_cuda and torch.get_float32_matmul_precision() == "medium":
        warnings.warn("cholqr: float32_matmul_precision is 'medium', which may use bfloat16-based matmul on some "
                      "backends. cholqr compensates TF32 (3xTF32) but not bfloat16 products. Use "
                      "torch.set_float32_matmul_precision('highest') or 'high' around the call if accuracy matters.",
                      RuntimeWarning, stacklevel=2)
    m, r = Y.shape[-2], Y.shape[-1]
    if r > m:
        raise ValueError(f"cholesky_qr expects tall-skinny input (r <= m), got m={m}, r={r}")
    if verify:
        _check_verify_call(Y, on_breakdown, tol)
        tol = _resolve_tol(Y, out_dtype, tol)
    if method == "auto":
        batch = math.prod(Y.shape[:-2])
        if batch <= 1 and r <= _AUTO_QR_MAX_R:
            if verify:
                return _verified_qr(Y, None, False, return_r, out_dtype, shift, passes, oversample, generator,
                                    on_breakdown, tol)
            return _householder(Y, return_r, out_dtype)
        method = "scqr3"
    if verify:
        return _verified_qr(Y, method, robust, return_r, out_dtype, shift, passes, oversample, generator, on_breakdown,
                            tol)
    Q, R, bad = _cqr(Y, method, robust, return_r, out_dtype, shift, passes, oversample, generator)
    if on_breakdown != "ignore" and not _is_compiling() and not _in_functorch(Y) and bool(bad.any()):
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
            Qs = Qs.float().index_copy(0, j, Qh.float()).to(Qs.dtype)
            Rs = Rs.float().index_copy(0, j, Rh.float()).to(Rs.dtype) if return_r else None
    else:
        Qs, Rs = _householder(Ys, True, out_dtype)
    # merge in FP32 (index_copy is not implemented for every dtype, e.g. float8), then cast back
    Q = Q.float().reshape(-1, m, r).index_copy(0, idx, Qs.float()).reshape(shape).to(Q.dtype)
    if return_r:
        R = R.float().reshape(-1, r, r).index_copy(0, idx, Rs.float()).reshape(R.shape).to(R.dtype)
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
        Gs = G + (s + 1e-30) * eye
        pre_fail = None
        if Gs.requires_grad and torch.is_grad_enabled():
            # Find failing matrices first without autograd and factor the identity for them instead: a failed factor can
            # hold NaN/inf (backend dependent), and torch.where would then turn its zero gradient into 0 * NaN = NaN.
            # The forward result is unchanged (a failed pass keeps Q); only the backward graph is made finite.
            with torch.no_grad():
                L0, info0 = torch.linalg.cholesky_ex(Gs)
                pre_fail = (info0 != 0) | ~torch.isfinite(L0.diagonal(dim1=-2, dim2=-1)).all(dim=-1)
            Gs = torch.where(pre_fail[..., None, None], eye.expand_as(Gs), Gs)
        L, info = torch.linalg.cholesky_ex(Gs)
        L_inv = torch.linalg.solve_triangular(L, eye.expand_as(L), upper=False)
        # a factorization counts as failed if cholesky_ex reports it OR its diagonal is not finite (defensive: some
        # backends can return NaN entries with info == 0)
        fail = (info != 0) | ~torch.isfinite(L.diagonal(dim1=-2, dim2=-1)).all(dim=-1)
        if pre_fail is not None:
            fail = fail | pre_fail
        ok = (~fail)[..., None, None]
        bad = fail if bad is None else (bad | fail)
        if i == n_pass - 1:
            bad = bad | (L.diagonal(dim1=-2, dim2=-1).amin(dim=-1) < _BREAKDOWN_LMIN)
        Q = torch.where(ok, _mm(Q, L_inv.mT, _tf32_active(Q)), Q)
        if return_r:
            # Q_before = Q_scaled * d  and  Q_scaled = Q_after @ L^T (if the pass succeeded)
            D = torch.diag_embed(d.squeeze(-2))
            # L^T D is a column scaling: done elementwise (no GEMM, so no TF32 rounding); the product with the
            # previous R goes through _mm so that it is 3xTF32-compensated like Q (it was a plain @ before 0.5.1)
            F = torch.where(ok, L.mT * d, D)
            R = F if R is None else _mm(F, R, _tf32_active(Q))
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


# ---------------------------------------------------------------------------------------------------------------------
# A-posteriori orthogonality certificate (verify=True). Proof: see README "Certified orthogonality" and the theory note.
# ---------------------------------------------------------------------------------------------------------------------
# Dtypes whose every finite value is also an FP32 value (so it is exact in FP64 and products of two entries are exact
# FP64 numbers that never underflow: every nonzero product is a multiple of 2^-298 >= 2^-1022).
_FP32_SUBSET = tuple(d for d in (torch.float32, torch.float16, torch.bfloat16,
                                 getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None),
                                 getattr(torch, "float8_e4m3fnuz", None), getattr(torch, "float8_e5m2fnuz", None))
                     if d is not None)


def _round_up(x: Fraction) -> float:
    """Smallest-or-next float >= x (x > 0)."""
    f = float(x)
    if Fraction(f) < x:
        f = math.nextafter(f, math.inf)
    return f


@functools.lru_cache(maxsize=256)
def _cert_constants(m: int, l: int):
    """(alpha, beta) as floats rounded up, such that  ||Q^T Q - I||_F <= fl(fl(alpha * e) + fl(beta * t))  where
    e = computed ||fl(G - I)||_F and t = computed tr(G), G = fl64(Q^T Q).  u = 2^-53, gamma_k = k u / (1 - k u):
      alpha = 1 / ((1 - u)^4 (1 - gamma_{l^2}))       [norm of fl(G - I) (1 - u), sum of squares, sqrt (1 - u),
                                                       final multiply-add (1 - u)^2; sqrt(1 - g) >= 1 - g used]
      beta  = gamma_m / (1 - gamma_m) / ((1 - gamma_l) (1 - u)^2)   [Gram rounding, trace sum, final multiply-add]."""
    u = Fraction(1, 2 ** 53)
    if not (2 * m * u < 1 and 2 * l * l * u < 1):                    # gamma_m, gamma_{l^2} < 1
        raise ValueError("cholqr: matrix too large for the FP64 certificate")
    g = lambda k: k * u / (1 - k * u)
    alpha = 1 / ((1 - u) ** 4 * (1 - g(l * l)))
    beta = g(m) / (1 - g(m)) / ((1 - g(l)) * (1 - u) ** 2)
    return _round_up(alpha), _round_up(beta)


_CUBLAS_EMULATION_ENV = ("CUBLAS_EMULATE_DOUBLE_PRECISION", "CUBLAS_EMULATION_STRATEGY")


def _check_device(Q: torch.Tensor) -> torch.device:
    """Device on which the FP64 check runs: CPU, and NVIDIA CUDA in place (cuBLAS DGEMM); everything else on the CPU:
    MPS (no FP64) and ROCm (device type 'cuda' too, but rocBLAS/hipBLASLt are excluded from the assumptions)."""
    if Q.device.type == "cpu":
        return Q.device
    if Q.device.type == "cuda" and not getattr(torch.version, "hip", None):
        return Q.device
    return torch.device("cpu")


def _fp64_emulation_requested(dev: torch.device) -> bool:
    """True if cuBLAS may emulate FP64 GEMMs (Ozaki scheme, cuBLAS 13.0u2+), requested through the environment.
    The standard-model assumption then does not hold, so no matrix is certified."""
    if dev.type != "cuda":
        return False
    return any(os.environ.get(k, "").strip() not in ("", "0") for k in _CUBLAS_EMULATION_ENV)


_INT_VIEW = {8: torch.int64, 4: torch.int32, 2: torch.int16, 1: torch.uint8}


def _fp32_values(Qd: torch.Tensor):
    """(Qf, ok): Qf = Qd as float32; ok (per matrix) iff the conversion is exact bit for bit and no entry of Qf is an
    FP32 subnormal. Bit views are immune to flush-to-zero / denormals-are-zero modes, which could otherwise turn a
    subnormal into 0 in the conversion to FP64 (then the matrix checked would differ from the one returned)."""
    Qf = Qd.float()
    ok = torch.ones(Qd.shape[:-2], dtype=torch.bool, device=Qd.device)
    if Qd.numel() == 0:
        return Qf, ok
    if Qd.dtype != torch.float32:
        it = _INT_VIEW[Qd.element_size()]
        back = Qf.to(Qd.dtype).contiguous().view(it)
        ok = ok & (back == Qd.contiguous().view(it)).flatten(-2).all(-1)
    b = Qf.contiguous().view(torch.int32)
    sub = ((b & 0x7F800000) == 0) & ((b & 0x007FFFFF) != 0)
    return Qf, ok & ~sub.flatten(-2).any(-1)


def orthogonality_bound(Q: torch.Tensor) -> torch.Tensor:
    """Rigorous upper bound on ``||Q^T Q - I||_F`` for the exact values of a (batch of) matrix Q (..., m, r).

    Returns a float64 tensor of shape ``Q.shape[:-2]`` (on Q's device for CPU/CUDA, on the CPU otherwise) with
    ``||Q^T Q - I||_F <= bound`` for the exact real matrix Q^T Q (not its computed value), provided the FP64 arithmetic
    satisfies the IEEE standard model (round to nearest; matmul = sums of products in any order and blocking, with or
    without FMA, no Strassen-type algorithm and no reduced-precision emulation of FP64). ``inf`` for matrices with
    non-finite entries, for entries that are not FP32 values or are FP32 subnormals (the no-underflow argument needs
    normal-or-zero FP32 values; every Q returned by ``cholesky_qr`` passes through FP32), and for all matrices when
    cuBLAS FP64 emulation is requested by environment variable. CUDA tensors are checked with cuBLAS DGEMM, CPU, MPS
    and ROCm tensors with the CPU BLAS (assumed IEEE and non-Strassen; not verified, e.g. for macOS Accelerate).
    Cost: one FP64 Gram product. Not for use while compiling or tracing.

    bound = fl(alpha * e + beta * t), e = computed ||fl(G - I)||_F, t = computed tr(G), G = fl64(Q^T Q),
    alpha = 1 + (r^2 + 4) u + O(u^2), beta = m u (1 + O((m + r) u)), u = 2^-53; i.e. the computed error plus about
    m u tr(G) ~ 1.3e-10 for m = 4096, r = 276 with unit columns.
    """
    if _is_compiling() or _jit_tracing():
        raise RuntimeError("cholqr.orthogonality_bound cannot run while compiling or tracing (the compiler may change "
                           "the arithmetic the bound relies on); call it outside the compiled/traced region")
    if Q.dim() < 2:
        raise ValueError("Q must have at least 2 dimensions (..., m, r)")
    if Q.dtype != torch.float64 and Q.dtype not in _FP32_SUBSET:
        raise TypeError(f"orthogonality_bound supports float64/float32/float16/bfloat16/float8 tensors, got {Q.dtype}")
    m, l = Q.shape[-2], Q.shape[-1]
    alpha, beta = _cert_constants(m, l)
    dev = _check_device(Q)
    emulated = _fp64_emulation_requested(dev)
    if emulated:
        warnings.warn("cholqr: CUBLAS_EMULATE_DOUBLE_PRECISION / CUBLAS_EMULATION_STRATEGY is set, so cuBLAS may emulate "
                      "FP64 GEMMs; the certificate's arithmetic assumption does not hold and no matrix is certified "
                      "(bound = inf)", RuntimeWarning, stacklevel=2)
    with torch.no_grad():
        Qf, on_grid = _fp32_values(Q.detach().to(dev))
        Qd = Qf.double()
        finite = torch.isfinite(Qd).flatten(-2).all(-1) if l and m else torch.ones(
            Qd.shape[:-2], dtype=torch.bool, device=Qd.device)
        G = Qd.mT @ Qd
        t = G.diagonal(dim1=-2, dim2=-1).sum(-1)
        T = G - torch.eye(l, dtype=torch.float64, device=Qd.device)       # off-diagonal: exact; diagonal: one rounding
        e = (T * T).sum(dim=(-2, -1)).sqrt()
        v = alpha * e + beta * t
        good = finite & on_grid
        if emulated:
            good = torch.zeros_like(good)
        return torch.where(good, v, torch.full_like(v, math.inf))


def _resolve_tol(Y, out_dtype, tol):
    """Default tol: 1e-4 for float32/float64 output; low-precision outputs need an explicit tol."""
    dt = Y.dtype if out_dtype is None else out_dtype
    if tol is None:
        if dt in (torch.float32, torch.float64):
            return 1e-4
        r = Y.shape[-1]
        ud = torch.finfo(dt).eps / 2
        hint = 1e-4 + 2 * math.sqrt(r) * ud + r * ud * ud
        raise ValueError(f"cholqr: verify=True with Q returned in {dt} needs an explicit tol: rounding Q to {dt} alone "
                         f"moves ||Q^T Q - I||_F by up to ~2 sqrt(r) u_d (u_d = {ud:g}). Use out_dtype=torch.float32, "
                         f"or pass tol >~ 1e-4 + 2 sqrt(r) u_d + r u_d^2 = {hint:.3g} for r = {r}")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not (0 < tol < math.inf):
        raise ValueError(f"tol must be a positive finite number, got {tol!r}")
    return float(tol)


def _check_verify_call(Y, on_breakdown, tol):
    if on_breakdown == "ignore":
        raise ValueError("verify=True needs on_breakdown in ('fallback', 'warn', 'raise'); for a bound without any "
                         "action call cholqr.orthogonality_bound(Q)")
    if _jit_tracing():
        raise RuntimeError("cholqr: verify=True is not available under torch.jit.trace (the data-dependent fallback "
                           "cannot be traced). Trace with verify=False and check the result with "
                           "cholqr.orthogonality_bound(Q)")
    if _is_compiling() or _in_functorch(Y):
        raise RuntimeError("cholqr: verify=True is not available while compiling (torch.compile/export) or inside "
                           "torch.func transforms (vmap, grad): the fallback is data-dependent. Call cholesky_qr "
                           "with verify=False there and check the result outside with cholqr.orthogonality_bound(Q)")


def _certified(Q, tol):
    return orthogonality_bound(Q) <= tol


def _householder_hp(Y, return_r, out_dtype):
    """Householder QR in FP64 where the device has FP64 (FP32 otherwise), normalized so that diag(R) >= 0. Results pass
    through FP32, like every cholqr output."""
    wd = torch.float64 if Y.device.type != "mps" else torch.float32
    Q, R = torch.linalg.qr(Y.to(wd), mode="reduced")
    sgn = torch.where(R.diagonal(dim1=-2, dim2=-1) < 0, -1.0, 1.0).to(wd)
    dt = Y.dtype if out_dtype is None else out_dtype
    Q = (Q * sgn[..., None, :]).float().to(dt)
    return Q, ((R * sgn[..., :, None]).float().to(dt) if return_r else None)


def _verify_message(n_bad, n_all, Q, tol, worst, extra=""):
    msg = (f"cholqr: {n_bad} of {n_all} matrices could not be certified to ||Q^T Q - I||_F <= {tol:g} "
           f"(largest bound {worst:.3g}){extra}")
    if Q.dtype not in (torch.float32, torch.float64):
        msg += (f"; Q is returned in {Q.dtype}, whose rounding alone moves the error by ~ sqrt(r) * eps "
                f"(use out_dtype=torch.float32 or a larger tol)")
    return msg


def _verified_qr(Y, method, robust, return_r, out_dtype, shift, passes, oversample, generator, on_breakdown, tol):
    """verify=True path. method=None means the 'auto' Householder path."""
    m, r = Y.shape[-2], Y.shape[-1]
    if method is None:
        Q, R = _householder(Y, return_r, out_dtype)
    else:
        Q, R, _ = _cqr(Y, method, robust, return_r, out_dtype, shift, passes, oversample, generator)
    bound = orthogonality_bound(Q)
    ok = bound <= tol
    if bool(ok.all()):
        return (Q, R) if return_r else Q
    n_bad, n_all = int((~ok).sum()), ok.numel()
    if on_breakdown in ("raise", "warn"):
        msg = _verify_message(n_bad, n_all, Q, tol, float(bound[~ok].max()))
        if on_breakdown == "raise":
            raise torch.linalg.LinAlgError(msg)
        warnings.warn(msg + "; returned unrepaired", RuntimeWarning, stacklevel=3)
        return (Q, R) if return_r else Q
    # fallback: recompute the uncertified matrices stage by stage; keep a stage's result only where it is certified
    stages = []
    if method is not None and Y.device.type != "mps" and not (method == "scqr3" and robust):
        stages.append(lambda Ys: _cqr(Ys, "scqr3", True, return_r, out_dtype, shift, None, oversample, generator)[:2])
    stages.append(lambda Ys: _householder_hp(Ys, return_r, out_dtype))
    shape, dtype = Q.shape, Q.dtype
    Qf = Q.float().reshape(-1, m, r)                     # merge in FP32 (exact: every value passes through FP32)
    Rf = R.float().reshape(-1, r, r) if return_r else None
    pending = (~ok).reshape(-1).nonzero().squeeze(1).to(Y.device)
    Yf = Y.reshape(-1, m, r)
    for k, stage in enumerate(stages):
        Qs, Rs = stage(Yf[pending])
        oks = _certified(Qs, tol).to(Y.device)
        if k == len(stages) - 1:
            oks = torch.ones_like(oks)                   # last stage: keep everything (the final check decides)
        take = oks.nonzero().squeeze(1)
        if take.numel():
            Qf = Qf.index_copy(0, pending[take], Qs[take].float())
            if return_r:
                Rf = Rf.index_copy(0, pending[take], Rs[take].float())
        pending = pending[~oks]
        if pending.numel() == 0:
            break
    Q = Qf.reshape(shape).to(dtype)
    R = Rf.reshape(R.shape).to(R.dtype) if return_r else None
    bound = orthogonality_bound(Q)                       # certify exactly the tensor that is returned
    ok = bound <= tol
    if not bool(ok.all()):
        raise torch.linalg.LinAlgError(_verify_message(int((~ok).sum()), ok.numel(), Q, tol, float(bound[~ok].max()),
                                                       " even after the fallback (Householder QR)"))
    return (Q, R) if return_r else Q
