"""verify=True / orthogonality_bound: soundness against exact rational arithmetic, and API behaviour."""
import math
import warnings
from fractions import Fraction

import pytest
import torch

import cholqr
import cholqr.core as core
from cholqr import cholesky_qr, orthogonality_bound


# ---------- exact reference ------------------------------------------------------------------------------------------
def exact_err2(Q):
    """||Q^T Q - I||_F^2 of one matrix, exactly (Fraction). Every entry is an FP32 value: integer * 2^-149."""
    Q = Q.detach().cpu().float().double()
    m, l = Q.shape
    S = 149
    N = [[int(Fraction(float(Q[k, i])) * 2 ** S) for i in range(l)] for k in range(m)]
    cols = list(zip(*N))
    total = Fraction(0)
    one = 2 ** (2 * S)
    for i in range(l):
        for j in range(l):
            a = sum(x * y for x, y in zip(cols[i], cols[j])) - (one if i == j else 0)
            total += Fraction(a * a, one * one)
    return total


def exact_err(Q):
    return math.sqrt(exact_err2(Q))


def _spectrum(m, r, kappa, seed=0, kind="geometric"):
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(m, r, generator=g, dtype=torch.float64))[0]
    V = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0]
    if kind == "geometric":
        s = torch.logspace(0, -math.log10(kappa), r, dtype=torch.float64)
    else:
        s = torch.linspace(1, kappa ** 0.5, r, dtype=torch.float64) ** -2
    return ((U * s) @ V.mT).float()


def _tf32_round(X):
    b = X.contiguous().view(torch.int32)
    return ((b + 0x1000) & ~0x1FFF).view(torch.float32)


class _EmulatedTF32:
    """Replace cholqr's Gram/forming GEMMs by plain TF32 GEMMs (inputs rounded to TF32, FP64 accumulation): the 0.4
    behaviour under TF32, which degrades orthogonality without the empirical detector noticing."""
    def __enter__(self):
        self.mm, self.gmm = core._mm, core._gram_mm

        def tmm(A, B):
            if A.dtype != torch.float32:
                return A @ B
            return (_tf32_round(A).double() @ _tf32_round(B.contiguous()).double()).float()
        core._mm = lambda A, B, tf32: tmm(A, B)
        core._gram_mm = lambda A, tf32: tmm(A.mT.contiguous(), A)
        return self

    def __exit__(self, *a):
        core._mm, core._gram_mm = self.mm, self.gmm


def _assert_sound(Q, tols_extra=()):
    """For every tol in a sweep around the exact error: certified => exact error <= tol."""
    Qs = Q.reshape(-1, *Q.shape[-2:])
    bounds = orthogonality_bound(Qs)
    for k in range(Qs.shape[0]):
        ex2 = exact_err2(Qs[k])
        ex = math.sqrt(ex2)
        b = float(bounds[k])
        assert b >= ex or not math.isfinite(ex)          # the bound itself
        tols = [1e-4, *tols_extra] + [ex * (1 - 2.0 ** -j) for j in (1, 10, 30, 45, 52)] + \
               [ex * (1 + 2.0 ** -j) for j in (10, 30, 52)]
        for tol in tols:
            if 0 < tol < math.inf and b <= tol:
                assert ex2 <= Fraction(tol) ** 2, (k, tol, b, ex)


# ---------- soundness (V1 at test sizes) -----------------------------------------------------------------------------
@pytest.mark.parametrize("kappa", [1e0, 1e4, 1e6, 1e8, 1e12])
@pytest.mark.parametrize("method", ["scqr3", "cqr2", "rcqr"])
def test_bound_sound_on_cholqr_outputs(method, kappa):
    Y = _spectrum(160, 10, kappa, seed=int(math.log10(kappa)))
    Q = cholesky_qr(Y, method=method, on_breakdown="ignore")
    _assert_sound(Q)


def test_bound_sound_tf32_emulated():
    with _EmulatedTF32():
        Qs = [cholesky_qr(_spectrum(200, 12, k, seed=i), on_breakdown="ignore") for i, k in enumerate([1e2, 1e5, 1e7])]
    for Q in Qs:
        _assert_sound(Q)


def test_bound_sound_adversarial_direct():
    g = torch.Generator().manual_seed(3)
    Q0 = torch.linalg.qr(torch.randn(96, 8, generator=g, dtype=torch.float64))[0]
    cases = []
    for eps in (1e-4, 3e-5, 1e-7):                       # true error placed near tol
        E = torch.randn(96, 8, generator=g, dtype=torch.float64)
        cases.append((Q0 + eps / 2 * E / E.norm()).float())
    cases.append((Q0 * torch.tensor([2.0 ** 60] + [1.0] * 7, dtype=torch.float64)).float())     # huge column
    cases.append((Q0 * 2.0 ** -140).float())                                                    # fp32 subnormals
    Qd = Q0.float().clone(); Qd[:, 1] = Qd[:, 0]                                                # duplicated column
    cases.append(Qd)
    cases.append(torch.zeros(96, 8))
    for dt in (torch.float16, torch.bfloat16, getattr(torch, "float8_e4m3fn", torch.float16)):
        cases.append(Q0.to(dt))
    for Q in cases:
        _assert_sound(Q, tols_extra=(float(orthogonality_bound(Q)),))


def test_bound_sound_low_precision_inputs():
    Y = _spectrum(128, 8, 1e3, seed=5)
    dts = [torch.float16, torch.bfloat16] + [d for d in (getattr(torch, "float8_e4m3fn", None),
                                                          getattr(torch, "float8_e5m2", None)) if d is not None]
    for dt in dts:
        Q = cholesky_qr(Y.to(dt), on_breakdown="ignore", out_dtype=torch.float32)
        _assert_sound(Q)
        Q = cholesky_qr(Y.to(dt), on_breakdown="ignore")
        _assert_sound(Q)


def test_bound_near_rank_deficient_and_batched():
    Y = _spectrum(150, 9, 1e2, seed=7)
    Y2 = Y.clone(); Y2[:, 3] = Y2[:, 2] + 1e-7 * torch.randn(150, generator=torch.Generator().manual_seed(1))
    Y3 = Y.clone(); Y3[:, 0] = 0
    Q = cholesky_qr(torch.stack([Y, Y2, Y3]), on_breakdown="ignore")
    _assert_sound(Q)


def test_bound_nonfinite_and_off_grid():
    Q = torch.linalg.qr(torch.randn(64, 4, dtype=torch.float64))[0]
    assert math.isinf(float(orthogonality_bound(Q)))                       # float64 not on the FP32 grid
    assert float(orthogonality_bound(Q.float().double())) < 1e-6
    for bad in (math.nan, math.inf):
        Qb = Q.float().clone(); Qb[3, 1] = bad
        assert math.isinf(float(orthogonality_bound(Qb)))
        assert not bool(orthogonality_bound(Qb) <= 1e300)


def test_bound_close_to_computed_error():
    Q = cholesky_qr(torch.randn(4096, 64, generator=torch.Generator().manual_seed(0)))
    e = (Q.double().mT @ Q.double() - torch.eye(64, dtype=torch.float64)).norm()
    b = orthogonality_bound(Q)
    assert 0 <= float(b - e) < 64 * 4096 * 2.0 ** -53 * 1.01 + 1e-15        # slack ~ m u tr(G)


def test_constants():
    a, b = core._cert_constants(4096, 276)
    u = 2.0 ** -53
    assert 1 < a < 1 + (276 ** 2 + 8) * u
    assert 4096 * u < b < 4096 * u * (1 + 1e-9)


# ---------- API ------------------------------------------------------------------------------------------------------
def test_verify_default_path_identical_when_certified():
    Y = torch.randn(3, 512, 32, generator=torch.Generator().manual_seed(0))
    Q0, R0 = cholesky_qr(Y, return_r=True)
    Q1, R1 = cholesky_qr(Y, return_r=True, verify=True)
    assert torch.equal(Q0, Q1) and torch.equal(R0, R1)


def test_verify_fallback_repairs_and_certifies():
    Ys = torch.stack([_spectrum(256, 16, 1e12, seed=0), _spectrum(256, 16, 1e2, seed=1)])
    assert float(orthogonality_bound(cholesky_qr(Ys, on_breakdown="ignore"))[0]) > 1e-4
    Q, R = cholesky_qr(Ys, verify=True, return_r=True)
    assert bool((orthogonality_bound(Q) <= 1e-4).all())
    assert torch.equal(Q[1], cholesky_qr(Ys[1:], verify=True)[0])           # certified member untouched
    _assert_sound(Q)


def test_verify_catches_tf32_degradation_missed_by_detector():
    Y = _spectrum(512, 32, 3e5, seed=11)
    with _EmulatedTF32():
        Qd = cholesky_qr(Y)                                                   # empirical detector (default fallback)
        Qv = cholesky_qr(Y, verify=True)
    assert float(orthogonality_bound(Qv)) <= 1e-4
    _assert_sound(Qv)
    assert float(orthogonality_bound(Qd)) > float(orthogonality_bound(Qv))


def test_verify_raise_and_warn():
    Y = _spectrum(256, 16, 1e12, seed=0)
    with pytest.raises(torch.linalg.LinAlgError):
        cholesky_qr(Y, verify=True, on_breakdown="raise")
    with pytest.warns(RuntimeWarning, match="could not be certified"):
        Q = cholesky_qr(Y, verify=True, on_breakdown="warn")
    assert torch.equal(Q, cholesky_qr(Y, on_breakdown="ignore"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cholesky_qr(_spectrum(256, 16, 1e2), verify=True, on_breakdown="warn")   # certified: no warning


def test_verify_argument_checks():
    Y = torch.randn(64, 4)
    with pytest.raises(ValueError):
        cholesky_qr(Y, verify=True, on_breakdown="ignore")
    for tol in (0.0, -1.0, math.inf, math.nan, True, "1e-4"):
        with pytest.raises(ValueError):
            cholesky_qr(Y, verify=True, tol=tol)
    cholesky_qr(Y, on_breakdown="ignore", tol=-1.0)                          # tol is ignored without verify


def test_verify_low_precision_out_dtype():
    Y = torch.randn(512, 32, generator=torch.Generator().manual_seed(0))
    for dt in (torch.bfloat16, torch.float16):
        with pytest.raises(ValueError, match="explicit tol"):
            cholesky_qr(Y, verify=True, out_dtype=dt)                        # at call time, before any fallback
        with pytest.raises(ValueError, match="explicit tol"):
            cholesky_qr(Y.to(dt), verify=True)
    with pytest.raises(torch.linalg.LinAlgError, match="bfloat16"):
        cholesky_qr(Y, verify=True, out_dtype=torch.bfloat16, tol=1e-4)      # explicit tol: honoured, not relaxed
    Q = cholesky_qr(Y, verify=True, out_dtype=torch.bfloat16, tol=0.5)
    assert float(orthogonality_bound(Q)) <= 0.5
    Q = cholesky_qr(Y.half(), verify=True, out_dtype=torch.float32)
    assert float(orthogonality_bound(Q)) <= 1e-4


def test_verify_out_dtype_float64_and_auto_path():
    Y = _spectrum(300, 20, 1e3).double()
    Q = cholesky_qr(Y, verify=True)                                            # float64 output (FP32 values)
    assert Q.dtype == torch.float64 and float(orthogonality_bound(Q)) <= 1e-4
    Q, R = cholesky_qr(_spectrum(300, 20, 1e12), method="auto", verify=True, return_r=True)   # Householder path
    assert float(orthogonality_bound(Q)) <= 1e-4


def test_verify_return_r_after_fallback_reconstructs():
    Y = _spectrum(256, 16, 1e9, seed=2).double()
    Q, R = cholesky_qr(Y.float(), verify=True, return_r=True)
    assert float(orthogonality_bound(Q)) <= 1e-4
    assert float((Q.double() @ R.double() - Y).norm() / Y.norm()) < 1e-5


def test_verify_autograd_unchanged():
    Y = torch.randn(128, 8, generator=torch.Generator().manual_seed(0), requires_grad=True)
    W = torch.randn(128, 128, generator=torch.Generator().manual_seed(1))
    g0, = torch.autograd.grad(((cholesky_qr(Y) @ cholesky_qr(Y).mT) * W).sum(), Y)
    g1, = torch.autograd.grad(((cholesky_qr(Y, verify=True) @ cholesky_qr(Y, verify=True).mT) * W).sum(), Y)
    assert torch.equal(g0, g1)


def test_verify_under_compile_raises():
    torch._dynamo.reset()
    f = torch.compile(lambda y: cholesky_qr(y, verify=True), backend="eager")
    with pytest.raises(RuntimeError, match="verify=True is not available while compiling"):
        f(torch.randn(64, 4))
    torch._dynamo.reset()
    f = torch.compile(lambda y: cholesky_qr(y, verify=True), backend="eager", fullgraph=True)
    # with fullgraph=True some torch versions (e.g. 2.8) wrap our RuntimeError in Dynamo's own message
    with pytest.raises(RuntimeError, match="verify=True is not available while compiling|Observed exception"):
        f(torch.randn(64, 4))
    torch._dynamo.reset()


def test_verify_under_vmap_raises_and_bound_outside_works():
    Y = torch.stack([_spectrum(128, 8, 1e2, seed=k) for k in range(3)])
    with pytest.raises(RuntimeError, match="torch.func"):
        torch.func.vmap(lambda y: cholesky_qr(y, verify=True))(Y)
    Q = torch.func.vmap(lambda y: cholesky_qr(y))(Y)
    assert bool((orthogonality_bound(Q) <= 1e-4).all())


def test_verify_default_path_compile_unaffected():
    explain = torch._dynamo.explain(cholesky_qr)(torch.randn(256, 32))
    assert explain.graph_break_count == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_verify_mps():
    Ys = torch.stack([_spectrum(256, 16, 1e12, seed=0), _spectrum(256, 16, 1e2, seed=1)]).to("mps")
    Q = cholesky_qr(Ys, verify=True)
    assert Q.device.type == "mps"
    b = orthogonality_bound(Q)
    assert b.device.type == "cpu" and bool((b <= 1e-4).all())
    _assert_sound(Q.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_verify_cuda_under_tf32_and_highest():
    old = torch.get_float32_matmul_precision()
    try:
        for prec in ("highest", "high"):
            torch.set_float32_matmul_precision(prec)
            Ys = torch.stack([_spectrum(512, 32, k, seed=i) for i, k in enumerate([1e2, 1e6, 1e9, 1e12])]).cuda()
            Q = cholesky_qr(Ys, verify=True)
            b = orthogonality_bound(Q)
            assert b.device.type == "cuda" and bool((b <= 1e-4).all())
            for k in range(Q.shape[0]):
                assert exact_err(Q[k].cpu()) <= 1e-4
    finally:
        torch.set_float32_matmul_precision(old)


def test_verify_fallback_autograd_and_inference_mode():
    Yb = torch.stack([_spectrum(256, 16, 1e12, seed=0), torch.randn(256, 16)]).requires_grad_()
    Q = cholesky_qr(Yb, verify=True)
    Q.sum().backward()
    assert torch.isfinite(Yb.grad).all()
    with torch.inference_mode():
        assert bool((orthogonality_bound(cholesky_qr(Yb.detach(), verify=True)) <= 1e-4).all())


def test_size_check_needs_2_l2_u():
    with pytest.raises(ValueError):
        core._cert_constants(8, 2 ** 26)                                     # 2 l^2 u = 2^0: rejected
    core._cert_constants(8, 2 ** 25)                                         # 2 l^2 u = 2^-2: accepted


def test_subnormal_entries_not_certified_any_flush_mode():
    Q = torch.linalg.qr(torch.randn(64, 4, dtype=torch.float64))[0].float()
    Q[5, 2] = 1e-40                                                          # FP32 subnormal
    Qd = Q.double()                                                          # made before any flush mode
    old = torch.is_flush_denormal_enabled() if hasattr(torch, "is_flush_denormal_enabled") else False
    try:
        for flush in (False, True):
            torch.set_flush_denormal(flush)
            assert math.isinf(float(orthogonality_bound(Q)))
            assert math.isinf(float(orthogonality_bound(Qd)))
            Qh = Q.half(); Qh[5, 2] = 2.0 ** -20                             # fp16 subnormal (normal in FP32)
            b = float(orthogonality_bound(Qh))
            assert b >= exact_err(Qh) or math.isinf(b)                       # sound in either mode
    finally:
        torch.set_flush_denormal(old)
    assert math.isfinite(float(orthogonality_bound(Qh)))                     # without flushing: certifiable


def test_cublas_fp64_emulation_env(monkeypatch):
    cuda = torch.device("cuda")
    assert not core._fp64_emulation_requested(cuda)
    for k in core._CUBLAS_EMULATION_ENV:
        monkeypatch.setenv(k, "1")
        assert core._fp64_emulation_requested(cuda)
        assert not core._fp64_emulation_requested(torch.device("cpu"))     # CPU checks do not use cuBLAS
        monkeypatch.setenv(k, "0")
        assert not core._fp64_emulation_requested(cuda)
        monkeypatch.delenv(k)
    if torch.cuda.is_available() and not getattr(torch.version, "hip", None):
        monkeypatch.setenv("CUBLAS_EMULATION_STRATEGY", "eager")
        Q = cholesky_qr(torch.randn(256, 8, device="cuda"))
        with pytest.warns(RuntimeWarning, match="emulate"):
            assert math.isinf(float(orthogonality_bound(Q)))


def test_rocm_and_mps_checked_on_cpu():
    if getattr(torch.version, "hip", None) and torch.cuda.is_available():
        assert core._check_device(torch.zeros(2, 2, device="cuda")).type == "cpu"
    assert core._check_device(torch.zeros(2, 2)).type == "cpu"


@pytest.mark.skipif(not hasattr(torch.jit, "trace"), reason="torch.jit.trace removed in this torch")
def test_verify_under_jit_trace_raises():
    with pytest.raises(RuntimeError, match="jit.trace"):
        torch.jit.trace(lambda y: cholesky_qr(y, verify=True), torch.randn(64, 4))
    with pytest.raises(RuntimeError, match="tracing"):
        torch.jit.trace(lambda q: orthogonality_bound(q), torch.randn(64, 4))


def test_fallback_gradient_finite_when_backend_returns_nan_factor(monkeypatch):
    # some LAPACK backends return NaN entries in a failed Cholesky factor; the fallback merge must not leak 0 * NaN
    real = torch.linalg.cholesky_ex

    def nan_on_failure(A, **kw):
        L, info = real(A, **kw)
        bad = (info != 0)[..., None, None]
        return torch.where(bad, torch.full_like(L, float("nan")), L), info

    monkeypatch.setattr(torch.linalg, "cholesky_ex", nan_on_failure)
    Yb = torch.stack([_spectrum(256, 16, 1e12, seed=0), torch.randn(256, 16)]).requires_grad_()
    for kw in ({}, {"verify": True}):
        Yb.grad = None
        cholesky_qr(Yb, **kw).sum().backward()
        assert torch.isfinite(Yb.grad).all(), kw
