import math
import shutil
import sys
import pytest
import torch
from cholqr import cholesky_qr


def make_spectrum(m, r, kappa, kind, seed=0):
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(m, r, generator=g, dtype=torch.float64))[0]
    V = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0]
    if kind == "geometric":
        s = torch.logspace(0, -math.log10(kappa), r, dtype=torch.float64)
    else:
        s = torch.arange(1, r + 1, dtype=torch.float64) ** -(math.log(kappa) / math.log(r))
    return ((U * s) @ V.mT).float()


def orth_err(Q):
    Q = Q.double()
    return (Q.mT @ Q - torch.eye(Q.shape[-1], dtype=torch.float64)).norm(dim=(-2, -1)).max().item()


def subspace_err(Q, Y):
    Qr = torch.linalg.qr(Y.double())[0]
    return torch.linalg.matrix_norm(Q.double() @ Q.double().mT - Qr @ Qr.mT, ord=2).max().item()


@pytest.mark.parametrize("kind", ["geometric", "powerlaw"])
@pytest.mark.parametrize("kappa", [1e1, 1e3, 1e5, 1e6])
def test_default_fp32_up_to_1e6(kind, kappa):
    Y = make_spectrum(3072, 210, kappa, kind)
    Q = cholesky_qr(Y)
    assert orth_err(Q) < 5e-5
    assert subspace_err(Q, Y) < 1e-2 * max(1.0, kappa / 1e5)


@pytest.mark.parametrize("kind", ["geometric", "powerlaw"])
@pytest.mark.parametrize("kappa", [1e6, 1e7, 1e8])
def test_robust_up_to_1e8(kind, kappa):
    Y = make_spectrum(3072, 210, kappa, kind)
    Q = cholesky_qr(Y, robust=True)
    assert orth_err(Q) < 5e-5
    assert subspace_err(Q, Y) < 1e-4


def test_default_breaks_down_beyond_range():
    # documents the limitation: plain FP32 loses orthogonality beyond its range (visible with on_breakdown="ignore").
    # The exact onset is platform-dependent: cond 1e7 breaks on Apple M4 CPU but not on x86 Linux (RTX 4090 pod,
    # 2026-11-12); 1e8 breaks on both.
    Y = make_spectrum(3072, 210, 1e8, "geometric")
    assert orth_err(cholesky_qr(Y, on_breakdown="ignore")) > 1.0


@pytest.mark.parametrize("method", ["scqr3", "cqr2", "rcqr"])
@pytest.mark.parametrize("kappa", [1e7, 1e9])
def test_breakdown_is_repaired_by_default(method, kappa):
    Y = make_spectrum(3072, 210, kappa, "geometric")
    Q = cholesky_qr(Y, method=method)
    # rcqr does not break down here; with a single CholeskyQR pass it only loses some orthogonality (3e-4 at 1e9)
    assert orth_err(Q) < (1e-3 if method == "rcqr" else 5e-5)
    assert subspace_err(Q, Y) < 1e-2 * max(1.0, kappa / 1e5)


def test_breakdown_warn_and_raise():
    Y = make_spectrum(3072, 210, 1e8, "geometric")
    with pytest.warns(RuntimeWarning, match="broke down for 1 of 1"):
        Q = cholesky_qr(Y, on_breakdown="warn")
    assert orth_err(Q) > 1.0
    with pytest.raises(torch.linalg.LinAlgError):
        cholesky_qr(Y, on_breakdown="raise")
    with pytest.raises(ValueError):
        cholesky_qr(Y, on_breakdown="nope")


def test_no_false_alarm_in_range():
    Y = torch.stack([make_spectrum(2048, 128, k, "geometric", seed=i) for i, k in enumerate([1e1, 1e3, 1e5, 1e6])])
    for method in ("scqr3", "rcqr"):
        cholesky_qr(Y, method=method, on_breakdown="raise")
    cholesky_qr(Y, robust=True, on_breakdown="raise")


def test_fallback_repairs_only_flagged_batch_members():
    Y = torch.stack([make_spectrum(2048, 128, k, "geometric", seed=i) for i, k in enumerate([1e3, 1e8, 1e2, 1e9])])
    Q_ign = cholesky_qr(Y, on_breakdown="ignore")
    Q, R = cholesky_qr(Y, return_r=True)
    assert orth_err(Q) < 5e-5
    assert torch.equal(Q[0], Q_ign[0]) and torch.equal(Q[2], Q_ign[2])   # untouched
    rel = ((Q.double() @ R.double() - Y.double()).norm(dim=(-2, -1)) / Y.double().norm(dim=(-2, -1))).max()
    assert rel < 1e-5


@pytest.mark.parametrize("method", ["scqr3", "cqr2", "rcqr"])
def test_rank_deficient_inputs_get_orthonormal_q(method):
    g = torch.Generator().manual_seed(0)
    zero_col = torch.cat([torch.randn(256, 15, generator=g), torch.zeros(256, 1)], 1)
    low_rank = torch.randn(256, 8, generator=g) @ torch.randn(8, 16, generator=g)
    for Y in (zero_col, low_rank, torch.zeros(256, 16)):
        Q, R = cholesky_qr(Y, method=method, return_r=True)
        assert orth_err(Q) < 1e-4
        assert (Q @ R - Y).norm() <= 1e-5 * max(1.0, Y.norm().item())


@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_extreme_scales(scale):
    g = torch.Generator().manual_seed(0)
    Y = torch.randn(512, 32, generator=g) * scale
    for method in ("scqr3", "cqr2", "rcqr"):
        assert orth_err(cholesky_qr(Y, method=method, on_breakdown="ignore")) < 1e-4


def test_batched_matches_unbatched():
    Y = torch.stack([make_spectrum(768, 64, 1e3, "powerlaw", seed=k) for k in range(4)])
    Qb = cholesky_qr(Y)
    for k in range(4):
        assert torch.allclose(Qb[k], cholesky_qr(Y[k]), atol=1e-5)


def test_degenerate_inputs_are_finite():
    Z = torch.zeros(64, 8)
    assert torch.isfinite(cholesky_qr(Z)).all()
    Y = torch.randn(64, 4) @ torch.randn(4, 8)                       # exactly rank 4 < 8
    assert torch.isfinite(cholesky_qr(Y)).all()


def test_dtypes():
    Y = torch.randn(128, 16)
    assert cholesky_qr(Y.bfloat16()).dtype == torch.bfloat16
    assert cholesky_qr(Y, out_dtype=torch.float64).dtype == torch.float64


def test_rejects_wide_input():
    with pytest.raises(ValueError):
        cholesky_qr(torch.randn(8, 16))


@pytest.mark.parametrize("robust", [False, True])
def test_compiles_without_graph_breaks(robust):
    explain = torch._dynamo.explain(cholesky_qr)(torch.randn(256, 32), robust=robust)
    assert explain.graph_break_count == 0


def test_rejects_complex():
    with pytest.raises(TypeError):
        cholesky_qr(torch.randn(64, 8, dtype=torch.complex64))


def test_autograd_matches_householder():
    torch.manual_seed(0)
    Y = torch.randn(64, 8, dtype=torch.float64, requires_grad=True)
    W = torch.randn(64, 64, dtype=torch.float64)
    loss = lambda Q: ((Q @ Q.mT) * W).sum()      # sign-invariant
    g1, = torch.autograd.grad(loss(cholesky_qr(Y, out_dtype=torch.float64)), Y)
    g2, = torch.autograd.grad(loss(torch.linalg.qr(Y)[0]), Y)
    assert ((g1 - g2).norm() / g2.norm()).item() < 1e-5


@pytest.mark.parametrize("kind", ["geometric", "powerlaw"])
@pytest.mark.parametrize("kappa", [7e1, 1e3, 4e3])
def test_cqr2_well_conditioned(kind, kappa):
    Y = make_spectrum(3072, 210, kappa, kind)
    assert orth_err(cholesky_qr(Y, method="cqr2")) < 5e-5


def test_cqr2_breaks_down_beyond_range():
    # With the chunked Gram (v0.4) fp32 cqr2 survives cond ~ 1e4 on CPU and breaks between 1e4 and 3e4.
    Y = make_spectrum(3072, 210, 1e5, "geometric")
    assert orth_err(cholesky_qr(Y, method="cqr2", on_breakdown="ignore")) > 1.0


@pytest.mark.parametrize("method,robust,kappa", [("scqr3", False, 1e5), ("scqr3", True, 1e7), ("cqr2", False, 1e3)])
def test_return_r_reconstructs(method, robust, kappa):
    Y = make_spectrum(1024, 64, kappa, "powerlaw")
    Q, R = cholesky_qr(Y, method=method, robust=robust, return_r=True)
    rel = ((Q.double() @ R.double() - Y.double()).norm() / Y.double().norm()).item()
    assert rel < 1e-5
    assert torch.allclose(R.double(), torch.triu(R.double()), atol=1e-6 * R.double().abs().max().item())
    assert orth_err(Q) < 5e-5


def test_return_r_batched_and_dtype():
    Y = torch.stack([make_spectrum(256, 16, 1e2, "geometric", seed=k) for k in range(3)])
    Q, R = cholesky_qr(Y, return_r=True)
    assert Q.shape == (3, 256, 16) and R.shape == (3, 16, 16)
    assert torch.allclose(Q @ R, Y, atol=1e-4 * Y.abs().max().item())
    Qb, Rb = cholesky_qr(Y.bfloat16(), return_r=True)
    assert Qb.dtype == Rb.dtype == torch.bfloat16


def test_invalid_method_and_robust_combo():
    with pytest.raises(ValueError):
        cholesky_qr(torch.randn(64, 8), method="householder")
    with pytest.raises(ValueError):
        cholesky_qr(torch.randn(64, 8), method="cqr2", robust=True)


@pytest.mark.parametrize("kw", [dict(method="cqr2"), dict(return_r=True)])
def test_compiles_without_graph_breaks_v02(kw):
    explain = torch._dynamo.explain(cholesky_qr)(torch.randn(256, 32), **kw)
    assert explain.graph_break_count == 0


# ---- rcqr (randomized Cholesky QR) ----
@pytest.mark.parametrize("kind", ["geometric", "powerlaw"])
@pytest.mark.parametrize("kappa", [1e1, 1e3, 1e5, 1e6])
def test_rcqr_fp32(kind, kappa):
    Y = make_spectrum(3072, 210, kappa, kind)
    Q = cholesky_qr(Y, method="rcqr", generator=torch.Generator().manual_seed(1))
    assert orth_err(Q) < 5e-5
    assert subspace_err(Q, Y) < 1e-2 * max(1.0, kappa / 1e5)


@pytest.mark.parametrize("kappa", [1e1, 1e5])
def test_rcqr_return_r(kappa):
    Y = make_spectrum(3072, 210, kappa, "powerlaw")
    Q, R = cholesky_qr(Y, method="rcqr", return_r=True)
    assert torch.allclose(R, R.triu())
    rel = ((Q.double() @ R.double() - Y.double()).norm() / Y.double().norm()).item()
    assert rel < 1e-5


def test_rcqr_batched_and_degenerate():
    Y = torch.randn(5, 600, 40)
    assert orth_err(cholesky_qr(Y, method="rcqr")) < 5e-5
    Z = torch.zeros(2, 300, 20)
    assert torch.isfinite(cholesky_qr(Z, method="rcqr")).all()
    small = torch.randn(100, 90)                       # k is capped at m
    assert orth_err(cholesky_qr(small, method="rcqr")) < 5e-5


_TORCH = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
# Inductor uses MSVC (cl) on Windows and g++/clang++ elsewhere
_HAS_CXX = bool(shutil.which("cl")) if sys.platform == "win32" else any(shutil.which(c) for c in ("g++", "clang++", "c++"))


@pytest.mark.skipif(not _HAS_CXX, reason="Inductor needs a C++ compiler on CPU (none on PATH)")
@pytest.mark.skipif(_TORCH < (2, 4), reason="torch < 2.4: Inductor fails (2.1) or hangs (2.3) compiling the rcqr sketch")
def test_rcqr_compiles():
    torch._dynamo.reset()
    f = torch.compile(lambda Y: cholesky_qr(Y, method="rcqr"), fullgraph=True)
    assert orth_err(f(torch.randn(3, 500, 30))) < 5e-5


def test_rcqr_rejects_robust():
    with pytest.raises(ValueError):
        cholesky_qr(torch.randn(100, 10), method="rcqr", robust=True)


# ---- method="auto" ----
def test_auto_single_small_uses_householder_and_is_accurate():
    Y = make_spectrum(3072, 32, 1e8, "powerlaw")              # Householder handles any conditioning
    Q, R = cholesky_qr(Y, method="auto", return_r=True)
    assert orth_err(Q) < 1e-5
    assert (R.diagonal() >= 0).all() and torch.allclose(R, R.triu())
    assert ((Q.double() @ R.double() - Y.double()).norm() / Y.double().norm()).item() < 1e-5


def test_auto_batched_matches_scqr3():
    Y = torch.randn(4, 500, 30)
    assert torch.allclose(cholesky_qr(Y, method="auto"), cholesky_qr(Y), atol=0, rtol=0)
    Y1 = torch.randn(1, 600, 100)                            # r > 64 -> scqr3 even for a single matrix
    assert torch.allclose(cholesky_qr(Y1, method="auto"), cholesky_qr(Y1), atol=0, rtol=0)


def test_auto_half_precision_and_robust():
    Y = torch.randn(700, 20).bfloat16()
    Q = cholesky_qr(Y, method="auto")
    assert Q.dtype == torch.bfloat16 and orth_err(Q.float()) < 2e-2
    assert orth_err(cholesky_qr(torch.randn(3, 400, 20), method="auto", robust=True)) < 5e-5


@pytest.mark.skipif(not _HAS_CXX, reason="Inductor needs a C++ compiler on CPU (none on PATH)")
def test_auto_compiles():
    torch._dynamo.reset()
    for shape in [(500, 20), (3, 500, 20)]:
        f = torch.compile(lambda Y: cholesky_qr(Y, method="auto"), fullgraph=True, dynamic=False)
        assert orth_err(f(torch.randn(*shape))) < 5e-5


def test_passes_zero_return_r():
    Y = torch.randn(64, 8)
    Q, R = cholesky_qr(Y, passes=0, return_r=True)
    assert torch.equal(Q, Y) and torch.equal(R, torch.eye(8))


def test_overflowing_column_norm_gives_orthonormal_q():
    Y = (torch.rand(512, 32, generator=torch.Generator().manual_seed(0)) * 2 - 1) * 3e37
    for method in ("scqr3", "cqr2", "rcqr"):
        assert orth_err(cholesky_qr(Y, method=method, on_breakdown="ignore")) < 1e-4


def test_cqr2_gradual_loss_is_flagged_and_repaired():
    # linear spectrum near cqr2's limit: all Cholesky factorizations succeed but orthogonality degrades gradually
    g = torch.Generator().manual_seed(3)
    U = torch.linalg.qr(torch.randn(1024, 64, generator=g, dtype=torch.float64))[0]
    V = torch.linalg.qr(torch.randn(64, 64, generator=g, dtype=torch.float64))[0]
    bad = 0
    for kappa in (3e6, 1e7, 3e7, 1e8):
        Y = ((U * torch.linspace(1, 1 / kappa, 64, dtype=torch.float64)) @ V.mT).float()
        bad += orth_err(cholesky_qr(Y, method="cqr2", on_breakdown="ignore")) > 1e-4
        assert orth_err(cholesky_qr(Y, method="cqr2")) < 1e-4
    assert bad >= 1      # the case actually occurs (otherwise the test is vacuous)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float8_e4m3fn"])
def test_low_precision_inputs_with_fallback(dtype):
    dt = getattr(torch, dtype, None)
    if dt is None:
        pytest.skip(f"{dtype} not available")
    g = torch.Generator().manual_seed(0)
    Y = torch.randn(256, 16, generator=g)
    Y[:, 3] = Y[:, 2]                                  # exactly rank deficient -> fallback path
    Q = cholesky_qr((Y / Y.abs().max()).to(dt), out_dtype=torch.float32)
    assert orth_err(Q) < 1e-4
    Q, R = cholesky_qr((Y / Y.abs().max()).to(dt), return_r=True)      # default out dtype = input dtype
    assert Q.dtype == dt and R.dtype == dt
