import math
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
    # documents the limitation: plain FP32 loses orthogonality for cond ~ 1e7
    Y = make_spectrum(3072, 210, 1e7, "geometric")
    assert orth_err(cholesky_qr(Y)) > 1.0


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
    assert orth_err(cholesky_qr(Y, method="cqr2")) > 1.0


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


def test_auto_compiles():
    torch._dynamo.reset()
    for shape in [(500, 20), (3, 500, 20)]:
        f = torch.compile(lambda Y: cholesky_qr(Y, method="auto"), fullgraph=True, dynamic=False)
        assert orth_err(f(torch.randn(*shape))) < 5e-5
