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
