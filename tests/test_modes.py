"""Execution modes and unusual inputs (compatibility)."""
import pytest
import torch
from cholqr import cholesky_qr


def _ill(m=256, r=16, kappa=1e9, seed=0):
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(m, r, generator=g, dtype=torch.float64))[0]
    V = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0]
    s = torch.logspace(0, -torch.log10(torch.tensor(kappa)).item(), r, dtype=torch.float64)
    return ((U * s) @ V.mT).float()


def _orth(Q):
    Q = Q.double()
    return (Q.mT @ Q - torch.eye(Q.shape[-1], dtype=torch.float64)).norm(dim=(-2, -1)).max().item()


def test_deterministic_mode_with_fallback():
    old = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        Y = torch.stack([_ill(seed=0), _ill(kappa=1e2, seed=1)])         # one matrix needs the fallback
        Q = cholesky_qr(Y)
        assert _orth(Q) < 1e-4
    finally:
        torch.use_deterministic_algorithms(old)


def test_inference_mode_and_no_grad():
    Y = torch.stack([_ill(seed=0), _ill(kappa=1e2, seed=1)])
    with torch.inference_mode():
        assert _orth(cholesky_qr(Y)) < 1e-4
    with torch.no_grad():
        assert _orth(cholesky_qr(Y, return_r=True)[0]) < 1e-4


def test_vmap_runs():
    # data-dependent checks are skipped under functorch transforms (like torch.compile)
    Y = torch.stack([_ill(kappa=1e2, seed=k) for k in range(3)])
    Q = torch.func.vmap(lambda y: cholesky_qr(y))(Y)
    assert _orth(Q) < 1e-4


def test_double_backward_smoke():
    Y = _ill(m=64, r=4, kappa=10.0).double().requires_grad_()
    Q = cholesky_qr(Y, out_dtype=torch.float64)
    (g,) = torch.autograd.grad(Q.sum(), Y, create_graph=True)
    (h,) = torch.autograd.grad(g.sum(), Y)
    assert torch.isfinite(h).all()


def test_default_dtype_float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        Y = _ill(kappa=1e3).float()
        Q = cholesky_qr(Y)
        assert Q.dtype == torch.float32 and _orth(Q) < 1e-4
        assert _orth(cholesky_qr(_ill(kappa=1e9))) < 1e-4          # fallback path
    finally:
        torch.set_default_dtype(old)


@pytest.mark.parametrize("shape", [(0, 64, 8), (64, 1), (32, 32), (3, 1, 1)])
def test_shapes(shape):
    Y = torch.randn(*shape)
    Q = cholesky_qr(Y)
    assert Q.shape == Y.shape
    if Y.numel():
        assert _orth(Q) < 1e-4


def test_non_contiguous():
    Y = torch.randn(64, 256).mT                                      # (256, 64) non-contiguous
    assert not Y.is_contiguous()
    assert _orth(cholesky_qr(Y)) < 1e-4
    assert _orth(cholesky_qr(_ill(kappa=1e9).mT.contiguous().mT)) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_cuda_graph_capture_with_ignore():
    Y = torch.randn(4, 1024, 64, device="cuda")
    cholesky_qr(Y, on_breakdown="ignore")                            # warm up
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        Q = cholesky_qr(Y, on_breakdown="ignore")
    g.replay(); torch.cuda.synchronize()
    assert _orth(Q.cpu()) < 1e-4


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_second_gpu():
    Y = _ill(kappa=1e9).to("cuda:1")
    Q = cholesky_qr(Y)
    assert Q.device == Y.device and _orth(Q.cpu()) < 1e-4
