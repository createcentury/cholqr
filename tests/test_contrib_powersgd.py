import os
import pytest
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as psh
from cholqr.contrib import powersgd


def orth_err(Q):
    Q = Q.double()
    return (Q.mT @ Q - torch.eye(Q.shape[-1], dtype=torch.float64)).norm(dim=(-2, -1)).max().item()


@pytest.mark.parametrize("b", [1, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("r", [1, 2, 8, 32])
def test_orthogonalize_matches_span(b, dtype, r):
    X = torch.randn(b, 300, r).to(dtype)
    A, B = X.clone(), X.clone()
    powersgd.orthogonalize_(B)
    assert B.dtype == dtype
    tol = 5e-5 if dtype == torch.float32 else 3e-2
    assert orth_err(B.float()) < tol
    # same column span as the input: projecting X onto span(B) recovers X
    Xf, Bf = X.double(), B.double()
    Bq = torch.linalg.qr(Bf)[0]
    assert ((Xf - Bq @ (Bq.mT @ Xf)).norm() / Xf.norm()).item() < (1e-5 if dtype == torch.float32 else 3e-2)


def test_fallback_on_ill_conditioned_batch():
    X = torch.randn(3, 400, 16)
    X[1, :, -1] = X[1, :, 0] + 1e-9 * torch.randn(400)      # nearly rank deficient -> CQR2 fails, fallback
    B = X.clone(); powersgd.orthogonalize_(B)
    assert torch.isfinite(B).all()
    assert orth_err(B[[0, 2]]) < 5e-5


def test_patch_unpatch_and_ddp_hook_equivalence():
    pytest.importorskip("numpy")    # torch's PowerSGDState imports numpy
    import socket
    with socket.socket() as sk:                  # a free port, so parallel test runs on one host do not collide
        sk.bind(("127.0.0.1", 0)); port = sk.getsockname()[1]
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", str(port))
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    from torch.nn.parallel import DistributedDataParallel as DDP

    # upstream powerSGD_hook calls torch.cuda.synchronize(tensor.device) whenever CUDA is available, which raises for
    # CPU tensors (also unpatched; seen on an RTX 4090 pod 2026-11-12), so run on CUDA when it exists
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def grads(patched, batched):
        powersgd.patch() if patched else powersgd.unpatch()
        torch.manual_seed(0)
        model = DDP(torch.nn.Sequential(torch.nn.Linear(64, 96), torch.nn.ReLU(), torch.nn.Linear(96, 96),
                                        torch.nn.ReLU(), torch.nn.Linear(96, 8)).to(dev))
        st = psh.PowerSGDState(process_group=None, matrix_approximation_rank=4, start_powerSGD_iter=2,
                               min_compression_rate=0.5, batch_tensors_with_same_shape=batched, random_seed=0)
        model.register_comm_hook(st, psh.powerSGD_hook)
        x = torch.randn(32, 64).to(dev)
        for _ in range(4):
            model.zero_grad(); model(x).square().mean().backward()
        return [p.grad.detach().cpu().clone() for p in model.parameters()]

    try:
        for batched in (False, True):
            g0, g1 = grads(False, batched), grads(True, batched)
            for a, c in zip(g0, g1):
                assert torch.allclose(a, c, rtol=1e-3, atol=1e-5), (batched, (a - c).abs().max())
        powersgd.patch(); assert powersgd.is_patched()
    finally:
        powersgd.unpatch()
    assert not powersgd.is_patched()
