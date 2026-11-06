"""Self-contained repro: PowerSGD orthogonalization vs. CholeskyQR2 (no third-party deps beyond torch).

Usage: python repro_powersgd_orth.py   (CUDA GPU recommended; runs on CPU too)
"""
import time

import torch
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as psh


def cholesky_qr2_(matrices: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
    """In-place orthogonalization of a (b, n, r) batch via CholeskyQR2, computed in FP32.

    Falls back to the existing routine for any batch whose Cholesky factorization fails
    (rank-deficient or too ill-conditioned input).
    """
    X = matrices.float()
    ok = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
    for _ in range(2):
        X = X / (X.norm(dim=1, keepdim=True) + epsilon).clamp_min(1e-30)   # column scaling
        L, info = torch.linalg.cholesky_ex(X.mT @ X)
        ok &= info == 0
        X = torch.linalg.solve_triangular(L, X.mT, upper=False).mT     # X <- X L^{-T}
        X = torch.where(ok[:, None, None], X, matrices.float())         # keep failed ones finite
    if not bool(ok.all()):                                               # rare path: host sync
        bad = ~ok
        Y = matrices[bad].clone()
        psh._orthogonalize(Y, epsilon) if Y.dtype == torch.float32 else psh._orthogonalize_gram_schmidt(Y, epsilon)
        X[bad] = Y.float()
    matrices.copy_(X)
    return matrices


def timeit(f, n=50):
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for _ in range(5):
        f()
    sync(); t = time.perf_counter()
    for _ in range(n):
        f()
    sync(); return (time.perf_counter() - t) / n * 1e3


def orth_err(Q):
    Q = Q.double()
    return (Q.mT @ Q - torch.eye(Q.shape[-1], dtype=Q.dtype, device=Q.device)).norm(dim=(-2, -1)).max().item()


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("highest")
    print(torch.__version__, torch.cuda.get_device_name() if dev == "cuda" else "cpu")
    print("b x n x r | dtype | current _orthogonalize | fp32 QR on upcast copy | CholeskyQR2 (ms) | orth current | orth CQR2")
    for dtype in (torch.float32, torch.bfloat16):
        for b, n, r in [(1, 768, 32), (1, 3072, 64), (12, 768, 16), (12, 3072, 32), (12, 3072, 64)]:
            X = torch.randn(b, n, r, device=dev).to(dtype)
            t0 = timeit(lambda: psh._orthogonalize(X.clone()))
            t1 = timeit(lambda: cholesky_qr2_(X.clone()))
            tu = timeit(lambda: psh._orthogonalize(X.float()))                  # upcast + existing fp32 path
            A = X.clone(); psh._orthogonalize(A); B = cholesky_qr2_(X.clone())
            print(f"{b}x{n}x{r} | {str(dtype)[6:]:8s} | {t0:7.3f} | {tu:7.3f} | {t1:7.3f} | {orth_err(A):.1e} | {orth_err(B):.1e}")
