# [R14] cholqr: rcqr (randomized CholeskyQR, as in Dion) vs scqr3 / cqr2 / robust on GPU. FP32, matmul precision "highest".
import subprocess, sys, importlib, math, time, warnings
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
                "git+https://github.com/createcentury/cholqr@rcqr"], check=True)
import torch, cholqr, cholqr.core
importlib.reload(cholqr.core); importlib.reload(cholqr)
from cholqr import cholesky_qr
torch._dynamo.reset(); torch.set_float32_matmul_precision("highest"); dev = "cuda"
print(torch.cuda.get_device_name(), torch.__version__, "cholqr", cholqr.__version__)

def make(m, r, kappa, seed=0):
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(m, r, generator=g, dtype=torch.float64))[0]
    V = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0]
    s = torch.arange(1, r + 1, dtype=torch.float64) ** -(math.log(kappa) / math.log(r))
    return ((U * s) @ V.mT).float().to(dev)
orth = lambda Q: (Q.double().mT @ Q.double() - torch.eye(Q.shape[-1], device=dev, dtype=torch.float64)).norm().item()
def sub(Q, Y):
    Qr = torch.linalg.qr(Y.double())[0]
    return torch.linalg.matrix_norm(Q.double() @ Q.double().mT - Qr @ Qr.mT, ord=2).item()

# Dion's reference rcqr (microsoft/dion, dion_reference.orthogonalize, MIT), single matrix
def dion_rcqr(P, oversample=1.25):
    m, n = P.shape; k = math.ceil(oversample * n / 128.0) * 128
    S = torch.empty((k, m), device=P.device, dtype=P.dtype).normal_(std=math.sqrt(1.0 / k))
    _, R = torch.linalg.qr((S @ P).float(), mode="r")
    Q = torch.linalg.solve_triangular(R, P.float(), upper=True, left=False)
    R, _ = torch.linalg.cholesky_ex(Q.T @ Q, upper=True)
    return torch.linalg.solve_triangular(R, Q, upper=True, left=False)

M = {"scqr3": dict(), "cqr2": dict(method="cqr2"), "rcqr": dict(method="rcqr"), "robust": dict(robust=True)}
print("\n[accuracy, 3072x210] kappa | orth: " + " ".join(f"{k:>7}" for k in M) + "  dion | subspace: " + " ".join(f"{k:>7}" for k in M))
for kappa in (1e1, 7e1, 1e3, 4.5e4, 1e6, 1e7, 1e8):
    Y = make(3072, 210, kappa)
    Qs = {k: cholesky_qr(Y, **kw) for k, kw in M.items()}
    Qd = dion_rcqr(Y)
    print(f"{kappa:7.0e} | " + " ".join(f"{orth(Q):7.1e}" for Q in Qs.values()) + f" {orth(Qd):7.1e} | "
          + " ".join(f"{sub(Q, Y):7.1e}" for Q in Qs.values()))

def timeit(f, Y, n=20):
    for _ in range(3): f(Y)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f(Y)
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
torch._dynamo.config.cache_size_limit = 128   # the four lambdas share one code object
F = {k: torch.compile(lambda Y, kw=kw: cholesky_qr(Y, **kw), fullgraph=True, dynamic=False) for k, kw in M.items()}
print("\n[speed ms, batched] shape | linalg.qr | " + " | ".join(F) + " | dion rcqr (loop)")
for b, m, r in [(40, 768, 210), (10, 3072, 210), (16, 4096, 256), (4, 768, 64), (1, 3072, 210), (1, 8192, 512)]:
    Y = torch.randn(b, m, r, device=dev)
    th = timeit(lambda X: torch.linalg.qr(X, mode="reduced")[0], Y)
    ts = [timeit(f, Y) for f in F.values()]
    td = timeit(lambda X: [dion_rcqr(x) for x in X], Y)
    print(f"{b}x{m}x{r} | {th:7.2f} | " + " | ".join(f"{t:6.2f}" for t in ts) + f" | {td:6.2f}")
