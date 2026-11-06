"""Benchmark cholqr.cholesky_qr against torch.linalg.qr on batched tall-skinny matrices (run on a GPU).
Usage: python benchmarks/bench_qr.py [--shapes 40x768x210,10x3072x210] [--compile] [--method scqr3|cqr2]"""
import argparse, time, torch
from cholqr import cholesky_qr

ap = argparse.ArgumentParser(); ap.add_argument("--shapes", default="40x768x210,10x3072x210"); ap.add_argument("--compile", action="store_true"); ap.add_argument("--method", default="scqr3")
a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
base = lambda Y: cholesky_qr(Y, method=a.method)
fn = torch.compile(base, fullgraph=True) if a.compile else base
def timeit(f, Y, n=20):
    for _ in range(3): f(Y)
    if dev == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n): f(Y)
    if dev == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3
for sh in a.shapes.split(","):
    b, m, r = map(int, sh.split("x")); Y = torch.randn(b, m, r, device=dev)
    th = timeit(lambda X: torch.linalg.qr(X, mode="reduced")[0], Y); tc = timeit(fn, Y)
    Q = fn(Y); err = (Q.mT @ Q - torch.eye(r, device=dev)).norm(dim=(-2, -1)).max().item()
    print(f"{dev} {b}x{m}x{r}: torch.linalg.qr {th:7.2f} ms | cholqr {tc:7.2f} ms | speedup {th/tc:5.1f}x | max ||Q^TQ-I||_F {err:.1e}")
