"""POST HOC (not pre-registered; written after seeing the judge output of downstream_orth.py, 2026-11-12).

Question: W2 failed because PowerSGD trajectories with an injected orthogonality error delta deviate from the exact
trajectory by up to ~3x delta per step. Is that specific to orthogonality, or does any perturbation of relative size
delta per step move the error-feedback trajectory by a similar amount?  Control: exact orthonormalization, but each
gradient G_t perturbed by a Gaussian matrix of relative Frobenius size delta. Same matrices/seeds as the registered run.

Usage: python benchmarks/downstream_posthoc.py > benchmarks/results/downstream_2026-11/posthoc_powersgd_control.txt
"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from downstream_orth import D64, DELTAS, exact_q, seed_of


def run(r, seed, delta, n=1024, m=512, T=20):
    g = torch.Generator().manual_seed(seed_of("psgd", r, seed))            # identical to the registered run
    UL = torch.linalg.qr(torch.randn(n, 64, dtype=D64, generator=g))[0]
    VL = torch.linalg.qr(torch.randn(m, 64, dtype=D64, generator=g))[0]
    L = (UL * torch.arange(1, 65, dtype=D64) ** -1.0) @ VL.mT
    L = L / torch.linalg.matrix_norm(L)
    Gs = []
    for _ in range(T):
        N = torch.randn(n, m, dtype=D64, generator=g)
        Gs.append(L + 0.5 * N / torch.linalg.matrix_norm(N))
    Q = torch.randn(m, r, dtype=D64, generator=g)
    gp = torch.Generator().manual_seed(seed_of("posthoc", r, seed, delta))
    e = torch.zeros(n, m, dtype=D64); errs = []; sumA = torch.zeros(n, m, dtype=D64)
    for t in range(T):
        G = Gs[t]
        if delta:
            Z = torch.randn(n, m, dtype=D64, generator=gp)
            G = G + delta * torch.linalg.matrix_norm(G) * Z / torch.linalg.matrix_norm(Z)
        M = G + e
        P = exact_q(M @ Q)
        Q = M.mT @ P
        A = P @ Q.mT
        errs.append((torch.linalg.matrix_norm(M - A) / torch.linalg.matrix_norm(M)).item())
        e = M - A; sumA += A
    sumG = sum(Gs)
    return errs, (torch.linalg.matrix_norm(sumG - sumA) / torch.linalg.matrix_norm(sumG)).item()


print("r seed delta | max_t |err_t - err_t(exact)| | cum_err rel. diff (gradient perturbed, exact orthonormal P)")
for r in (4, 16):
    for seed in (0, 1):
        e0, c0 = run(r, seed, 0.0)
        for d in DELTAS:
            e1, c1 = run(r, seed, d)
            print(f"{r} {seed} {d:g} | {max(abs(a - b) for a, b in zip(e1, e0)):.1e} | {c1 / c0 - 1:+.1e}")
