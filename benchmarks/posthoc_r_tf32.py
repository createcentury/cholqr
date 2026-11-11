"""POST HOC (after the TF32HW judge): reproduces the 0.5.0 R error under emulated TF32.

Not pre-registered; written after the judge of tf32_hw_range.py (prereg benchmarks/prereg/PREREG_TF32HW_2026-11-12.md)
reported a TF32-level residual ||Y - QR||_F / ||Y||_F for cholqr 0.5's 3xTF32 path on an RTX 4090.

CPU emulation, no tensor-core accumulation effects. Variant A emulates 0.5's 3xTF32 split for _mm / _gram_mm only.
Variant B additionally rounds the operands of every remaining float32 `@` (the R assembly L.mT @ D and F @ R in 0.5.0)
to TF32, i.e. plain TF32 products. 18 inputs: 1024x64 and 4096x276; geometric, cluster, staircase; cond 1e2, 1e3, 1e4.

Run against cholqr 0.5.0 (commit 22789b7): B reproduces the TF32-level residual (median ~4.4e-4, max ~4.7e-4),
A stays at FP32 level (~5.9e-7). Against 0.5.1 (commit 371c721 and later) B equals A.

Usage: python benchmarks/posthoc_r_tf32.py
"""
import os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, cholqr
from cholqr import core
import tf32_hw_range as T
from downstream_orth import make64, seed_of, tf32_round, exact_q, sin_theta
orig = torch.Tensor.__matmul__
def mm(a, b):
    if a.dtype == torch.float32 and b.dtype == torch.float32:
        return orig(tf32_round(a.contiguous()).double(), tf32_round(b.contiguous()).double()).float()
    return orig(a, b)
out = {"A": [], "B": []}
for (m, l) in [(1024, 64), (4096, 276)]:
    for spec in ("geometric", "cluster", "staircase"):
        for kappa in (1e2, 1e3, 1e4):
            g = torch.Generator().manual_seed(seed_of("resid", m, l, spec, kappa, 0))
            Y64, U = make64(m, l, kappa, spec, g); Y32 = Y64.float(); Yd = Y32.double()
            for tag in ("A", "B"):
                with T.Patch(_mm=T._mm3, _gram_mm=T._gram3):
                    if tag == "B": torch.Tensor.__matmul__ = mm
                    try:
                        Q, R = cholqr.cholesky_qr(Y32, return_r=True)
                    finally:
                        torch.Tensor.__matmul__ = orig
                Q, R = Q.double(), R.double()
                out[tag].append((torch.linalg.matrix_norm(Yd - Q @ R) / torch.linalg.matrix_norm(Yd)).item())
for t in out: print(t, "resid median %.3g max %.3g" % (st.median(out[t]), max(out[t])))
