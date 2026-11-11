"""Probe: does torch.linalg.cholesky_ex run on a batch of b SPD matrices of size l x l on this GPU?
One shape per process (a HIP launch failure leaves the context unusable). Seen on AMD Instinct MI350X, torch 2.10.0 +
ROCm 7.1.1: batch >= 2 with l >= 257 raised "HIP error: unspecified launch failure"; single matrices and l <= 256 ran.
Usage: for bl in "2 256" "2 257"; do python benchmarks/probe_rocm_cholesky_batch.py $bl; done"""
import sys, torch
b, l = int(sys.argv[1]), int(sys.argv[2]); dev = torch.device("cuda"); torch.manual_seed(0)
A = torch.randn(b, 2 * l, l, device=dev); G = A.mT @ A
try:
    L, info = torch.linalg.cholesky_ex(G); torch.cuda.synchronize(); print(b, l, "ok")
except Exception as e:
    print(b, l, "FAIL", str(e).splitlines()[0])
