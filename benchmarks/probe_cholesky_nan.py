"""Does torch.linalg.cholesky_ex on this device return non-finite factors with info == 0 for non-PD inputs?
Usage: python benchmarks/probe_cholesky_nan.py --device cuda"""
import argparse, json, math, sys, os
import torch
sys.path.insert(0, os.path.dirname(__file__))
from device_sweep import make
ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None)
a = ap.parse_args(); dev = torch.device(a.device); torch.set_float32_matmul_precision("highest")
rows = []
g = torch.Generator().manual_seed(0)
for l in (64, 276):
    for e in [x / 2 for x in range(6, 21)]:
        Y = make(1024 if l == 64 else 4096, l, 10 ** e, "geometric", g).to(dev)
        Q = Y / Y.norm(dim=0, keepdim=True)
        G = Q.mT @ Q
        for s in (0.0, 2.0 ** -23):
            for batched in (False, True):
                A = G + s * torch.eye(l, device=dev)
                A = A[None].repeat(3, 1, 1) if batched else A
                L, info = torch.linalg.cholesky_ex(A)
                fin = torch.isfinite(L).all(dim=(-2, -1))
                info = info.reshape(-1); fin = fin.reshape(-1)
                rows.append(dict(l=l, kappa=10 ** e, shift=s, batched=batched, info=info.tolist(), finite=fin.tolist(),
                                 nan_with_info0=bool(((info == 0) & ~fin).any())))
bad = [r for r in rows if r["nan_with_info0"]]
print(json.dumps(dict(device=str(dev), name=torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
                      torch=torch.__version__, cases=len(rows), nan_with_info0=len(bad), examples=bad[:5])))
if a.out: json.dump(rows, open(a.out, "w"))
