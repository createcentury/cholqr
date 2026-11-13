"""Pre-registered check (prereg/PREREG_LAM2_2026-11-14.md): is lam2 = smin^2 / (smin^2 + s) the quantity that gates a breakdown of FP32 scqr3?

For each input: pass-1 input after column normalisation, smin = its smallest singular value (FP64 SVD of the FP32 matrix),
s = 1e-6 * l (the first-pass shift), lam2 = smin^2 / (smin^2 + s) (Sec. 2, T0: smallest eigenvalue of the pass-2 Gram matrix in exact arithmetic).
Runs the instrumented mirror of cholqr's scqr3 passes on the CPU and records whether any factorization failed.
Writes one JSON line per input. Never prints results; --smoke prints timings only and writes nothing.
Usage: python benchmarks/lam2_check.py --out lam2_cpu.jsonl        |  python benchmarks/lam2_check.py --smoke
"""
import argparse, json, math, sys, time, zlib, os
import torch
sys.path.insert(0, os.path.dirname(__file__))
from device_sweep import make64, mirror, device_info  # noqa: E402

CELLS = ((1680, 210), (3072, 210), (4096, 512))
SPECTRA = ("geometric", "powerlaw", "spike", "cluster", "linear", "staircase")
KAPPAS = [10 ** (x / 2) for x in range(6, 21)]
DRAWS = 3
U = 2.0 ** -24


def one(m, l, spec, kappa, draw):
    key = f"L2-{m}-{l}-{spec}-{round(math.log10(kappa) * 2)}-{draw}"
    g = torch.Generator().manual_seed(9000 + zlib.crc32(key.encode()) % 100000)
    Y = make64(m, l, kappa, spec, g).float()
    _, passes = mirror(Y, "scqr3")
    p1 = passes[0]
    lam2 = p1["smin"] ** 2 / (p1["smin"] ** 2 + p1["s"])
    failed = [p["k"] for p in passes if p["info"] > 0 or p.get("nanL")]
    return dict(m=m, l=l, spec=spec, kappa=kappa, draw=draw, lam2_over_u=lam2 / U, smin=p1["smin"], s=p1["s"],
                failed=bool(failed), first_failed_pass=(failed[0] if failed else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_float32_matmul_precision("highest")
    if a.smoke:
        t0 = time.time()
        for m, l in CELLS:
            t = time.time(); one(m, l, "geometric", 1e5, 1); print(f"smoke {m}x{l}: {time.time() - t:.1f} s")
        print(f"total {time.time() - t0:.1f} s"); return
    with open(a.out, "a") as f:
        f.write(json.dumps(dict(header=True, **device_info(torch.device("cpu")))) + "\n")
        for m, l in CELLS:
            for spec in SPECTRA:
                for kappa in KAPPAS:
                    for draw in range(1, DRAWS + 1):
                        f.write(json.dumps(one(m, l, spec, kappa, draw)) + "\n"); f.flush()


if __name__ == "__main__":
    main()
