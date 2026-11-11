"""Residual, orthogonality and range accuracy of cholqr 0.5 under TF32 on real tensor cores (pre-registered, see
benchmarks/prereg/PREREG_TF32HW_2026-11-12.md).

Grid = the `resid` grid of benchmarks/downstream_orth.py (Sec. 6.11 of the note): 1024x64 and 4096x276; geometric,
two-cluster and staircase spectra; nominal cond 1e2..1e8 in half decades; 2 seeds; same seeds and construction
(make64), so the inputs are bit-identical to those of downstream_2026-11/resid.jsonl. All metrics in FP64 on the CPU,
with the functions of downstream_orth.py (orth_f = ||Q^TQ - I||_F, resid vs fl(Y), sin_fl vs range(fl(Y))).

Variants
  --device cuda (RTX 4090):
    c3_scqr3, c3_scqr3_raw   precision "high": cholqr 0.5's 3xTF32 split active (default call / on_breakdown="ignore")
    plain_scqr3              precision "high" with core._tf32_active patched to False: the products run in plain
                             TF32 on the tensor cores (cholqr 0.4 behaviour; control)
    fp32_scqr3, fp32_scqr3_raw  precision "highest"
  --device cpu:
    emu3_scqr3, emu3_scqr3_raw  CPU emulation of the 0.5 split path (reviewer script /tmp/refchk/comp.py, made
                             reproducible here): hi/lo operands each rounded to TF32, FP64 accumulation, FP32 result.

Usage: python benchmarks/tf32_hw_range.py --device cuda --out DIR/hw.jsonl
       python benchmarks/tf32_hw_range.py --device cpu  --out DIR/emu.jsonl
       python benchmarks/tf32_hw_range.py --device cuda --smoke     (tiny grid, prints timings only, writes nothing)
       python benchmarks/tf32_hw_range.py --judge DIR               (needs DIR/hw.jsonl and DIR/emu.jsonl)
"""
import argparse, json, math, os, statistics, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import cholqr
from cholqr import core
from downstream_orth import make64, seed_of, tf32_round, exact_q, sin_theta, orth_f, finite, D64

BROKEN = 0.1


# ---------------------------------------------------------------- CPU emulation of the 0.5 3xTF32 path
def _tmm(A, B):
    return (tf32_round(A.contiguous()).double() @ tf32_round(B.contiguous()).double()).float()


def _mm3(A, B, tf32):
    if A.dtype != torch.float32:
        return A @ B
    Ah, Al = core._tf32_split(A)
    Bh, Bl = core._tf32_split(B.contiguous())
    return _tmm(Ah, Bh) + (_tmm(Ah, Bl) + _tmm(Al, Bh))


def _gram3(A, tf32):
    if A.dtype != torch.float32:
        return A.mT @ A
    Ah, Al = core._tf32_split(A)
    M = _tmm(Ah.mT, Al)
    return _tmm(Ah.mT, Ah) + (M + M.mT)


class Patch:
    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.old = {k: getattr(core, k) for k in self.kw}
        for k, v in self.kw.items():
            setattr(core, k, v)

    def __exit__(self, *a):
        for k, v in self.old.items():
            setattr(core, k, v)


def run_variant(name, Y, sync):
    raw = name.endswith("_raw")
    kw = dict(return_r=True, on_breakdown="ignore" if raw else "fallback")
    base = name[:-4] if raw else name
    if base == "c3_scqr3":
        torch.set_float32_matmul_precision("high"); ctx = Patch()
    elif base == "plain_scqr3":
        torch.set_float32_matmul_precision("high"); ctx = Patch(_tf32_active=lambda Q: False)
    elif base == "fp32_scqr3":
        torch.set_float32_matmul_precision("highest"); ctx = Patch()
    elif base == "emu3_scqr3":
        torch.set_float32_matmul_precision("highest"); ctx = Patch(_mm=_mm3, _gram_mm=_gram3)
    else:
        raise ValueError(name)
    with ctx:
        tf32 = core._tf32_active(Y)
        sync(); t = time.perf_counter()
        Q, R = cholqr.cholesky_qr(Y, **kw)
        sync(); dt = time.perf_counter() - t
    torch.set_float32_matmul_precision("highest")
    return Q, R, dt, tf32


def main_run(a):
    dev = torch.device(a.device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    variants = (("c3_scqr3", "c3_scqr3_raw", "plain_scqr3", "fp32_scqr3", "fp32_scqr3_raw") if dev.type == "cuda"
                else ("emu3_scqr3", "emu3_scqr3_raw"))
    shapes = [(1024, 64)] if a.smoke else [(1024, 64), (4096, 276)]
    kappas = [1e3] if a.smoke else [10 ** (e / 2) for e in range(4, 17)]
    specs = ("geometric",) if a.smoke else ("geometric", "cluster", "staircase")
    seeds = (0,) if a.smoke else (0, 1)
    out = None if a.smoke else open(a.out, "w")
    if out:
        meta = dict(meta=True, torch=torch.__version__, cuda=torch.version.cuda, cholqr=cholqr.__version__,
                    cholqr_file=cholqr.__file__, device=str(dev),
                    gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
                    capability=list(torch.cuda.get_device_capability(dev)) if dev.type == "cuda" else None)
        try:
            meta["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                                                 "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        except Exception:
            pass
        out.write(json.dumps(meta) + "\n")
    # warm-up (kernels, cuBLAS handles)
    for v in variants:
        run_variant(v, torch.randn(1024, 64, device=dev), sync)
    t0 = time.perf_counter()
    for (m, l) in shapes:
        for spec in specs:
            for kappa in kappas:
                for seed in seeds:
                    g = torch.Generator().manual_seed(seed_of("resid", m, l, spec, kappa, seed))
                    Y64, U = make64(m, l, kappa, spec, g)
                    Y32 = Y64.float()
                    Yd = Y32.double()
                    Ufl = exact_q(Yd)
                    nY = torch.linalg.matrix_norm(Yd).item()
                    Ydev = Y32.to(dev)
                    for v in variants:
                        Q, R, dt, tf32 = run_variant(v, Ydev, sync)
                        if a.smoke:
                            print(f"{m}x{l} {spec} {v}: {1e3 * dt:.2f} ms", flush=True)
                            continue
                        Q, R = Q.cpu().double(), R.cpu().double()
                        resid = (torch.linalg.matrix_norm(Yd - Q @ R).item() / nY
                                 if finite(Q) and finite(R) else float("nan"))
                        rec = dict(m=m, l=l, spec=spec, kappa=kappa, seed=seed, method=v, tf32_active=tf32,
                                   orth=orth_f(Q), resid=resid, sin_fl=sin_theta(Q, Ufl), sin_ex=sin_theta(Q, U),
                                   qr_ms=1e3 * dt)
                        out.write(json.dumps(rec) + "\n")
                    if out:
                        out.flush()
    print(f"total {time.perf_counter() - t0:.1f} s", flush=True)


# ---------------------------------------------------------------- judge
def load(path):
    return [r for r in map(json.loads, open(path)) if not r.get("meta")]


def ok(r):
    return math.isfinite(r["orth"]) and r["orth"] <= BROKEN and math.isfinite(r["sin_fl"])


def judge(d):
    hw, emu = load(os.path.join(d, "hw.jsonl")), load(os.path.join(d, "emu.jsonl"))
    key = lambda r: (r["m"], r["spec"], r["kappa"], r["seed"])
    by = {}
    for r in hw + emu:
        by.setdefault(r["method"], {})[key(r)] = r
    lines, res = [], {}

    def say(k, passed, detail):
        res[k] = passed
        lines.append(f"{k}: {'PASS' if passed else 'FAIL'}  {detail}")

    c3 = by["c3_scqr3"]
    valid = all(r["tf32_active"] for r in c3.values()) and not any(r["tf32_active"] for r in by["fp32_scqr3"].values())
    lines.append(f"VALIDITY (TF32 active at 'high', inactive at 'highest'): {'OK' if valid else 'INVALID'}")
    good = [r for r in c3.values() if ok(r)]
    ratio = [r["sin_fl"] / r["kappa"] for r in good]
    med = statistics.median(ratio)
    say("H1", 1e-8 <= med <= 1e-7, f"c3_scqr3 median sin_fl/cond = {med:.3g} over {len(good)}/{len(c3)} non-broken "
                                   f"(target [1e-8, 1e-7])")
    n = sum(x <= 3e-7 for x in ratio)
    say("H2", n >= 0.98 * len(ratio), f"c3_scqr3 sin_fl <= 3e-7 cond in {n}/{len(ratio)} (need >= 98%); "
                                      f"max sin_fl/cond = {max(ratio):.3g}")
    pairs = [(c3[k], by["fp32_scqr3"][k]) for k in c3 if ok(c3[k]) and ok(by["fp32_scqr3"][k])]
    n = sum(p["sin_fl"] <= 3 * q["sin_fl"] for p, q in pairs)
    rr = [p["sin_fl"] / q["sin_fl"] for p, q in pairs]
    say("H3", n >= 0.9 * len(pairs), f"c3 sin_fl <= 3 x fp32 sin_fl in {n}/{len(pairs)} (need >= 90%); "
                                     f"ratio median {statistics.median(rr):.3g}, max {max(rr):.3g}")
    pairs = [(c3[k], by["emu3_scqr3"][k]) for k in c3 if ok(c3[k]) and ok(by["emu3_scqr3"][k])]
    rr = [p["sin_fl"] / q["sin_fl"] for p, q in pairs]
    n = sum(1 / 3 <= x <= 3 for x in rr)
    say("H4", n >= 0.9 * len(pairs), f"hardware/emulation sin_fl ratio in [1/3, 3] in {n}/{len(pairs)} (need >= 90%); "
                                     f"ratio median {statistics.median(rr):.3g}, range {min(rr):.3g}-{max(rr):.3g}")
    rs = [r["resid"] for r in good]
    say("H5", max(rs) <= 1e-5 and statistics.median(rs) <= 1e-6,
        f"c3_scqr3 residual max {max(rs):.3g} (<= 1e-5), median {statistics.median(rs):.3g} (<= 1e-6)")
    pl = [r["sin_fl"] / r["kappa"] for r in by["plain_scqr3"].values() if r["kappa"] <= 10 ** 3.5 * 1.0001
          and math.isfinite(r["sin_fl"])]
    mp = statistics.median(pl)
    say("H6", 3e-5 <= mp <= 5e-4, f"plain_scqr3 (control) median sin_fl/cond at cond <= 10^3.5 = {mp:.3g} over "
                                  f"{len(pl)} (target [3e-5, 5e-4])")
    f = [r["sin_fl"] / r["kappa"] for r in by["fp32_scqr3"].values() if ok(r)]
    n = sum(x <= 3e-7 for x in f)
    say("H7", n >= 0.98 * len(f), f"fp32_scqr3 sin_fl <= 3e-7 cond in {n}/{len(f)} (need >= 98%); "
                                  f"max {max(f):.3g}, median {statistics.median(f):.3g}")
    o = [r["orth"] for r in c3.values()]
    n = sum(x <= 1e-4 for x in o)
    say("H8", n >= 0.9 * len(o), f"c3_scqr3 (default) orth_F <= 1e-4 in {n}/{len(o)} (need >= 90%); "
                                 f"max {max(o):.3g}, median {statistics.median(o):.3g}")
    dec = valid and res["H1"] and res["H2"] and res["H3"]
    lines.append(f"DECISION (FP32-like range of 0.5's 3xTF32 on RTX 4090 = VALID and H1 and H2 and H3): "
                 f"{'PASS' if dec else 'FAIL'}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--judge", metavar="DIR")
    a = ap.parse_args()
    if a.judge:
        txt = judge(a.judge); print(txt)
        open(os.path.join(a.judge, "judge.txt"), "w").write(txt + "\n")
    else:
        main_run(a)
