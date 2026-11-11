"""Forming error of Q = X L^{-T}: explicit triangular inverse + one GEMM (as in cholqr) vs a triangular solve.
(CPU only, FP32, precision "highest"; pre-registered, see benchmarks/prereg/PREREG_FORMING_2026-11-12.md)

Two variants of cholqr's passes (scqr3 and cqr2), identical except for the forming step:
  inv   L_inv = solve_triangular(L, I) (l x l), Q = X @ L_inv^T          (cholqr 0.5; checked bit-identical to the library)
  trsm  Q = solve_triangular(L^T, X, upper=True, left=False)             (row-wise triangular solve over the m rows)
Each variant runs its own passes from the same input, so pass 1 is an exactly paired comparison (same X and L) and later
passes differ in their inputs. Per pass (successful factorizations), in FP64 from the FP32 objects:
  eta      = ||Q~ - X L^{-T}||_2, reference X L^{-T} in FP64 from the FP32 X and L (explicit FP64 inverse);
  ref_disc = ||ref - solve_triangular_FP64(L^T, X)||_2 (two FP64 references; bounds the reference error);
  condL = ||L||_2 ||L^{-1}||_2, invL2 = ||L^{-1}||_2^2, e = ||L L^T - X^T X - s I||_2, d = (s + e) invL2 (Theorem D(a)),
  bound_c = d + 2 eta sqrt(1 + d) + eta^2 (Theorem D(c)), orth_out = ||Q~^T Q~ - I||_2.
Final ||Q^T Q - I||_2 per variant. u = 2^-24 throughout.

Inputs: shapes 1024 x 64 and 4096 x 276; spectra geometric, two-cluster, staircase (construction of
benchmarks/device_sweep.py); nominal cond 10^2, 10^2.5, ..., 10^8; draws 1, 2; seed key prefix "FE-" (new draws).

Usage:  python benchmarks/forming_error.py --out FILE.jsonl
        python benchmarks/forming_error.py --smoke           (one input, prints timings only)
        python benchmarks/forming_error.py --judge FILE.jsonl
"""
import argparse, json, math, os, platform, statistics, sys, time, zlib
import torch
import cholqr
from cholqr import core

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from device_sweep import make64  # noqa: E402

torch.set_float32_matmul_precision("highest")
EPS = torch.finfo(torch.float32).eps      # 2^-23 = 2u, cholqr's later-pass shift
U = 2.0 ** -24
D64 = torch.float64


def n2(M):
    """Spectral norm in FP64 via the eigenvalues of the smaller Gram matrix (no SVD convergence failures)."""
    if not torch.isfinite(M).all():
        return float("inf")
    G = M.mT @ M if M.shape[0] >= M.shape[1] else M @ M.mT
    return math.sqrt(max(float(torch.linalg.eigvalsh(G)[-1]), 0.0))


def nsym(M):
    return float(torch.linalg.eigvalsh(M).abs().amax()) if torch.isfinite(M).all() else float("inf")


def run(Y, method, variant, log=True):
    l = Y.shape[-1]
    n_pass = {"scqr3": 3, "cqr2": 2}[method]
    Q = Y.float()
    eye = torch.eye(l, dtype=torch.float32); I64 = torch.eye(l, dtype=D64)
    passes = []
    for i in range(n_pass):
        if i == 0:
            Q, _ = core._prescale(Q)
        d = Q.norm(dim=-2, keepdim=True).clamp_min(1e-30)
        Q = Q / d
        G = core._gram(Q)
        s = 1e-6 * l if (method == "scqr3" and i == 0) else EPS
        L, info = torch.linalg.cholesky_ex(G + (s + 1e-30) * eye)
        fail = int(info) != 0 or not bool(torch.isfinite(L.diagonal()).all())
        if variant == "inv":
            L_inv = torch.linalg.solve_triangular(L, eye, upper=False)
            Qn = Q @ L_inv.mT
        else:
            Qn = torch.linalg.solve_triangular(L.mT, Q, upper=True, left=False)
        Qn = Q if fail else Qn
        rec = dict(k=i + 1, fail=fail, Lmin=float(L.diagonal().amin()))
        if log and not fail:
            X = Q.double(); L64 = L.double()
            Li = torch.linalg.solve_triangular(L64, I64, upper=False)
            ref = X @ Li.mT
            ref2 = torch.linalg.solve_triangular(L64.mT, X, upper=True, left=False)
            Qo = Qn.double()
            eta = n2(Qo - ref)
            invL2 = n2(Li) ** 2
            e = nsym(L64 @ L64.mT - X.mT @ X - s * I64)
            dd = (s + e) * invL2
            rec.update(eta=eta, ref_disc=n2(ref - ref2), condL=n2(L64) * math.sqrt(invL2), invL2=invL2, e=e, d=dd,
                       bound_c=dd + 2 * eta * math.sqrt(1 + dd) + eta ** 2, orth_out=nsym(Qo.mT @ Qo - I64),
                       orth_exact=nsym(ref.mT @ ref - I64))
        passes.append(rec)
        Q = Qn
    Qd = Q.double()
    return Q, passes, nsym(Qd.mT @ Qd - I64)


def grid():
    for (m, l) in ((1024, 64), (4096, 276)):
        for spec in ("geometric", "cluster", "staircase"):
            for x in range(4, 17):
                for draw in (1, 2):
                    yield m, l, spec, 10 ** (x / 2), draw


def gen(m, l, spec, kappa, draw):
    key = f"FE-{m}-{l}-{spec}-{round(math.log10(kappa) * 2)}-{draw}"
    g = torch.Generator().manual_seed(7000 + zlib.crc32(key.encode()) % 100000)
    return key, make64(m, l, kappa, spec, g).float()


def judge(path):
    R = [json.loads(x) for x in open(path) if x.strip()]
    R = [r for r in R if not r.get("header")]
    by = {}
    for r in R:
        by[(r["key"], r["method"], r["variant"])] = r
    print(f"runs: {len(R)}; library bit-identity of 'inv': {sum(r.get('lib_identical', True) for r in R if r['variant'] == 'inv')}"
          f"/{sum(r['variant'] == 'inv' for r in R)}")
    keys = sorted({(r["key"], r["method"]) for r in R})
    q = lambda v, p: sorted(v)[min(len(v) - 1, max(0, math.ceil(p * len(v)) - 1))] if v else float("nan")
    # per-pass eta/u distributions
    print("\neta/u per pass (successful passes): median / p90 / max")
    for meth in ("scqr3", "cqr2"):
        for (m, l) in ((1024, 64), (4096, 276)):
            for k in (1, 2, 3):
                row = []
                for var in ("inv", "trsm"):
                    v = [p["eta"] / U for r in R if r["method"] == meth and r["variant"] == var and r["m"] == m
                         for p in r["passes"] if p["k"] == k and not p["fail"]]
                    row.append(f"{var}: {q(v, .5):8.2f} {q(v, .9):9.2f} {q(v, 1):10.2f} (n={len(v)})" if v else "")
                if any(row):
                    print(f"  {meth:5s} {m}x{l} pass {k}: " + " | ".join(row))
    # F1: pass 1 paired ratio
    rat1 = []
    for key, meth in keys:
        a, b = by.get((key, meth, "inv")), by.get((key, meth, "trsm"))
        if a and b and not a["passes"][0]["fail"]:
            rat1.append(a["passes"][0]["eta"] / max(b["passes"][0]["eta"], 1e-300))
    f1 = q(rat1, .5); print(f"\nF1 pass-1 paired ratio eta_inv/eta_trsm: median {f1:.2f}, p10 {q(rat1, .1):.2f}, p90 {q(rat1, .9):.2f}, "
                            f"max {q(rat1, 1):.2f} (n={len(rat1)}) -> {'PASS' if 1 <= f1 <= 10 and q(rat1, 1) <= 100 else 'FAIL'}"
                            " (registered: median in [1, 10], max <= 100)")
    # F2: later passes
    med = {}
    for var in ("inv", "trsm"):
        v = [p["eta"] / U for r in R if r["variant"] == var for p in r["passes"] if p["k"] >= 2 and not p["fail"]]
        med[var] = q(v, .5)
    rr = med["inv"] / med["trsm"]
    print(f"F2 passes >= 2: median eta/u inv {med['inv']:.2f}, trsm {med['trsm']:.2f}, ratio {rr:.2f} -> "
          f"{'PASS' if max(med.values()) <= 5 and 0.5 <= rr <= 2 else 'FAIL'} (registered: both <= 5, ratio in [0.5, 2])")
    # F3: final orthogonality and breakdown pattern
    ro, same, n = [], 0, 0
    for key, meth in keys:
        a, b = by.get((key, meth, "inv")), by.get((key, meth, "trsm"))
        if not (a and b):
            continue
        n += 1
        fa, fb = any(p["fail"] for p in a["passes"]), any(p["fail"] for p in b["passes"])
        same += fa == fb
        if not fa and not fb:
            ro.append(a["orth_final"] / max(b["orth_final"], 1e-300))
    f3 = q(ro, .5)
    print(f"F3 final orth ratio inv/trsm (both non-broken): median {f3:.2f}, p10 {q(ro, .1):.2f}, p90 {q(ro, .9):.2f}, "
          f"min {q(ro, 0):.3f}, max {q(ro, 1):.2f} (n={len(ro)}); same breakdown status {same}/{n} -> "
          f"{'PASS' if 0.5 <= f3 <= 2 and same >= 0.95 * n else 'FAIL'} (registered: median in [0.5, 2], >= 95% same)")
    # F4: reference accuracy
    pp = [p for r in R for p in r["passes"] if not p["fail"]]
    ok4 = sum(p["ref_disc"] <= 0.01 * p["eta"] for p in pp)
    print(f"F4 reference discrepancy <= 0.01 eta: {ok4}/{len(pp)} -> {'PASS' if ok4 >= 0.99 * len(pp) else 'FAIL'}; "
          f"max ref_disc/u {max(p['ref_disc'] for p in pp) / U:.2e}")
    # F5: Theorem D(c)
    viol = sum(p["orth_out"] > p["bound_c"] * (1 + 1e-9) for p in pp)
    print(f"F5 Theorem D(c) violations: {viol}/{len(pp)} -> {'PASS' if viol == 0 else 'FAIL'}")
    # descriptive: share of the eta terms in the bound, final orth by size/method
    print("\n(descriptive) eta terms 2 eta sqrt(1+d) + eta^2 as a fraction of bound_c, last pass: median / max")
    for var in ("inv", "trsm"):
        v = []
        for r in R:
            if r["variant"] != var or any(p["fail"] for p in r["passes"]):
                continue
            p = r["passes"][-1]; t = 2 * p["eta"] * math.sqrt(1 + p["d"]) + p["eta"] ** 2
            v.append(t / p["bound_c"])
        print(f"  {var}: {q(v, .5):.3f} / {q(v, 1):.3f} (n={len(v)})")
    print("(descriptive) final ||Q^TQ - I||_2 / u, non-broken: median / max")
    for meth in ("scqr3", "cqr2"):
        for (m, l) in ((1024, 64), (4096, 276)):
            row = []
            for var in ("inv", "trsm"):
                v = [r["orth_final"] / U for r in R if r["method"] == meth and r["variant"] == var and r["m"] == m
                     and not any(p["fail"] for p in r["passes"])]
                row.append(f"{var} {q(v, .5):.1f} / {q(v, 1):.3g} (n={len(v)})")
            print(f"  {meth:5s} {m}x{l}: " + " | ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--judge")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.judge:
        return judge(a.judge)
    if a.smoke:
        for (m, l) in ((1024, 64), (4096, 276)):
            t0 = time.time(); Y = make64(m, l, 1e4, "geometric", torch.Generator().manual_seed(1)).float()
            for meth in ("scqr3", "cqr2"):
                for var in ("inv", "trsm"):
                    run(Y, meth, var)
            print(f"smoke: {m}x{l}: {time.time() - t0:.1f} s per input (4 runs)")
        return
    with open(a.out, "w") as f:
        f.write(json.dumps(dict(header=True, torch=torch.__version__, cholqr=cholqr.__version__,
                                python=platform.python_version(), machine=platform.machine(), threads=a.threads)) + "\n")
        for (m, l, spec, kappa, draw) in grid():
            key, Y = gen(m, l, spec, kappa, draw)
            for meth in ("scqr3", "cqr2"):
                for var in ("inv", "trsm"):
                    Q, passes, o = run(Y, meth, var)
                    rec = dict(key=key, m=m, l=l, spec=spec, kappa=kappa, draw=draw, method=meth, variant=var,
                               passes=passes, orth_final=o)
                    if var == "inv":
                        Ql = cholqr.cholesky_qr(Y, method=meth, on_breakdown="ignore")
                        rec["lib_identical"] = bool(torch.equal(Ql, Q))
                    f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
