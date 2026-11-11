"""Fit of the size-dependent last-pass threshold tau(l) on EXISTING data only (see
benchmarks/prereg/PREREG_THRESH_SIZE_2026-11-12.md). Deterministic; re-running it reproduces the constants frozen in
benchmarks/thresh_size_rule.py (the frozen values are the printed ones rounded to 3 significant digits).

Usage: python benchmarks/thresh_size_fit.py QRROUND3_DIR THRESH_WIDE_DIR [MPS_CERT_JSONL]
  QRROUND3_DIR:    the QRROUND3 run files r3_*.jsonl (CPU, l = 64 / 276; the validation data of the 0.195 rule;
                   not in this repository)
  THRESH_WIDE_DIR: benchmarks/results/thresh_wide_2026-11 (CPU, l = 512 / 1024)
  MPS_CERT_JSONL:  optional, the Apple GPU validation run of the 0.195 rule (Frobenius norm; only for the in-sample
                   false-alarm constraint, not for the fit)

Fitting rule (fixed in the prereg):
  1. runs: no failed factorization, not TF32-emulated; orth = final ||Q^T Q - I||_2 (FP64); L = Lmin of the last pass.
  2. c = orth L^2 / u over runs with L < 0.6 and 1e-6 < orth <= 1e-3 (first-order regime: runs with orth near 1 are
     saturated and give meaningless c, up to 3554 at l = 1024).
  3. q_l = max c at each l (pooled over m, Gram orders, methods, spectra).
  4. least-squares fit ln q_l = ln A + p ln l over l in {64, 276, 512, 1024}.
  5. C(l) = M A l^p with margin M = the largest of {2, 1.5, 1.25, 1} for which, in sample, (a) no degraded run is
     missed at any fitting size and (b) the false alarms on the 0.195 rule's validation data (CPU QRROUND3, MPS CERT)
     are not above those of 0.195.
  6. tau(l) = min(TAU_MAX, max(0.195, sqrt(C(l) u / 1e-4))), TAU_MAX = 0.75.
"""
import glob, json, math, os, sys

U = 2.0 ** -24
TOL = 1e-4
TH0 = math.sqrt(64 * U / TOL)
TAU_MAX = 0.75


def jl(paths):
    out = []
    for f in paths:
        for x in open(f):
            if x.strip():
                r = json.loads(x)
                if not r.get("header"):
                    out.append(r)
    return out


def nonbroken(r):
    return not any(p.get("info", 0) > 0 or p.get("fail", False) for p in r["passes"])


def main():
    r3 = [r for r in jl(sorted(glob.glob(os.path.join(sys.argv[1], "r3_*.jsonl")))) if r["mode"] != "tf32e"]
    tw = jl(sorted(glob.glob(os.path.join(sys.argv[2], "runs_*.jsonl"))))
    mps = jl([sys.argv[3]]) if len(sys.argv) > 3 else []
    D = [r for r in r3 + tw if nonbroken(r)]
    sizes = sorted({r["l"] for r in D})
    q = {}
    for l in sizes:
        cs = [r["orth2_final"] * r["passes"][-1]["Lmin"] ** 2 / U for r in D if r["l"] == l
              and r["passes"][-1]["Lmin"] < 0.6 and 1e-6 < r["orth2_final"] <= 1e-3]
        q[l] = max(cs)
        print(f"l = {l}: {len(cs)} runs in the first-order regime, max c = {q[l]:.1f}")
    n = len(sizes)
    xs = [math.log(l) for l in sizes]; ys = [math.log(q[l]) for l in sizes]
    mx, my = sum(xs) / n, sum(ys) / n
    p = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    A = math.exp(my - p * mx)
    print(f"fit: max c ~ A l^p with A = {A:.4f}, p = {p:.4f}")

    def tau(l, M):
        return min(TAU_MAX, max(TH0, math.sqrt(M * A * l ** p * U / TOL)))

    def counts(runs, M, okey, lmin):
        nb = [r for r in runs if nonbroken(r)]
        bad = [r for r in nb if r[okey] > 1e-4]; good = [r for r in nb if r[okey] <= 1e-5]
        fn = sum(lmin(r) >= tau(r["l"], M) for r in bad); fp = sum(lmin(r) < tau(r["l"], M) for r in good)
        fn0 = sum(lmin(r) >= TH0 for r in bad); fp0 = sum(lmin(r) < TH0 for r in good)
        return fn, fp, fn0, fp0, len(bad), len(good)

    last = lambda r: r["passes"][-1]["Lmin"]
    chosen = None
    for M in (2.0, 1.5, 1.25, 1.0):
        ok = True
        print(f"M = {M}: tau = " + ", ".join(f"{l}: {tau(l, M):.3f}" for l in (64, 276, 384, 512, 768, 1024, 1536)))
        for l in sizes:
            fn, fp, fn0, fp0, nbad, ngood = counts([r for r in r3 + tw if r["l"] == l], M, "orth2_final", last)
            print(f"   CPU l = {l}: FN {fn}/{nbad} (0.195: {fn0}), FA {fp}/{ngood} (0.195: {fp0})")
            ok &= fn == 0
        fn, fp, fn0, fp0, nbad, ngood = counts(r3, M, "orth2_final", last)
        print(f"   CPU QRROUND3 total: FN {fn}/{nbad} (0.195: {fn0}), FA {fp}/{ngood} (0.195: {fp0})")
        ok &= fp <= fp0
        if mps:
            fn, fp, fn0, fp0, nbad, ngood = counts(mps, M, "orth", lambda r: r["Lmin_last"])
            print(f"   MPS CERT total: FN {fn}/{nbad} (0.195: {fn0}), FA {fp}/{ngood} (0.195: {fp0})")
            ok &= fp <= fp0
        print(f"   constraints (a) and (b): {'satisfied' if ok else 'violated'}")
        if ok and chosen is None:
            chosen = M
    print(f"chosen margin M = {chosen}; C(l) = {chosen * A:.4f} l^{p:.4f}")


if __name__ == "__main__":
    main()
