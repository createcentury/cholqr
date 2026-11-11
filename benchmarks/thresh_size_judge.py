"""Judge for PREREG_THRESH_SIZE_2026-11-12 (written and committed before any run of benchmarks/thresh_size.py).

Usage: python benchmarks/thresh_size_judge.py DIR QRROUND3_DIR MPS_CERT_JSONL
  DIR:            output of benchmarks/thresh_size.py (DIR/runs_*.jsonl), fresh sizes l = 384, 768, 1536
  QRROUND3_DIR:   r3_*.jsonl, the CPU validation data of the 0.195 rule (l = 64, 276; spectral norm)
  MPS_CERT_JSONL: the Apple GPU validation data of the 0.195 rule (l = 64, 276; Frobenius norm)

Protocol (as in the note's Sec. 7 and thresh_wide_judge.py): denominators are the runs in which no factorization
failed (flagged through info in any case; excluded). Degraded: final ||Q^T Q - I|| > 1e-4. Good: <= 1e-5. New flag:
Lmin of the last pass < tau(l) (benchmarks/thresh_size_rule.py). FN: degraded and not flagged. FA: good and flagged.
"""
import glob, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thresh_size_rule import tau, TAU_MIN, U  # noqa: E402

FRESH = (384, 768, 1536)
A_FIT, P_FIT = 0.6725, 0.8678     # unmargined least-squares fit of the per-size max of c (thresh_size_fit.py)


def jl(paths):
    R = {}
    for f in paths:
        for i, x in enumerate(open(f)):
            if x.strip():
                r = json.loads(x)
                if not r.get("header"):
                    k = (r["key"], r["method"], r["mode"]) if "key" in r else (f, i)
                    R[k] = r                                           # de-duplicate (resumed shards)
    return list(R.values())


def nonbroken(r):
    return not any(p.get("info", 0) > 0 or p.get("fail", False) for p in r["passes"])


def main():
    R = jl(sorted(glob.glob(os.path.join(sys.argv[1], "runs_*.jsonl"))))
    r3 = [r for r in jl(sorted(glob.glob(os.path.join(sys.argv[2], "r3_*.jsonl")))) if r["mode"] != "tf32e"]
    mps = jl([sys.argv[3]])
    last = lambda r: r["passes"][-1]["Lmin"]
    print(f"runs: {len(R)}; rule tau(l) = " + ", ".join(f"{l}: {tau(l):.4f}" for l in FRESH))
    lib = [r for r in R if r["mode"] == "lib"]
    nid = sum(not r["lib_identical"] for r in lib); nfl = sum(r["flag"] != r["lib_flag"] for r in lib)
    print(f"S0 sanity (not a criterion): mirror vs cholqr._cqr bit-identical in {len(lib) - nid}/{len(lib)} lib runs; "
          f"0.195 flag agrees in {len(lib) - nfl}/{len(lib)}")

    V = {}
    for l in FRESH:
        Rl = [r for r in R if r["l"] == l]
        nb = [r for r in Rl if nonbroken(r)]
        bad = [r for r in nb if r["orth2_final"] > 1e-4]
        good = [r for r in nb if r["orth2_final"] <= 1e-5]
        t = tau(l)
        fn = [r for r in bad if last(r) >= t]; fp = [r for r in good if last(r) < t]
        fn0 = [r for r in bad if last(r) >= TAU_MIN]; fp0 = [r for r in good if last(r) < TAU_MIN]
        t1 = "not testable" if len(bad) < 20 else ("PASS" if len(fn) <= 0.01 * len(bad) else "FAIL")
        t2 = "PASS" if len(fp) <= 0.02 * len(good) else "FAIL"
        t3 = "PASS" if len(fp) <= 0.005 * len(good) else "FAIL"
        cs = [r["orth2_final"] * last(r) ** 2 / U for r in nb if last(r) < 0.6 and 1e-6 < r["orth2_final"] <= 1e-3]
        cmax = max(cs) if cs else float("nan")
        V[l] = dict(t1=t1, t2=t2, t3=t3, fn=len(fn), fp=len(fp), nbad=len(bad), ngood=len(good), fn0=len(fn0),
                    ratio=cmax / (A_FIT * l ** P_FIT))
        print(f"\nl = {l} (m = {Rl[0]['m'] if Rl else '?'}), tau = {t:.4f}: runs {len(Rl)}, non-broken {len(nb)}, "
              f"degraded {len(bad)}, good {len(good)}, in between {len(nb) - len(bad) - len(good)}")
        print(f"  T1 caught {len(bad) - len(fn)}/{len(bad)} (FN {len(fn)}) -> {t1}")
        print(f"  T2 false alarms {len(fp)}/{len(good)} (<= 2%) -> {t2}")
        print(f"  T3 (secondary, protocol band <= 0.5%): {len(fp)}/{len(good)} -> {t3}")
        for r in fn:
            print(f"    FN: {r['key']} {r['method']} {r['mode']} orth {r['orth2_final']:.2e} Lmin_last {last(r):.3f}")
        for r in fp:
            print(f"    FA: {r['key']} {r['method']} {r['mode']} orth {r['orth2_final']:.2e} Lmin_last {last(r):.3f}")
        print(f"  (old rule 0.195, descriptive) FN {len(fn0)}/{len(bad)}, FA {len(fp0)}/{len(good)}")
        print(f"  c (first-order regime: Lmin_last < 0.6, 1e-6 < orth <= 1e-3) over {len(cs)} runs: max {cmax:.1f}; "
              f"fit A l^p = {A_FIT * l ** P_FIT:.1f}; ratio {V[l]['ratio']:.2f}; margined C(l) = "
              f"{1.01 * l ** 0.868:.1f}")
        print(f"  (descriptive) largest Lmin_last among degraded: {max((last(r) for r in bad), default=float('nan')):.3f}; "
              f"smallest among good: {min((last(r) for r in good), default=float('nan')):.3f}")
        for meth in ("scqr3", "cqr2"):
            for mode in ("lib", "pw8"):
                b = [r for r in bad if r["method"] == meth and r["mode"] == mode]
                g = [r for r in good if r["method"] == meth and r["mode"] == mode]
                print(f"    {meth:5s} {mode}: degraded {len(b)} (FN {sum(r in fn for r in b)}), good {len(g)} "
                      f"(FA {sum(r in fp for r in g)})")

    print("\nN1 non-regression on the 0.195 rule's validation data (in sample: these data were used for the fit)")
    n1 = True
    for name, runs, okey, lm in (("CPU QRROUND3", r3, "orth2_final", last), ("MPS CERT", mps, "orth", lambda r: r["Lmin_last"])):
        nb = [r for r in runs if nonbroken(r)]
        bad = [r for r in nb if r[okey] > 1e-4]; good = [r for r in nb if r[okey] <= 1e-5]
        lost = sum(lm(r) < TAU_MIN and lm(r) >= tau(r["l"]) for r in bad)
        fa_new = sum(lm(r) < tau(r["l"]) for r in good); fa_old = sum(lm(r) < TAU_MIN for r in good)
        caught_new = sum(lm(r) < tau(r["l"]) for r in bad); caught_old = sum(lm(r) < TAU_MIN for r in bad)
        ok = lost == 0 and fa_new <= fa_old
        n1 &= ok
        print(f"  {name}: degraded caught {caught_new}/{len(bad)} (0.195: {caught_old}; lost {lost}), false alarms "
              f"{fa_new}/{len(good)} (0.195: {fa_old}) -> {'PASS' if ok else 'FAIL'}")

    overall = n1 and all(V[l]["t1"] == "PASS" and V[l]["t2"] == "PASS" for l in FRESH)
    print("\nRegistered criteria (gate for the library change): T1 and T2 PASS at every fresh size, and N1 PASS -> "
          f"{'PASS' if overall else 'FAIL'}")
    print("\nRegistered predictions:")
    v = V
    print(f"  P1 l=384: T1 PASS with FN 0, FA <= 2: {v[384]['t1']}, FN {v[384]['fn']}, FA {v[384]['fp']} -> "
          f"{'PASS' if v[384]['t1'] == 'PASS' and v[384]['fn'] == 0 and v[384]['fp'] <= 2 else 'FAIL'}")
    print(f"  P2 l=768: T1 PASS with FN 0, FA <= 3: {v[768]['t1']}, FN {v[768]['fn']}, FA {v[768]['fp']} -> "
          f"{'PASS' if v[768]['t1'] == 'PASS' and v[768]['fn'] == 0 and v[768]['fp'] <= 3 else 'FAIL'}")
    print(f"  P3 l=1536: T1 PASS and T2 PASS: {v[1536]['t1']}, {v[1536]['t2']} (FN {v[1536]['fn']}, FA {v[1536]['fp']}) -> "
          f"{'PASS' if v[1536]['t1'] == 'PASS' and v[1536]['t2'] == 'PASS' else 'FAIL'}")
    ok4 = all(0.5 <= v[l]["ratio"] <= 2 for l in FRESH)
    print(f"  P4 max c within [0.5, 2] x A l^p at every fresh size: "
          + ", ".join(f"{l}: {v[l]['ratio']:.2f}" for l in FRESH) + f" -> {'PASS' if ok4 else 'FAIL'}")
    ok5 = v[768]["fn0"] >= 1 and v[1536]["fn0"] >= 1
    print(f"  P5 the old rule 0.195 misses >= 1 degraded run at l = 768 and at l = 1536: FN {v[768]['fn0']}, "
          f"{v[1536]['fn0']} -> {'PASS' if ok5 else 'FAIL'}")


if __name__ == "__main__":
    main()
