"""Judge for PREREG_THRESH_WIDE_2026-11-12 (written and committed before any run of benchmarks/thresh_wide.py).

Usage: python benchmarks/thresh_wide_judge.py DIR      (reads DIR/runs_*.jsonl)

Protocol (identical to the note's Sec. 7 validation): denominators are the runs in which no factorization failed (any
pass with info > 0 or a non-finite factor is flagged through info in any case and excluded). Degraded: final
||Q^T Q - I||_2 > 1e-4. Good: final ||Q^T Q - I||_2 <= 1e-5. Flag: Lmin of the last pass < 0.195.
False negative (FN): degraded and not flagged. False alarm (FP): good and flagged.
"""
import glob, json, math, os, sys

U = 2.0 ** -24
TH = math.sqrt(64 * U / 1e-4)


def load(d):
    R = {}
    for f in sorted(glob.glob(os.path.join(d, "runs_*.jsonl"))):
        for x in open(f):
            if x.strip():
                r = json.loads(x)
                if not r.get("header"):
                    R[(r["key"], r["method"], r["mode"])] = r          # de-duplicate (resumed shards)
    return list(R.values())


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))] if v else float("nan")


def main():
    R = load(sys.argv[1])
    print(f"runs: {len(R)}; threshold {TH:.4f}")
    lib = [r for r in R if r["mode"] == "lib"]
    nid = sum(not r["lib_identical"] for r in lib); nfl = sum(r["flag"] != r["lib_flag"] for r in lib)
    print(f"S0 sanity (not a prediction): mirror vs cholqr._cqr bit-identical in {len(lib) - nid}/{len(lib)} lib runs; "
          f"breakdown flag agrees in {len(lib) - nfl}/{len(lib)}")
    verdict, c90 = {}, {}
    for l in (512, 1024):
        Rl = [r for r in R if r["l"] == l]
        nb = [r for r in Rl if not any(p["fail"] for p in r["passes"])]
        bad = [r for r in nb if r["orth2_final"] > 1e-4]
        good = [r for r in nb if r["orth2_final"] <= 1e-5]
        fn = [r for r in bad if r["passes"][-1]["Lmin"] >= TH]
        fp = [r for r in good if r["passes"][-1]["Lmin"] < TH]
        c1 = "not testable" if len(bad) < 20 else ("PASS" if len(bad) - len(fn) >= 0.99 * len(bad) else "FAIL")
        c2 = "PASS" if len(fp) <= 0.005 * len(good) else "FAIL"
        verdict[l] = dict(c1=c1, c2=c2, fn=len(fn), fp=len(fp))
        print(f"\nl = {l} (m = {Rl[0]['m'] if Rl else '?'}): runs {len(Rl)}, non-broken {len(nb)}, degraded {len(bad)}, "
              f"good {len(good)}, in between {len(nb) - len(bad) - len(good)}")
        print(f"  C1 caught {len(bad) - len(fn)}/{len(bad)} (FN {len(fn)}) -> {c1}")
        print(f"  C2 false alarms {len(fp)}/{len(good)} -> {c2}")
        for r in fn:
            print(f"    FN: {r['key']} {r['method']} {r['mode']} orth {r['orth2_final']:.2e} Lmin_last "
                  f"{r['passes'][-1]['Lmin']:.3f}")
        for r in fp:
            print(f"    FP: {r['key']} {r['method']} {r['mode']} orth {r['orth2_final']:.2e} Lmin_last "
                  f"{r['passes'][-1]['Lmin']:.3f}")
        cs = [r["orth2_final"] * r["passes"][-1]["Lmin"] ** 2 / U for r in nb
              if r["passes"][-1]["Lmin"] < 0.6 and r["orth2_final"] > 1e-6]
        c90[l] = pct(cs, 0.9)
        print(f"  c = orth * Lmin_last^2 / u over {len(cs)} runs (Lmin_last < 0.6, orth > 1e-6): median {pct(cs, 0.5):.1f}, "
              f"p90 {c90[l]:.1f}, max {max(cs) if cs else float('nan'):.1f}")
        need = max((r["passes"][-1]["Lmin"] for r in bad), default=float("nan"))
        print(f"  (descriptive) largest Lmin_last among degraded runs: {need:.3f}")
        for meth in ("scqr3", "cqr2"):
            for mode in ("lib", "pw8"):
                b = [r for r in bad if r["method"] == meth and r["mode"] == mode]
                g = [r for r in good if r["method"] == meth and r["mode"] == mode]
                print(f"    {meth:5s} {mode}: degraded {len(b)} (FN {sum(r in fn for r in b)}), good {len(g)} "
                      f"(FP {sum(r in fp for r in g)})")
    print("\nRegistered predictions:")
    v5, v10 = verdict.get(512), verdict.get(1024)
    print(f"  W1 l=512: C1 PASS and FN in [0, 2]: C1 {v5['c1']}, FN {v5['fn']} -> "
          f"{'PASS' if v5['c1'] == 'PASS' and v5['fn'] <= 2 else 'FAIL'}")
    print(f"  W2 l=1024: C1 FAIL with FN in [1, 15]: C1 {v10['c1']}, FN {v10['fn']} -> "
          f"{'PASS' if v10['c1'] == 'FAIL' and 1 <= v10['fn'] <= 15 else 'FAIL'}")
    print(f"  W3 C2 PASS at both l, FP <= 3 each: l=512 {v5['c2']} ({v5['fp']}), l=1024 {v10['c2']} ({v10['fp']}) -> "
          f"{'PASS' if v5['c2'] == v10['c2'] == 'PASS' and max(v5['fp'], v10['fp']) <= 3 else 'FAIL'}")
    print(f"  W4 p90 of c: l=1024 ({c90[1024]:.1f}) >= 64 and > l=512 ({c90[512]:.1f}) -> "
          f"{'PASS' if c90[1024] >= 64 and c90[1024] > c90[512] else 'FAIL'}")
    fns = [r for r in R if not any(p["fail"] for p in r["passes"]) and r["orth2_final"] > 1e-4
           and r["passes"][-1]["Lmin"] >= TH]
    ok5 = all(r["passes"][-1]["Lmin"] < 0.6 for r in fns)
    print(f"  W5 every FN has Lmin_last in [0.195, 0.6): {len(fns)} FN -> "
          f"{'PASS' if ok5 else 'FAIL'}{' (vacuous: no FN)' if not fns else ''}")


if __name__ == "__main__":
    main()
