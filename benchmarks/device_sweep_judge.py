"""Evaluate a device_sweep.py output file against the registered checks (PREREG_DEVSWEEP).
Usage: python benchmarks/device_sweep_judge.py sweep_X.jsonl [more.jsonl ...]"""
import json, math, statistics as st, sys

u = 2.0 ** -24; EPS = 2.0 ** -23; TOL = 1e-13
TH = math.sqrt(64 * u / 1e-4)


def live(r):
    out = []
    for i, p in enumerate(r["passes"]):
        if i > 0 and r["passes"][i - 1]["info"] > 0: break
        out.append(p)
    return out


def judge(path):
    rows = [json.loads(x) for x in open(path) if x.strip()]
    hdr = next((r for r in rows if r.get("header")), {}); R = [r for r in rows if not r.get("header")]
    prec = R[0]["precision"] if R else "?"
    print(f"== {path}: {hdr.get('gpu', hdr.get('device'))} {hdr.get('capability', '')} torch {hdr.get('torch')} "
          f"precision {prec}; runs {len(R)}")
    res = {}
    broken = lambda r: any(p["info"] > 0 for p in r["passes"])
    # D1 eps class
    meds = {}
    for m in sorted({r["m"] for r in R}):
        for l in sorted({r["l"] for r in R}):
            v = [r["passes"][1]["eps_gram"] / u for r in R if r["m"] == m and r["l"] == l and r["method"] == "scqr3"
                 and r["passes"][0]["info"] == 0]
            if v: meds[(m, l)] = st.median(v)
    print("   eps_2/u medians:", {f"{m}x{l}": round(v, 1) for (m, l), v in meds.items()})
    res["D1_fp32_class"] = all(1 <= v <= 100 for v in meds.values())
    # D2 Theorem D, no margin
    va = vb = vc = vd = na = nc = nd = 0
    for r in R:
        Ps = r["passes"]
        for p in live(r):
            if p["info"] > 0: continue
            dl = (p["s"] + p["eps_tot"]) * p["invL2"]; na += 1
            va += p["orth_exact"] > dl * (1 + 1e-9) + TOL
            vb += p["orth_out"] > (dl + 2 * p["eta"] * math.sqrt(1 + dl) + p["eta"] ** 2) * (1 + 1e-9) + TOL
            den = p["smin"] ** 2 + p["s"] - p["eps_tot"]
            if den > 0: nc += 1; vc += p["invL2"] > (1 / den) * (1 + 1e-9) + TOL
        for k in range(len(Ps) - 1):
            a, b = Ps[k], Ps[k + 1]
            if a["info"] > 0: break
            dl = (a["s"] + a["eps_tot"]) * a["invL2"]; D = dl + 2 * a["eta"] * math.sqrt(1 + dl) + a["eta"] ** 2
            if D >= 1: continue
            nd += 1; vd += b["smin"] < (1 - r["m"] * u) * math.sqrt((1 - D) / (1 + D)) - 2 * math.sqrt(r["l"]) * u - TOL
    print(f"   Theorem D violations: (a) {va}/{na}, (c) {vb}/{na}, (b) {vc}/{nc}; D1 {vd}/{nd}")
    res["D2_theoremD"] = va + vb + vc + vd == 0
    # D3 rules
    P = [p for r in R for p in live(r)]
    rho = lambda p: (p["smin"] ** 2 + p["s"]) / max(p["eps_gram"], 1e-300)
    hi = [p["info"] > 0 for p in P if rho(p) >= 5]
    neg = [p["info"] > 0 for p in P if p["lam_hat"] < -2 * u]; pos = [p["info"] == 0 for p in P if p["lam_hat"] > 5 * u]
    print(f"   rho>=5 failures {sum(hi)}/{len(hi)}; lam_hat<-2u failures {sum(neg)}/{len(neg)}; lam_hat>5u successes {sum(pos)}/{len(pos)}")
    res["D3_rules"] = sum(hi) <= 0.01 * max(len(hi), 1) and sum(neg) >= 0.99 * len(neg) and sum(pos) >= 0.99 * len(pos)
    # D4 a priori pass-2 rule
    s3 = [r for r in R if r["method"] == "scqr3"]
    cal = {}
    for r in s3:
        if r["draw"] == 1: cal.setdefault((r["m"], r["l"]), []).append(r["passes"][1]["eps_gram"])
    cal = {k: st.median(v) for k, v in cal.items()}
    tot = cor = 0
    for r in s3:
        if r["draw"] != 2 or r.get("smin2_q1n_exact") is None: continue
        rp = (r["smin2_q1n_exact"] + EPS) / cal[(r["m"], r["l"])]
        if 1 / 3 <= rp <= 3: continue
        tot += 1; cor += (rp < 1) == (r["passes"][1]["info"] > 0)
    print(f"   a-priori pass-2 rule: {cor}/{tot}")
    res["D4_apriori"] = cor >= 0.9 * max(tot, 1)
    # D5 library default
    lib = [r["orth_lib_default"] for r in R]
    print(f"   library default: orth max {max(lib):.1e}; runs >= 1e-4: {sum(x >= 1e-4 for x in lib)}/{len(lib)}")
    res["D5_library"] = all(x < 1e-4 for x in lib)
    # D6 detector
    nb = [r for r in R if not broken(r)]
    bad = [r for r in nb if r["orth"] > 1e-4]; good = [r for r in nb if r["orth"] <= 1e-5]
    fb = sum(r["passes"][-1]["Lmin"] < TH for r in bad); fg = sum(r["passes"][-1]["Lmin"] < TH for r in good)
    print(f"   detector: caught {fb}/{len(bad)}; false alarms {fg}/{len(good)}")
    res["D6_detector"] = (fb >= 0.99 * len(bad)) and (fg <= 0.005 * max(len(good), 1))
    # cliff (reported): smallest nominal kappa with any failure, scqr3 / cqr2, geometric
    for meth in ("scqr3", "cqr2"):
        cl = {}
        for (m, l) in meds:
            ks = sorted(r["kappa"] for r in R if r["method"] == meth and r["m"] == m and r["l"] == l and r["spec"] == "geometric" and broken(r))
            cl[f"{m}x{l}"] = f"{ks[0]:.0e}" if ks else "none<=1e10"
        print(f"   first breakdown (geometric), {meth}: {cl}")
    print("   verdicts:", {k: ("PASS" if v else "FAIL") for k, v in res.items()})
    return res


if __name__ == "__main__":
    for p in sys.argv[1:]:
        judge(p)
