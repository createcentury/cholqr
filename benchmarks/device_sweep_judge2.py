"""Evaluate round-2 device_sweep outputs (input dtypes, robust, medium, compile, x86 CPU) (pre-registered round-2 checks).
Usage: python benchmarks/device_sweep_judge2.py FILE.jsonl [...]"""
import json, math, sys

u = 2.0 ** -24; TOL = 1e-13; TH = math.sqrt(64 * u / 1e-4)
U_IN = {"float64": 2.0 ** -53, "float32": 2.0 ** -24, "bfloat16": 2.0 ** -9, "float16": 2.0 ** -11, "float8_e4m3fn": 2.0 ** -4}


def live(r):
    out = []
    for i, p in enumerate(r["passes"]):
        if i > 0 and r["passes"][i - 1]["info"] > 0: break
        out.append(p)
    return out


def judge(path):
    rows = [json.loads(x) for x in open(path) if x.strip()]
    hdr = next((r for r in rows if r.get("header")), {}); R = [r for r in rows if not r.get("header")]
    if not R: print(f"== {path}: empty"); return {}
    r0 = R[0]; cell = f"in={r0['input_dtype']} robust={r0['robust']} prec={r0['precision']} compile={r0['compile']}"
    print(f"== {path}: {hdr.get('gpu', hdr.get('device'))} | {cell} | runs {len(R)}")
    res = {}
    if r0["compile"]:
        # E7: first nominal kappa with orth > 1e-2 per (m, l, spec, method), compiled vs eager
        cells = {}
        for r in R: cells.setdefault((r["m"], r["l"], r["spec"], r["method"]), []).append(r)
        agree = 0
        for k, v in cells.items():
            v.sort(key=lambda r: r["kappa"])
            fc = next((math.log10(r["kappa"]) for r in v if r["orth_compiled"] > 1e-2), 99)
            fe = next((math.log10(r["kappa"]) for r in v if r["orth_eager"] > 1e-2), 99)
            agree += abs(fc - fe) <= 0.5 + 1e-9
        ok = [r for r in R if r["orth_eager"] < 1e-4]
        close = sum(r["orth_compiled"] <= 10 * r["orth_eager"] + 1e-12 for r in ok)
        print(f"   E7: cliff within one half-decade in {agree}/{len(cells)} cells; orth_compiled <= 10 x eager in {close}/{len(ok)}")
        res["E7_compile"] = agree >= 0.9 * len(cells) and close >= 0.95 * len(ok)
        print("   verdicts:", {k: ("PASS" if v else "FAIL") for k, v in res.items()}); return res
    broken = lambda r: any(p["info"] > 0 for p in r["passes"])
    # E1 componentwise input floor (low-precision inputs)
    if r0["input_dtype"] in ("bfloat16", "float16", "float8_e4m3fn"):
        uin = U_IN[r0["input_dtype"]]
        sel = [r for r in R if r["kappa"] >= 10 / uin * 0.99]
        w = sum(1 / 3 <= r["kappa_eff"] / r["kappa_cw"] <= 3 for r in sel if math.isfinite(r["kappa_cw"]) and math.isfinite(r["kappa_eff"]))
        print(f"   E1: kappa(fl(Y)) within 3x of componentwise model in {w}/{len(sel)}; kappa_eff range "
              f"{min(r['kappa_eff'] for r in R):.2g}..{max(r['kappa_eff'] for r in R):.2g}")
        res["E1_input_floor"] = w >= 0.9 * len(sel)
    # E2 Theorem D, no margin
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
    print(f"   E2 Theorem D violations: (a) {va}/{na}, (c) {vb}/{na}, (b) {vc}/{nc}; D1 {vd}/{nd}")
    res["E2_theoremD"] = va + vb + vc + vd == 0
    # E3 lam_hat rule on FP32 passes
    P = [p for r in R for p in live(r) if p.get("dtype", "float32") == "float32"]
    neg = [p["info"] > 0 for p in P if p["lam_hat"] < -2 * u]; pos = [p["info"] == 0 for p in P if p["lam_hat"] > 5 * u]
    print(f"   E3: lam_hat<-2u failures {sum(neg)}/{len(neg)}; lam_hat>5u successes {sum(pos)}/{len(pos)}")
    res["E3_lamhat"] = sum(neg) >= 0.99 * len(neg) and sum(pos) >= 0.99 * len(pos)
    # E4 library default; E5 detector (precision highest only)
    lib = [r["orth_lib_default"] for r in R]
    print(f"   library default (FP32 output): orth max {max(lib):.1e}; >= 1e-4 in {sum(x >= 1e-4 for x in lib)}/{len(lib)}")
    nb = [r for r in R if not broken(r)]
    bad = [r for r in nb if r["orth"] > 1e-4]; good = [r for r in nb if r["orth"] <= 1e-5]
    fb = sum(r["passes"][-1]["Lmin"] < TH for r in bad); fg = sum(r["passes"][-1]["Lmin"] < TH for r in good)
    print(f"   detector: caught {fb}/{len(bad)}; false alarms {fg}/{len(good)}")
    if r0["precision"] == "highest":
        res["E4_library"] = all(x < 1e-4 for x in lib)
        res["E5_detector"] = fb >= 0.99 * len(bad) and fg <= 0.005 * max(len(good), 1)
    else:
        print("   (precision != highest: E4/E5 reported only)")
    if r0["robust"]:
        nbk = sum(broken(r) for r in R)
        print(f"   E6 robust: runs with any breakdown {nbk}/{len(R)}")
        res["E6_robust"] = nbk == 0
    eps = sorted(p["eps_gram"] / u for r in R for p in live(r)[1:2] if p.get("dtype") == "float32")
    if eps: print(f"   eps_2/u median {eps[len(eps)//2]:.1f}")
    print("   verdicts:", {k: ("PASS" if v else "FAIL") for k, v in res.items()})
    return res


if __name__ == "__main__":
    for p in sys.argv[1:]:
        judge(p)
