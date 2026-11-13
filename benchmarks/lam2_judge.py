"""Judge for prereg/PREREG_LAM2_2026-11-14.md. Usage: python benchmarks/lam2_judge.py lam2_cpu.jsonl"""
import json, sys, collections
rows = [r for r in map(json.loads, open(sys.argv[1])) if "lam2_over_u" in r]
by = collections.defaultdict(list)
for r in rows:
    by[r["l"]].append(r)
res = {}
for l, rs in sorted(by.items()):
    hi = [r for r in rs if r["lam2_over_u"] >= 10]
    lo = [r for r in rs if r["lam2_over_u"] < 0.1]
    fails = [r for r in rs if r["failed"]]
    res[l] = dict(n=len(rs), fail_total=len(fails), n_ge10u=len(hi), fail_ge10u=sum(r["failed"] for r in hi),
                  n_lt01u=len(lo), fail_frac_lt01u=(sum(r["failed"] for r in lo) / len(lo) if lo else None),
                  max_lam2_over_u_among_failed=max((r["lam2_over_u"] for r in fails), default=None),
                  pass2_first=sum(r["first_failed_pass"] == 2 for r in fails), of_fails=len(fails))
    print(l, res[l])
H1 = res.get(210, {}).get("fail_ge10u") == 0 and res[210]["n_ge10u"] >= 20
H2 = res.get(512, {}).get("fail_ge10u") == 0 and res[512]["n_ge10u"] >= 20
H3 = all((res[l]["fail_frac_lt01u"] or 0) >= 0.5 for l in res) and all(res[l]["n_lt01u"] >= 20 for l in res)
H4 = (res[512]["max_lam2_over_u_among_failed"] or 0) > (res[210]["max_lam2_over_u_among_failed"] or 0)
H5 = all(res[l]["pass2_first"] == res[l]["of_fails"] for l in res)
print(dict(H1=H1, H2=H2, H3=H3, H4=H4, H5=H5))
