"""Cost of cholqr's opt-in certificate on a device: cholesky_qr(Y) (default) vs cholesky_qr(Y, verify=True), and
orthogonality_bound(Q) alone, for well-conditioned FP32 inputs (no fallback triggered).
Median of `reps` calls after `warm` warm-up calls, device synchronized around every call. The default and verify
timings are taken in `rounds` interleaved rounds (default, verify, bound, default, verify, bound, ...); the reported
value is the median over all rounds, and the per-round medians are kept to show the spread.
Usage: python benchmarks/bench_verify_cost.py --device cuda --out verify_cost.json"""
import argparse, json, statistics, subprocess, time
import torch
import cholqr

SHAPES = [(1, 1024, 64), (1, 4096, 276), (16, 2048, 128), (40, 768, 210), (8, 4096, 276), (16, 1024, 64),
          (4, 8192, 128)]


def times(fn, sync, reps, warm):
    for _ in range(warm):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return [1e3 * x for x in ts]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default="highest", choices=["highest", "high", "medium"])
    ap.add_argument("--reps", type=int, default=50); ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args(); dev = torch.device(a.device)
    torch.set_float32_matmul_precision(a.precision)
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(dev.type, lambda: None)
    gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" else None
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        smi = None
    res = dict(precision=a.precision, device=str(dev), torch=torch.__version__, cuda=torch.version.cuda,
               cholqr=cholqr.__version__, gpu=gpu, nvidia_smi=smi, reps=a.reps, warm=a.warm, rounds=a.rounds,
               dtype="float32", rows=[])
    g = torch.Generator().manual_seed(0)
    for b, m, r in SHAPES:
        Y = torch.randn(b, m, r, generator=g).to(dev)
        if b == 1:
            Y = Y[0]
        # no fallback / no rejection: raise instead of falling back, for both paths
        Q = cholqr.cholesky_qr(Y, on_breakdown="raise")
        Qv = cholqr.cholesky_qr(Y, verify=True, on_breakdown="raise")
        bound = cholqr.orthogonality_bound(Q)
        row = dict(shape=[b, m, r], bound_max=float(bound.max()), verify_equals_default=bool(torch.equal(Q, Qv)))
        acc = {"default": [], "verify": [], "bound_only": []}
        per_round = {k: [] for k in acc}
        for _ in range(a.rounds):
            for k, fn in (("default", lambda: cholqr.cholesky_qr(Y)),
                          ("verify", lambda: cholqr.cholesky_qr(Y, verify=True)),
                          ("bound_only", lambda: cholqr.orthogonality_bound(Q))):
                ts = times(fn, sync, a.reps, a.warm)
                acc[k] += ts; per_round[k].append(statistics.median(ts))
        for k in acc:
            row[k + "_ms"] = statistics.median(acc[k]); row[k + "_ms_rounds"] = per_round[k]
        row["verify_overhead_pct"] = 100 * (row["verify_ms"] / row["default_ms"] - 1)
        row["bound_over_default_pct"] = 100 * row["bound_only_ms"] / row["default_ms"]
        res["rows"].append(row); print(json.dumps(row), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
