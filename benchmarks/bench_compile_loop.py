"""Cost of the compiled-loop pattern of the note's Sec. 9 (descriptive, no prediction):

    qr = torch.compile(cholqr.cholesky_qr)        # in the graph: no check, no fallback, no host sync
    Q = qr(Y)
    if step % N == 0:                             # outside the compiled region
        if (cholqr.orthogonality_bound(Q) > 1e-4).any():
            Q = cholqr.cholesky_qr(Y, verify=True)

Configurations per shape: compiled call alone ("compiled"), the pattern with N in {1, 10, 100}, and for reference the
eager default call ("eager_default", which has its own host sync) and orthogonality_bound alone ("bound_only").
Inputs: a pool of 8 well-conditioned FP32 Gaussian matrices, cycled (values change every step, shape fixed).
Two timing modes, each in `rounds` interleaved rounds of `steps` steps:
  synced     torch.cuda.synchronize() before and after every step: per-step times (median, mean, p99, max; check steps
             and other steps separately);
  pipelined  one synchronize before and after a block of `steps` steps: wall time / steps (what a training loop that
             does not sync sees; the check's .any() in `if` forces a host sync every N steps).
Recompilations: torch._dynamo.utils.counters before/after warm-up and after all measurements (run with
TORCH_LOGS=recompiles,graph_breaks to log them as well).
Usage: python benchmarks/bench_compile_loop.py --out compile_loop.json [--steps 1000 --rounds 3]"""
import argparse, copy, json, statistics, subprocess, time
import torch
import torch._dynamo
import cholqr

SHAPES = [(40, 768, 210), (8, 4096, 276)]
NS = (1, 10, 100)


def counters():
    return {k: dict(v) for k, v in copy.deepcopy(torch._dynamo.utils.counters).items()}


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warm", type=int, default=30); ap.add_argument("--precision", default="highest")
    a = ap.parse_args()
    torch.set_float32_matmul_precision(a.precision)
    dev = torch.device("cuda"); sync = torch.cuda.synchronize
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        smi = None
    res = dict(torch=torch.__version__, cuda=torch.version.cuda, cholqr=cholqr.__version__,
               gpu=torch.cuda.get_device_name(dev), nvidia_smi=smi, precision=a.precision, steps=a.steps,
               rounds=a.rounds, warm=a.warm, shapes=[])
    g = torch.Generator().manual_seed(0)
    for b, m, r in SHAPES:
        torch._dynamo.reset(); torch._dynamo.utils.counters.clear()
        pool = [torch.randn(b, m, r, generator=g).to(dev) for _ in range(8)]
        qr = torch.compile(cholqr.cholesky_qr)
        sync(); t = time.perf_counter(); qr(pool[0]); sync(); first_call_s = time.perf_counter() - t
        for i in range(a.warm):
            qr(pool[i % 8])
        for i in range(a.warm):
            cholqr.orthogonality_bound(qr(pool[i % 8]))
            cholqr.cholesky_qr(pool[i % 8])
        sync()
        c_warm = counters()
        n_fallback = 0

        def step(i, N):
            nonlocal n_fallback
            Y = pool[i % 8]
            Q = qr(Y)
            if N and i % N == 0:
                if (cholqr.orthogonality_bound(Q) > 1e-4).any():
                    n_fallback += 1
                    Q = cholqr.cholesky_qr(Y, verify=True)
            return Q

        cfgs = {"compiled": lambda i: step(i, 0), **{f"N{N}": (lambda i, N=N: step(i, N)) for N in NS},
                "eager_default": lambda i: cholqr.cholesky_qr(pool[i % 8]),
                "bound_only": lambda i: cholqr.orthogonality_bound(pool[i % 8])}
        synced = {k: [] for k in cfgs}; pipe = {k: [] for k in cfgs}
        for _ in range(a.rounds):
            for k, fn in cfgs.items():
                ts = []
                for i in range(a.steps):
                    sync(); t = time.perf_counter(); fn(i); sync(); ts.append(1e3 * (time.perf_counter() - t))
                synced[k].append(ts)
                sync(); t = time.perf_counter()
                for i in range(a.steps):
                    fn(i)
                sync(); pipe[k].append(1e3 * (time.perf_counter() - t) / a.steps)
        c_end = counters()
        row = dict(shape=[b, m, r], first_call_s=first_call_s, counters_after_warm=c_warm, counters_end=c_end,
                   fallbacks_triggered=n_fallback, configs={})
        for k in cfgs:
            allts = [x for ts in synced[k] for x in ts]
            d = dict(synced_median_ms=statistics.median(allts), synced_mean_ms=statistics.mean(allts),
                     synced_p99_ms=pct(allts, 99), synced_max_ms=max(allts),
                     synced_round_medians_ms=[statistics.median(ts) for ts in synced[k]],
                     pipelined_ms_per_step=statistics.median(pipe[k]), pipelined_rounds_ms=pipe[k])
            if k.startswith("N"):
                N = int(k[1:])
                chk = [ts[i] for ts in synced[k] for i in range(a.steps) if i % N == 0]
                oth = [ts[i] for ts in synced[k] for i in range(a.steps) if i % N != 0]
                d.update(check_steps=len(chk), check_step_median_ms=statistics.median(chk),
                         other_step_median_ms=statistics.median(oth) if oth else None)
            row["configs"][k] = d
        base = row["configs"]["compiled"]
        for k, d in row["configs"].items():
            d["synced_mean_over_compiled"] = d["synced_mean_ms"] / base["synced_mean_ms"]
            d["pipelined_over_compiled"] = d["pipelined_ms_per_step"] / base["pipelined_ms_per_step"]
        res["shapes"].append(row)
        print(json.dumps({k: (v if k != "configs" else {c: {x: y for x, y in dd.items() if not isinstance(y, list)}
                                                          for c, dd in v.items()})
                          for k, v in row.items() if not k.startswith("counters")}), flush=True)
        print("counters_after_warm", c_warm, "\ncounters_end", c_end, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
