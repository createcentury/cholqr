"""Cost of cholqr's breakdown check on a device: on_breakdown="fallback" (default, one host sync) vs "ignore" (0.4
behaviour, no sync) vs torch.compile vs torch.linalg.qr, for well-conditioned inputs (no fallback triggered).
Usage: python benchmarks/bench_overhead.py --device cuda --out overhead.json"""
import argparse, json, statistics, time
import torch
import cholqr

SHAPES = [(1, 1024, 64), (1, 4096, 276), (16, 2048, 128), (40, 768, 210), (8, 4096, 276), (64, 1024, 32)]


def timeit(fn, dev, reps=50, warm=10):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        if dev.type == "cuda":
            torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize(); t = time.perf_counter(); fn(); torch.mps.synchronize()
        else:
            t = time.perf_counter(); fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = torch.device(a.device)
    torch.set_float32_matmul_precision("highest")
    comp = torch.compile(lambda Y: cholqr.cholesky_qr(Y, on_breakdown="ignore"), fullgraph=True)
    res = dict(device=str(dev), torch=torch.__version__, cholqr=cholqr.__version__,
               gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else None, rows=[])
    for b, m, r in SHAPES:
        Y = torch.randn(b, m, r, device=dev) if b > 1 else torch.randn(m, r, device=dev)
        row = dict(shape=[b, m, r])
        row["torch_qr_ms"] = timeit(lambda: torch.linalg.qr(Y, mode="reduced")[0], dev)
        row["ignore_ms"] = timeit(lambda: cholqr.cholesky_qr(Y, on_breakdown="ignore"), dev)
        row["fallback_ms"] = timeit(lambda: cholqr.cholesky_qr(Y), dev)
        try:
            row["compiled_ms"] = timeit(lambda: comp(Y), dev)
        except Exception as e:                      # compile unavailable (e.g. no Triton)
            row["compiled_ms"] = None; row["compile_error"] = type(e).__name__
        row["check_overhead_pct"] = 100 * (row["fallback_ms"] / row["ignore_ms"] - 1)
        res["rows"].append(row); print(row, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
