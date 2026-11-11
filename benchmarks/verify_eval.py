"""Evaluation of verify=True: soundness (V1), slack (V2) and overhead (V3) of the FP64 orthogonality certificate.

    python benchmarks/verify_eval.py --part v12 --device cpu --size small --out v12_small.jsonl
    python benchmarks/verify_eval.py --part v12 --device cpu --size large --out v12_large.jsonl
    python benchmarks/verify_eval.py --part v12 --device mps --size small --out v12_mps.jsonl
    python benchmarks/verify_eval.py --part v3 --device cpu --out v3_cpu.json
    python benchmarks/verify_eval.py --part summary v12_*.jsonl

Reference ("true") ||Q^T Q - I||_F: exact (Fraction) for m*l^2 <= 3e5; otherwise each entry of Q^T Q - I correctly
rounded by math.fsum of the exact FP64 products, squares summed exactly, so true is within ref*(1 -+ 2u).
"""
import argparse
import json
import math
import sys
import time
from fractions import Fraction

import numpy as np
import torch

import cholqr
import cholqr.core as core
from cholqr import cholesky_qr, orthogonality_bound

U = 2.0 ** -53


# ---------------------------------------------------------------- reference
def ref_err(Q):
    """(ref, lo, hi, kind): lo <= ||Q^T Q - I||_F <= hi."""
    Q = Q.detach().cpu().float().double().numpy()
    m, l = Q.shape
    if not np.isfinite(Q).all():
        return math.inf, math.inf, math.inf, "nonfinite"
    if m * l * l <= 300_000:
        S = 149
        N = [[int(Fraction(float(x)) * 2 ** S) for x in row] for row in Q.T]
        one = 2 ** (2 * S)
        tot = 0
        for i in range(l):
            for j in range(i, l):
                a = sum(x * y for x, y in zip(N[i], N[j])) - (one if i == j else 0)
                tot += (1 if i == j else 2) * a * a
        val = Fraction(tot, one * one)
        r = math.sqrt(val)
        # sqrt of an exact rational: bracket it by the neighbouring floats
        lo, hi = r, r
        while Fraction(lo) ** 2 > val:
            lo = math.nextafter(lo, 0)
        while Fraction(hi) ** 2 < val:
            hi = math.nextafter(hi, math.inf)
        return r, lo, hi, "exact"
    tot = Fraction(0)
    for i in range(l):
        P = Q[:, i:i + 1] * Q[:, i:]                         # exact FP64 products of FP32 values
        for jj in range(P.shape[1]):
            t = math.fsum(list(P[:, jj]) + ([-1.0] if jj == 0 else []))
            tot += (1 if jj == 0 else 2) * Fraction(t) ** 2
    r = math.sqrt(tot)
    return r, r * (1 - 2 * U) * (1 - U), r * (1 + 2 * U) * (1 + U), "fsum"


# ---------------------------------------------------------------- inputs
def spectrum(m, r, kappa, kind, seed):
    g = torch.Generator().manual_seed(seed)
    Uq = torch.linalg.qr(torch.randn(m, r, generator=g, dtype=torch.float64))[0]
    V = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0]
    if kind == "geometric":
        s = torch.logspace(0, -math.log10(kappa), r, dtype=torch.float64)
    else:
        s = torch.linspace(1, kappa ** 0.5, r, dtype=torch.float64) ** -2
    return ((Uq * s) @ V.mT).float()


def tf32_round(X):
    b = X.contiguous().view(torch.int32)
    return ((b + 0x1000) & ~0x1FFF).view(torch.float32)


class EmulatedTF32:
    def __enter__(self):
        self.mm, self.gmm = core._mm, core._gram_mm

        def tmm(A, B):
            if A.dtype != torch.float32 or A.device.type != "cpu":
                return A @ B
            return (tf32_round(A).double() @ tf32_round(B.contiguous()).double()).float()
        core._mm = lambda A, B, tf32: tmm(A, B)
        core._gram_mm = lambda A, tf32: tmm(A.mT.contiguous(), A)

    def __exit__(self, *a):
        core._mm, core._gram_mm = self.mm, self.gmm


class PlainTF32:
    """CUDA: let TF32 run uncompensated (0.4 behaviour) by disabling cholqr's 3xTF32 detection."""
    def __enter__(self):
        self.f = core._tf32_active
        core._tf32_active = lambda Q: False

    def __exit__(self, *a):
        core._tf32_active = self.f


def cases(size, device):
    """Yield (name, Y, kwargs, context)."""
    from contextlib import nullcontext
    if size == "small":
        shapes, seeds = [(200, 12)], range(3)
    else:
        shapes, seeds = [(1024, 64), (4096, 276)], range(1)
    kappas = [1e0, 1e2, 1e4, 1e6, 1e7, 1e8, 1e10, 1e12]
    tf32ctx = EmulatedTF32 if device == "cpu" else (PlainTF32 if device == "cuda" else None)
    for (m, l) in shapes:
        for kind in ("geometric", "powerlaw"):
            for kappa in kappas:
                for seed in seeds:
                    Y = spectrum(m, l, kappa, kind, seed)
                    tag = f"{m}x{l}/{kind}/{kappa:g}/s{seed}"
                    for method in ("scqr3", "cqr2", "rcqr"):
                        yield f"cholqr/{method}/{tag}", Y, dict(method=method), nullcontext
                    if device != "mps":                                     # no FP64 on MPS
                        yield f"cholqr/scqr3-robust/{tag}", Y, dict(robust=True), nullcontext
                    if tf32ctx is not None:
                        for method in ("scqr3", "cqr2"):
                            yield f"tf32/{method}/{tag}", Y, dict(method=method), tf32ctx
        # low-precision inputs and outputs
        Y = spectrum(m, l, 1e3, "geometric", 7)
        dts = [torch.float16, torch.bfloat16] + [d for d in (getattr(torch, "float8_e4m3fn", None),
                                                              getattr(torch, "float8_e5m2", None)) if d is not None]
        for dt in dts:
            yield f"lowp-in/{dt}/{m}x{l}", Y.to(dt), dict(out_dtype=torch.float32), nullcontext
            yield f"lowp-inout/{dt}/{m}x{l}", Y.to(dt), dict(), nullcontext
        if device != "mps":
            yield f"out-f64/{m}x{l}", Y.double(), dict(), nullcontext
        # near rank deficient, batched
        g = torch.Generator().manual_seed(1)
        Y2 = Y.clone(); Y2[:, 3] = Y2[:, 2] + 1e-7 * torch.randn(m, generator=g)
        Y3 = Y.clone(); Y3[:, 0] = 0
        Y4 = Y.clone(); Y4[:, 5] = Y4[:, 4]
        yield f"rankdef-batch/{m}x{l}", torch.stack([Y, Y2, Y3, Y4]), dict(), nullcontext


def adversarial(size):
    """Directly constructed matrices (no cholqr), CPU, FP32 unless stated."""
    m, l = (96, 8) if size == "small" else (4096, 276)
    g = torch.Generator().manual_seed(3)
    Q0 = torch.linalg.qr(torch.randn(m, l, generator=g, dtype=torch.float64))[0]
    out = []
    for eps in (2e-4, 1e-4, 5e-5, 1e-6):
        E = torch.randn(m, l, generator=g, dtype=torch.float64)
        out.append((f"adv/perturb{eps:g}", (Q0 + eps / 2 * E / E.norm()).float()))
    sc = torch.ones(l, dtype=torch.float64); sc[0] = 2.0 ** 60
    out.append(("adv/col-2^60", (Q0 * sc).float()))
    sc = torch.ones(l, dtype=torch.float64); sc[1] = 2.0 ** -60
    out.append(("adv/col-2^-60", (Q0 * sc).float()))
    out.append(("adv/subnormal", (Q0 * 2.0 ** -140).float()))
    Qd = Q0.float().clone(); Qd[:, 1] = Qd[:, 0]
    out.append(("adv/dup-col", Qd))
    out.append(("adv/zero", torch.zeros(m, l)))
    out.append(("adv/near-exact-fp32", Q0.float()))
    for dt in (torch.float16, torch.bfloat16, getattr(torch, "float8_e4m3fn", None)):
        if dt is not None:
            out.append((f"adv/round-{dt}", Q0.to(dt)))
    Qn = Q0.float().clone(); Qn[0, 0] = math.nan
    out.append(("adv/nan", Qn))
    Qi = Q0.float().clone(); Qi[0, 0] = math.inf
    out.append(("adv/inf", Qi))
    return out


# ---------------------------------------------------------------- V1/V2 records
def record(name, Q, flagged=None, e2e=None):
    Qs = Q.reshape(-1, *Q.shape[-2:])
    v = orthogonality_bound(Qs).cpu()
    Qd = Qs.detach().cpu().float().double()
    e_hat = (Qd.mT @ Qd - torch.eye(Qs.shape[-1], dtype=torch.float64)).square().sum((-2, -1)).sqrt()
    recs = []
    for k in range(Qs.shape[0]):
        ref, lo, hi, kind = ref_err(Qs[k])
        vk, ek = float(v[k]), float(e_hat[k])
        tols = [1e-4, ek, (ek + ref) / 2] + [ref * (1 - 2.0 ** -j) for j in range(1, 53)] + \
               [ref * (1 + 2.0 ** -j) for j in range(1, 53)]
        viol = amb = 0
        for tol in tols:
            if not (tol > 0) or not (vk <= tol):
                continue
            if lo > tol:
                viol += 1
            elif hi > tol:
                amb += 1
        recs.append(dict(case=name, k=k, m=Qs.shape[-2], l=Qs.shape[-1], dtype=str(Q.dtype), ref=ref, ref_lo=lo,
                         ref_hi=hi, ref_kind=kind, e_hat=ek, bound=vk, slack=vk - ek,
                         dev=(abs(ek - ref) if math.isfinite(ref) else None),
                         cert_1e4=bool(vk <= 1e-4), true_gt_1e4=bool(lo > 1e-4), viol=viol, amb=amb,
                         bound_ge_true=bool(vk >= hi) if math.isfinite(hi) else (not math.isfinite(vk)),
                         flagged=None if flagged is None else bool(flagged.reshape(-1)[k]), e2e=e2e))
    return recs


def run_v12(args):
    dev = torch.device(args.device)
    if args.precision and dev.type == "cuda":
        torch.set_float32_matmul_precision(args.precision)
    out = open(args.out, "w")
    n = 0
    t0 = time.time()
    for name, Y, kw, ctx in cases(args.size, dev.type):
        try:
            Yd = Y.to(dev)
        except (RuntimeError, TypeError) as ex:                 # e.g. float8_e5m2 on MPS
            print(f"skip {name}: {ex}", file=sys.stderr)
            continue
        with ctx():
            try:
                Q, _, bad = core._cqr(Yd, kw.get("method", "scqr3"), kw.get("robust", False), False,
                                      kw.get("out_dtype"), 1e-6, None, 1.25, None)
            except (RuntimeError, TypeError) as ex:             # e.g. float8_e5m2 -> float32 on MPS
                print(f"skip {name}: {ex}", file=sys.stderr)
                continue
            # end-to-end: verify=True, fallback and raise
            try:
                Qv = cholesky_qr(Yd, verify=True, **kw)
                e2e_f = "returned"
            except torch.linalg.LinAlgError:
                Qv, e2e_f = None, "LinAlgError"
            except ValueError:                                  # low-precision output without explicit tol (0.5 API)
                Qv, e2e_f = None, "ValueError"
            try:
                cholesky_qr(Yd, verify=True, on_breakdown="raise", **kw)
                e2e_r = "returned"
            except torch.linalg.LinAlgError:
                e2e_r = "LinAlgError"
            except ValueError:
                e2e_r = "ValueError"
        for r in record(name, Q, flagged=bad):
            r["src"] = "raw"; r["e2e_raise"] = e2e_r; r["e2e_fallback"] = e2e_f
            out.write(json.dumps(r) + "\n"); n += 1
        if Qv is not None:
            for r in record(name, Qv, e2e="fallback-returned"):
                r["src"] = "verified"
                out.write(json.dumps(r) + "\n"); n += 1
        out.flush()
        print(f"[{time.time() - t0:7.1f}s] {name} ({n} records)", file=sys.stderr, flush=True)
    if dev.type == "cpu":
        for name, Q in adversarial(args.size):
            for r in record(name, Q):
                r["src"] = "adversarial"
                out.write(json.dumps(r) + "\n"); n += 1
            out.flush()
    out.close()


# ---------------------------------------------------------------- V3
def run_v3(args):
    dev = torch.device(args.device)
    if args.precision and dev.type == "cuda":
        torch.set_float32_matmul_precision(args.precision)
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(dev.type, lambda: None)
    res = []
    for (m, l, b) in [(4096, 276, 1), (4096, 276, 8), (1024, 64, 16), (8192, 128, 4)]:
        Y = torch.randn(b, m, l, generator=torch.Generator().manual_seed(0)).to(dev)
        if b == 1:
            Y = Y[0]
        row = dict(m=m, l=l, batch=b, device=args.device, torch=torch.__version__)
        for label, kw in (("default", {}), ("verify", dict(verify=True))):
            for _ in range(3):
                cholesky_qr(Y, **kw); sync()
            ts = []
            for _ in range(args.reps):
                sync(); t = time.perf_counter(); cholesky_qr(Y, **kw); sync(); ts.append(time.perf_counter() - t)
            row[label + "_ms"] = 1e3 * float(np.median(ts))
        Q = cholesky_qr(Y)
        for _ in range(2):
            orthogonality_bound(Q)
        ts = []
        for _ in range(args.reps):
            sync(); t = time.perf_counter(); orthogonality_bound(Q); sync(); ts.append(time.perf_counter() - t)
        row["bound_only_ms"] = 1e3 * float(np.median(ts))
        row["overhead"] = row["verify_ms"] / row["default_ms"] - 1
        print(json.dumps(row), flush=True)
        res.append(row)
    json.dump(res, open(args.out, "w"), indent=1)


# ---------------------------------------------------------------- summary
def summary(files):
    R = [json.loads(line) for f in files for line in open(f)]
    fin = [r for r in R if r["dev"] is not None]
    print(f"records {len(R)} (finite {len(fin)}), files {files}")
    print("V1 violations", sum(r["viol"] for r in R), " ambiguous", sum(r["amb"] for r in R),
          " bound<true", sum(not r["bound_ge_true"] for r in R))
    print("V1 certified@1e-4 with true>1e-4:", sum(r["cert_1e4"] and r["true_gt_1e4"] for r in R))
    nf = [r for r in R if r["dev"] is None]
    print("non-finite matrices:", len(nf), " certified:", sum(r["cert_1e4"] for r in nf))
    ver = [r for r in R if r.get("src") == "verified"]
    print("end-to-end verified (fallback) returned:", len(ver), " true>1e-4:", sum(r["ref_lo"] > 1e-4 for r in ver),
          " max true", max((r["ref_hi"] for r in ver), default=None))
    raw = [r for r in R if r.get("src") == "raw"]
    deg = [r for r in raw if r["true_gt_1e4"]]
    print(f"raw outputs with true>1e-4: {len(deg)}; detector flagged {sum(bool(r['flagged']) for r in deg)}; "
          f"certificate rejected {sum(not r['cert_1e4'] for r in deg)}")
    ok_raw = [r for r in raw if r["ref_hi"] <= 1e-4 * (1 - 1e-5)]
    print(f"raw outputs with true<=1e-4(1-1e-5): {len(ok_raw)}; certified {sum(r['cert_1e4'] for r in ok_raw)}")
    e2e = [r for r in raw if r["k"] == 0]
    from collections import Counter
    print("e2e fallback outcomes", Counter(r["e2e_fallback"] for r in e2e), " raise outcomes",
          Counter(r["e2e_raise"] for r in e2e))
    print("e2e LinAlgError under fallback:", sorted({r["case"] for r in e2e if r["e2e_fallback"] != "returned"}))
    ratios = [r["dev"] / r["slack"] for r in fin if r["slack"] > 0]
    print("V2 |e_hat-true| > slack:", sum(r["dev"] > r["slack"] for r in fin),
          f" max dev/slack {max(ratios):.3g}  median {float(np.median(ratios)):.3g}")
    print("slack range", min(r["slack"] for r in fin), max(r["slack"] for r in fin))
    for kind in ("exact", "fsum"):
        print(kind, sum(r["ref_kind"] == kind for r in R))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["v12", "v3", "summary"], required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--size", default="small", choices=["small", "large"])
    ap.add_argument("--precision", default=None)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="verify_out.jsonl")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    {"v12": run_v12, "v3": run_v3, "summary": lambda a: summary(a.files)}[a.part](a)
