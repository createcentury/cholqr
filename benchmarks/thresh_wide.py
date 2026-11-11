"""Does the last-pass breakdown threshold Lmin_last < 0.195 still catch degraded results at wider matrices?
(CPU only; pre-registered, see benchmarks/prereg/PREREG_THRESH_WIDE_2026-11-12.md; judge: benchmarks/thresh_wide_judge.py)

The threshold sqrt(64 u / 1e-4) = 0.195 (u = 2^-24) was derived and validated for l in {64, 276} (rounding note,
Sec. 7). This script repeats the same validation protocol at (m, l) = (4096, 512) and (8192, 1024): an instrumented
copy of cholqr's passes (scqr3 and cqr2, FP32, precision "highest") logs, per pass, the Cholesky info flag (a non-finite
factor counts as failed) and the smallest diagonal entry Lmin of the factor; the final orthogonality
||Q^T Q - I|| is computed in FP64 (spectral and Frobenius).

Gram accumulation orders: "lib" (cholqr's own _gram: split-K over 256-row blocks for m >= 2048) and "pw8" (pairwise
over 8-row blocks: the 8-row block Grams are summed in a balanced binary tree; computed in aligned subtrees of 64 blocks,
which is the same tree because m / 8 is a power of two here). In "lib" mode the mirror's Q is compared bit for bit with
cholqr.core._cqr on the same input (field lib_identical), and the mirror's breakdown flag with _cqr's flag.

Inputs: Y = U diag(sigma) V^T rounded to FP32, the six spectra of benchmarks/device_sweep.py (same construction),
nominal cond 10^2, 10^2.5, ..., 10^10, draws 1-3, seed key prefix "TW-" (new draws).

Usage:  python benchmarks/thresh_wide.py --out DIR --shard I --nshard N      (appends DIR/runs_I.jsonl; resumable)
        python benchmarks/thresh_wide.py --smoke                            (two runs, prints timings only)
"""
import argparse, json, math, os, platform, sys, time, zlib
import torch
import cholqr
from cholqr import core

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from device_sweep import make64, SPECTRA  # noqa: E402  (same input construction as the device sweep)

torch.set_float32_matmul_precision("highest")
EPS = torch.finfo(torch.float32).eps          # 2^-23 = 2u: the later-pass shift in cholqr
SIZES = ((4096, 512), (8192, 1024))
KAPPAS = [10 ** (x / 2) for x in range(4, 21)]
DRAWS = (1, 2, 3)
METHODS = ("scqr3", "cqr2")
MODES = ("lib", "pw8")
TH = core._BREAKDOWN_LMIN


def gram_pw8(Q, sub=64):
    """Pairwise Gram over 8-row blocks (balanced tree), computed per aligned subtree of `sub` blocks (memory)."""
    m, r = Q.shape
    nb = m // 8
    assert m % 8 == 0 and nb & (nb - 1) == 0 and nb % sub == 0
    B = Q.reshape(nb, 8, r)

    def tree(Gs):
        while Gs.shape[0] > 1:
            Gs = Gs[0::2] + Gs[1::2]
        return Gs[0]

    parts = torch.stack([tree(B[j:j + sub].mT @ B[j:j + sub]) for j in range(0, nb, sub)])
    return tree(parts)


def mirror(Y, method, mode):
    """cholqr 0.5 _cqr passes (FP32, robust=False); returns Q and per-pass (info, Lmin)."""
    l = Y.shape[-1]
    n_pass = {"scqr3": 3, "cqr2": 2}[method]
    Q = Y.float()
    eye = torch.eye(l, dtype=torch.float32)
    passes = []
    for i in range(n_pass):
        if i == 0:
            Q, _ = core._prescale(Q)
        d = Q.norm(dim=-2, keepdim=True).clamp_min(1e-30)
        Q = Q / d
        G = core._gram(Q) if mode == "lib" else gram_pw8(Q)
        s = 1e-6 * l if (method == "scqr3" and i == 0) else EPS
        L, info = torch.linalg.cholesky_ex(G + (s + 1e-30) * eye)
        L_inv = torch.linalg.solve_triangular(L, eye, upper=False)
        dg = L.diagonal()
        fail = int(info) != 0 or not bool(torch.isfinite(dg).all())
        passes.append(dict(k=i + 1, info=int(info), fail=fail, Lmin=float(dg.amin())))
        if i == n_pass - 1 and not fail:
            Li64 = torch.linalg.solve_triangular(L.double(), torch.eye(l, dtype=torch.float64), upper=False)
            passes[-1]["invL2"] = float(torch.linalg.eigvalsh(Li64 @ Li64.mT)[-1])
        Q = Q @ L_inv.mT if not fail else Q
    return Q, passes


def orth(Q):
    Qd = Q.double()
    D = Qd.mT @ Qd - torch.eye(Q.shape[-1], dtype=torch.float64)
    if not torch.isfinite(D).all():
        return float("inf"), float("inf")
    return float(torch.linalg.eigvalsh(D).abs().amax()), float(torch.linalg.matrix_norm(D))


def jobs():
    J = []
    for (m, l) in SIZES:
        for spec in SPECTRA:
            for kappa in KAPPAS:
                for draw in DRAWS:
                    J.append((m, l, spec, kappa, draw))
    return J


def run_input(m, l, spec, kappa, draw, check_lib=True):
    key = f"TW-{m}-{l}-{spec}-{round(math.log10(kappa) * 2)}-{draw}"
    g = torch.Generator().manual_seed(7000 + zlib.crc32(key.encode()) % 100000)
    Y = make64(m, l, kappa, spec, g).float()
    out = []
    for method in METHODS:
        for mode in MODES:
            t0 = time.time()
            Q, passes = mirror(Y, method, mode)
            o2, oF = orth(Q)
            rec = dict(key=key, m=m, l=l, spec=spec, kappa=kappa, draw=draw, method=method, mode=mode, passes=passes,
                       orth2_final=o2, orthF_final=oF,
                       flag=any(p["fail"] for p in passes) or passes[-1]["Lmin"] < TH)
            if mode == "lib" and check_lib:
                Ql, _, bad = core._cqr(Y, method, False, False, None, 1e-6, None, 1.25, None)
                rec.update(lib_identical=bool(torch.equal(Ql, Q)), lib_flag=bool(bad))
            rec["sec"] = round(time.time() - t0, 2)
            out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.smoke:                       # timings only; nothing is written, no result is printed
        for (m, l) in SIZES:
            t0 = time.time()
            run_input(m, l, "geometric", 1e3, 99)
            print(f"smoke: m={m} l={l}: {time.time() - t0:.1f} s per input (4 method/mode runs)")
        return
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"runs_{a.shard}.jsonl")
    done = set()
    if os.path.exists(path):
        for x in open(path):
            if x.strip() and not x.startswith('{"header"'):
                r = json.loads(x); done.add(r["key"])
    J = [j for idx, j in enumerate(jobs()) if idx % a.nshard == a.shard]
    with open(path, "a") as f:
        if not done:
            f.write(json.dumps(dict(header=True, torch=torch.__version__, cholqr=cholqr.__version__,
                                    python=platform.python_version(), machine=platform.machine(),
                                    threads=a.threads, threshold=TH)) + "\n")
        for (m, l, spec, kappa, draw) in J:
            key = f"TW-{m}-{l}-{spec}-{round(math.log10(kappa) * 2)}-{draw}"
            if key in done:
                continue
            for rec in run_input(m, l, spec, kappa, draw):
                f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
