"""Test of the size-dependent last-pass threshold tau(l) (benchmarks/thresh_size_rule.py) at fresh sizes and inputs.
(CPU only; pre-registered, see benchmarks/prereg/PREREG_THRESH_SIZE_2026-11-12.md; judge: benchmarks/thresh_size_judge.py)

Same measurement as benchmarks/thresh_wide.py (copied, not imported, so that script stays as run), at the fresh sizes
(m, l) = (3072, 384), (6144, 768), (12288, 1536) (aspect ratio m = 8 l, as in thresh_wide): an instrumented
copy of cholqr's passes (scqr3 and cqr2, FP32, precision "highest") logs, per pass, the Cholesky info flag (a non-finite
factor counts as failed) and the smallest diagonal entry Lmin of the factor; the final orthogonality
||Q^T Q - I|| is computed in FP64 (spectral and Frobenius).

Gram accumulation orders: "lib" (cholqr's own _gram: split-K over 256-row blocks for m >= 2048) and "pw8" (pairwise
over 8-row blocks: the 8-row block Grams are summed pairwise; computed in aligned subtrees of 64 blocks (balanced
binary trees), whose sums are then added pairwise level by level, an odd last element being carried up unchanged; for a
power-of-two block count this is the balanced tree of thresh_wide.py; here m / 8 = 384, 768, 1536 = 64 x 6, 12, 24).
In "lib" mode the mirror's Q is compared bit for bit with cholqr.core._cqr on the same input (field lib_identical), and the mirror's breakdown flag with _cqr's flag.

Inputs: Y = U diag(sigma) V^T rounded to FP32, the six spectra of benchmarks/device_sweep.py (same construction),
nominal cond 10^2, 10^2.5, ..., 10^10, draws 1-3, seed key prefix "TS-" (new draws, new sizes).
Each record keeps Lmin of every pass, so the judge can apply any threshold; the field "flag" is the cholqr 0.5 rule
(0.195), used only for the S0 agreement check with the library.

Usage:  python benchmarks/thresh_size.py --out DIR --shard I --nshard N      (appends DIR/runs_I.jsonl; resumable)
        python benchmarks/thresh_size.py --smoke                            (two runs, prints timings only)
"""
import argparse, json, math, os, platform, sys, time, zlib
import torch
import cholqr
from cholqr import core

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from device_sweep import make64, SPECTRA  # noqa: E402  (same input construction as the device sweep)

torch.set_float32_matmul_precision("highest")
EPS = torch.finfo(torch.float32).eps          # 2^-23 = 2u: the later-pass shift in cholqr
SIZES = ((3072, 384), (6144, 768), (12288, 1536))
KAPPAS = [10 ** (x / 2) for x in range(4, 21)]
DRAWS = (1, 2, 3)
METHODS = ("scqr3", "cqr2")
MODES = ("lib", "pw8")
TH = core._BREAKDOWN_LMIN


def gram_pw8(Q, sub=64):
    """Pairwise Gram over 8-row blocks (balanced tree), computed per aligned subtree of `sub` blocks (memory)."""
    m, r = Q.shape
    nb = m // 8
    assert m % 8 == 0 and nb % sub == 0 and sub & (sub - 1) == 0
    B = Q.reshape(nb, 8, r)

    def tree(Gs):
        while Gs.shape[0] > 1:
            k = Gs.shape[0] // 2
            nxt = Gs[0:2 * k:2] + Gs[1:2 * k:2]
            Gs = torch.cat([nxt, Gs[2 * k:]]) if Gs.shape[0] % 2 else nxt
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
    key = f"TS-{m}-{l}-{spec}-{round(math.log10(kappa) * 2)}-{draw}"
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
            key = f"TS-{m}-{l}-{spec}-{round(math.log10(kappa) * 2)}-{draw}"
            if key in done:
                continue
            for rec in run_input(m, l, spec, kappa, draw):
                f.write(json.dumps(rec) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
