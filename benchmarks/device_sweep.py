"""Device sweep: where does CholeskyQR break on THIS device, and do the bounds hold?

Runs an instrumented copy of cholqr's passes (same operations, same order, on the chosen device) over a grid of
condition numbers and spectra, and logs, per pass, every rounding term in FP64 on the CPU:
  eps_gram  = ||fl(X^T X) - X^T X||_2               (Gram product on the device)
  eps_chol  = ||L L^T - A||_2                       (Cholesky backward error; A = fl(fl(G) + sI) as stored)
  eps_tot   = ||L L^T - X^T X - sI||_2              (E of Theorem D)
  eta       = ||Q~ - X L^{-T}||_2                   (forming error)
  lam_hat   = lambda_min(A),  invL2 = ||L^{-1}||_2^2,  sigma_min / sigma_max of the pass input X, info, Lmin.
It also runs cholqr.cholesky_qr with its default settings (breakdown check + fallback) on the same input.

Usage:  python benchmarks/device_sweep.py --device cuda --precision highest --out sweep_cuda_highest.jsonl
        python benchmarks/device_sweep.py --device cuda --precision high    --out sweep_cuda_tf32.jsonl
        python benchmarks/device_sweep.py --sanity --device cuda          (bit-identity of the mirror vs cholqr)
"""
import argparse, json, math, platform, sys, time, warnings, zlib
import torch
import cholqr
from cholqr.core import _gram, _prescale

EPS = torch.finfo(torch.float32).eps
SPECTRA = ("geometric", "powerlaw", "spike", "cluster", "linear", "staircase")


def make(m, l, kappa, spec, g):
    return make64(m, l, kappa, spec, g).float()


def make64(m, l, kappa, spec, g):
    U = torch.linalg.qr(torch.randn(m, l, dtype=torch.float64, generator=g))[0]
    V = torch.linalg.qr(torch.randn(l, l, dtype=torch.float64, generator=g))[0]
    if spec == "geometric":
        s = torch.logspace(0, -math.log10(kappa), l, dtype=torch.float64)
    elif spec == "powerlaw":
        s = torch.arange(1, l + 1, dtype=torch.float64) ** -(math.log(kappa) / math.log(l))
    elif spec == "spike":
        s = torch.full((l,), 1 / kappa, dtype=torch.float64); s[0] = 1
    elif spec == "cluster":
        s = torch.full((l,), 1 / kappa, dtype=torch.float64); s[: l // 2] = 1
    elif spec == "linear":
        s = torch.linspace(1, 1 / kappa, l, dtype=torch.float64)
    else:  # staircase
        s = torch.cat([torch.full((l - 3 * (l // 4),) if b == 3 else (l // 4,), kappa ** (-b / 3), dtype=torch.float64)
                       for b in range(4)])
    return (U * s) @ V.T


def _svd_vals(M):
    try:
        return torch.linalg.svdvals(M)
    except Exception:
        import scipy.linalg
        return torch.from_numpy(scipy.linalg.svd(M.numpy(), compute_uv=False, lapack_driver="gesvd"))


def sv(X):
    if not torch.isfinite(X).all():
        return torch.tensor([float("inf"), float("nan")], dtype=torch.float64)
    return _svd_vals(torch.linalg.qr(X.double(), mode="r")[1])


def spec2(M):
    if not torch.isfinite(M).all():
        return float("inf")
    return float(_svd_vals(M)[0])


def mirror(Y, method, log=True, robust=False):
    """cholqr's _cqr passes for scqr3 / cqr2 (robust: FP64 first pass), on Y's device."""
    l = Y.shape[-1]
    n_pass = {"scqr3": 3, "cqr2": 2}[method]
    Q = Y.double() if robust else Y.float()
    eye = torch.eye(l, dtype=Q.dtype, device=Y.device)
    I64 = torch.eye(l, dtype=torch.float64)
    passes = []
    for i in range(n_pass):
        if robust and i == 1:
            Q = Q.float(); eye = eye.float()
        if i == 0 and Q.dtype == torch.float32:
            Q, _ = _prescale(Q)
        d = Q.norm(dim=-2, keepdim=True).clamp_min(1e-30)
        Q = Q / d
        G = _gram(Q)
        s = ((1e-14 if robust else 1e-6) * l) if (method == "scqr3" and i == 0) else EPS
        A = G + (s + 1e-30) * eye
        L, info = torch.linalg.cholesky_ex(A)
        nanL = bool((~torch.isfinite(L.diagonal())).any())           # backend returned non-finite factor
        info = torch.where(torch.isfinite(L.diagonal()).all(), info, info.clamp_min(1) if int(info) == 0 else info)
        L_inv = torch.linalg.solve_triangular(L, eye.expand_as(L), upper=False)
        Qn = torch.where((info == 0)[..., None, None], Q @ L_inv.mT, Q)
        if log:
            X = Q.cpu().double(); Gc = G.cpu().double(); Lc = L.cpu().double()
            A64 = torch.tril(A.cpu().double()); A64 = A64 + torch.tril(A64, -1).mT
            XtX = X.mT @ X
            ev, V = torch.linalg.eigh(XtX); v = V[:, 0]
            sg = sv(X)
            rec = dict(k=i + 1, s=s, dtype=str(Q.dtype).replace("torch.", ""), info=int(info), nanL=nanL, Lmin=float(L.diagonal().amin()), smin=float(sg[-1]),
                       smax=float(sg[0]), gnorm=float(ev[-1]), eps_gram=spec2(Gc - XtX),
                       lam_hat=float(torch.linalg.eigvalsh(A64)[0]), vEv=float(v @ (Gc - XtX) @ v))
            if int(info) == 0:
                Li = torch.linalg.solve_triangular(Lc, I64, upper=False)
                Qex = X @ Li.mT
                Qo = Qn.cpu().double()
                rec.update(eps_chol=spec2(Lc @ Lc.mT - A64), eps_tot=spec2(Lc @ Lc.mT - XtX - s * I64),
                           invL2=spec2(Li) ** 2, orth_exact=spec2(Qex.mT @ Qex - I64), eta=spec2(Qo - Qex),
                           orth_out=spec2(Qo.mT @ Qo - I64))
            passes.append(rec)
        Q = Qn
    return Q, passes


def exact_pass1(Y, s1):
    """FP64 exact pass 1 on the FP32 input (after the same normalisation), normalised: sigma_min^2 (a-priori rule)."""
    Yn = Y.double(); Yn = Yn / Yn.norm(dim=0, keepdim=True).clamp_min(1e-300)
    L, info = torch.linalg.cholesky_ex(Yn.mT @ Yn + s1 * torch.eye(Y.shape[1], dtype=torch.float64))
    if int(info) != 0:          # exactly rank-deficient input (e.g. FP8): not defined
        return None
    Q1 = torch.linalg.solve_triangular(L, Yn.mT, upper=False).mT
    Q1 = Q1 / Q1.norm(dim=0, keepdim=True)
    return float(sv(Q1)[-1]) ** 2


def device_info(dev):
    info = dict(torch=torch.__version__, cholqr=cholqr.__version__, python=platform.python_version(),
                device=str(dev), precision=torch.get_float32_matmul_precision(), cpu=platform.processor() or platform.machine())
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        info.update(gpu=p.name, capability=f"{p.major}.{p.minor}", cuda=torch.version.cuda,
                    hip=getattr(torch.version, "hip", None), allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", default="highest", choices=["highest", "high", "medium"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--prefix", default="DEV-")
    ap.add_argument("--quick", action="store_true", help="small grid (smoke test)")
    ap.add_argument("--input-dtype", default="float32", choices=["float64", "float32", "bfloat16", "float16", "float8_e4m3fn"])
    ap.add_argument("--robust", action="store_true", help="FP64 first pass (scqr3 only)")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--compile", action="store_true", help="library only: torch.compile(cholesky_qr, fullgraph=True), on_breakdown ignored")
    a = ap.parse_args()
    torch.set_float32_matmul_precision(a.precision)
    warnings.filterwarnings("ignore", message="cholqr: TF32")
    dev = torch.device(a.device)
    if a.sanity:
        g = torch.Generator().manual_seed(8999); bad = 0
        for t in range(20):
            m, l = [(1024, 64), (1024, 276), (4096, 276), (4096, 64)][t % 4]
            Y = make(m, l, 10 ** (2 + t % 9), SPECTRA[t % 6], g).to(dev)
            for method in ("scqr3", "cqr2"):
                Qm = mirror(Y, method, log=False)[0]
                Ql = cholqr.cholesky_qr(Y, method=method, on_breakdown="ignore")
                bad += int(not torch.equal(Qm.cpu(), Ql.cpu()))
        print(json.dumps(dict(sanity_mismatches=bad, **device_info(dev)))); return
    ms, ls = ((1024,), (64,)) if a.quick else ((1024, 4096), (64, 276))
    kap = [10 ** (x / 2) for x in (range(4, 21, 4) if a.quick else range(4, 21))]
    t0 = time.time(); n = 0
    in_dt = getattr(torch, a.input_dtype)
    methods = ("scqr3",) if a.robust else ("scqr3", "cqr2")
    U_IN = {"float64": 2.0 ** -53, "float32": 2.0 ** -24, "bfloat16": 2.0 ** -9, "float16": 2.0 ** -11, "float8_e4m3fn": 2.0 ** -4}[a.input_dtype]
    fn = torch.compile(cholqr.cholesky_qr, fullgraph=True) if a.compile else None
    with open(a.out, "a") as f:
        f.write(json.dumps(dict(header=True, input_dtype=a.input_dtype, robust=a.robust, compile=a.compile,
                                **device_info(dev))) + "\n")
        for m in ms:
            for l in ls:
                for spec in SPECTRA:
                    for kappa in kap:
                        for draw in range(1, a.draws + 1):
                            key = f"{a.prefix}{m}-{l}-{spec}-{round(math.log10(kappa) * 2)}-{draw}"
                            g = torch.Generator().manual_seed(7000 + zlib.crc32(key.encode()) % 100000)
                            Y64 = make64(m, l, kappa, spec, g)
                            if a.input_dtype != "float64":     # scale to max |entry| = 1 (as for scaled low-precision data)
                                Y64 = Y64 / Y64.abs().max()
                            Yin = Y64.to(in_dt)                 # the input as the caller holds it
                            Yi64 = Yin.double()
                            ky = sv(Yi64)
                            gd = torch.Generator().manual_seed(1 + zlib.crc32(key.encode()) % 100000)
                            Dm = (torch.rand(Y64.shape, dtype=torch.float64, generator=gd) * 2 - 1) * U_IN
                            kc = sv(Y64 + Y64 * Dm)
                            pre = exact_pass1(Yin.float(), 1e-6 * l) if not a.robust else None
                            Yd = Yin.to(dev)
                            I64 = torch.eye(l, dtype=torch.float64)
                            for method in methods:
                                rec = dict(m=m, l=l, spec=spec, kappa=kappa, draw=draw, method=method, precision=a.precision,
                                           input_dtype=a.input_dtype, robust=a.robust, compile=a.compile,
                                           kappa_eff=float(ky[0] / ky[-1]), kappa_cw=float(kc[0] / kc[-1]), smin2_q1n_exact=pre)
                                if a.compile:
                                    Qc = fn(Yd, method=method, robust=a.robust, on_breakdown="ignore").cpu().double()
                                    Qe = cholqr.cholesky_qr(Yd, method=method, robust=a.robust, on_breakdown="ignore").cpu().double()
                                    rec.update(orth_compiled=spec2(Qc.mT @ Qc - I64), orth_eager=spec2(Qe.mT @ Qe - I64))
                                else:
                                    Q, passes = mirror(Yd, method, robust=a.robust)
                                    Qd = Q.cpu().double()
                                    Ql = cholqr.cholesky_qr(Yd, method=method, robust=a.robust, out_dtype=torch.float32).cpu().double()
                                    rec.update(passes=passes, orth=spec2(Qd.mT @ Qd - I64), orth_lib_default=spec2(Ql.mT @ Ql - I64))
                                f.write(json.dumps(rec) + "\n")
                                n += 1
                            f.flush()
    print(f"done {n} runs in {time.time() - t0:.0f}s -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
