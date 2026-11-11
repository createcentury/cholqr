"""What does a loss of orthogonality ||Q^T Q - I|| do downstream?  (CPU only; pre-registered, see
benchmarks/prereg/PREREG_DOWNSTREAM_2026-11-12.md)

Parts
  resid    residual ||Y - QR||_F/||Y||_F and principal-angle distance range(Q) vs range(Y) for cholqr methods
           (geometric, two-cluster, staircase spectra; cond 1e2..1e8 in half decades; 1024x64 and 4096x276).
  rsvd     randomized range finder / SVD (q = 0, 1, 2 power steps) with Q exact, from cholqr (FP32, emulated TF32),
           or exact Q with an injected orthogonality error ||F||_F = delta in {1e-4, 1e-3, 1e-2}.
  subiter  subspace iteration (10 steps), orthogonality error per step; injected errors at every step.
  powersgd PowerSGD-style rank-r compression with error feedback, 20 steps.

All analysis quantities are computed in FP64 on the CPU. "tf32e" = TF32 emulated on the CPU (inputs of every cholqr
matrix product rounded to TF32, products accumulated in FP64 and rounded to FP32), identical to
benchmarks/verify_eval.py (EmulatedTF32). Injected perturbations:
  rp    (range-preserving)  Q = U (I + F)^{1/2}, F symmetric, so Q^T Q - I = F exactly and range(Q) = range(U);
  align as rp, F = delta w w^T with w the top left singular vector of U^T A (worst case for projections);
  gen   (range-changing)    Q = U (I + F)^{1/2} + (I - U U^T) Z, ||Z||_2 = delta / 2.

Usage:  python benchmarks/downstream_orth.py --part all --out benchmarks/results/downstream_2026-11
        python benchmarks/downstream_orth.py --smoke            (tiny run, prints timings only)
        python benchmarks/downstream_orth.py --judge benchmarks/results/downstream_2026-11
"""
import argparse, json, math, os, statistics, sys, time, zlib
import torch
import cholqr
from cholqr import core

torch.set_float32_matmul_precision("highest")
D64 = torch.float64
DELTAS = (1e-4, 1e-3, 1e-2)


# ---------------------------------------------------------------- TF32 emulation (as benchmarks/verify_eval.py)
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


# ---------------------------------------------------------------- QR variants (input FP32 Y32)
def qr_variant(name, Y32, seed=0):
    """Returns (Q, R) as FP64 tensors (R may be None)."""
    g = torch.Generator().manual_seed(seed)
    if name == "householder32":
        Q, R = torch.linalg.qr(Y32)
        return Q.double(), R.double()
    tf32 = name.startswith("tf32e_")
    base = name[6:] if tf32 else name
    raw = base.endswith("_raw")
    base = base[:-4] if raw else base
    kw = dict(return_r=True, on_breakdown="ignore" if raw else "fallback")
    if base == "robust":
        kw.update(robust=True)
    else:
        kw.update(method=base)
    if base == "rcqr":
        kw.update(generator=g)
    if tf32:
        with EmulatedTF32():
            Q, R = cholqr.cholesky_qr(Y32, **kw)
    else:
        Q, R = cholqr.cholesky_qr(Y32, **kw)
    return Q.double(), R.double()


def exact_q(Y64):
    return torch.linalg.qr(Y64)[0]


# ---------------------------------------------------------------- metrics (FP64)
def finite(X):
    return bool(torch.isfinite(X).all())


def orth_f(Q):
    if not finite(Q):
        return float("inf")
    I = torch.eye(Q.shape[-1], dtype=D64)
    return torch.linalg.matrix_norm(Q.mT @ Q - I).item()


def sin_theta(Q, U):
    """sin of the largest principal angle between range(Q) and range(U) (equal dimensions)."""
    if not finite(Q):
        return float("nan")
    Qo = torch.linalg.qr(Q)[0]
    return torch.linalg.svdvals(Qo - U @ (U.mT @ Qo))[0].item()


def capture(Q, Uk):
    """||(I - P_range(Q)) Uk||_2: how well range(Q) contains the k-dim subspace Uk."""
    if not finite(Q):
        return float("nan")
    Qo = torch.linalg.qr(Q)[0]
    return torch.linalg.svdvals(Uk - Qo @ (Qo.mT @ Uk))[0].item()


def sqrt_psd(I_plus_F):
    w, V = torch.linalg.eigh(I_plus_F)
    return (V * w.sqrt()) @ V.mT


def sym_rand(k, delta, g):
    G = torch.randn(k, k, dtype=D64, generator=g)
    G = G + G.mT
    return delta * G / torch.linalg.matrix_norm(G)


def inject(U, kind, delta, g, B=None):
    """U orthonormal (m x k). Returns (Q, F) with Q^T Q = I + F (+ O(delta^2) for 'gen')."""
    k = U.shape[1]
    if kind == "align":
        w = torch.linalg.svd(B, full_matrices=False)[0][:, :1]       # top left singular vector of B = U^T A
        F = delta * (w @ w.mT)
    else:
        F = sym_rand(k, delta, g)
    Q = U @ sqrt_psd(torch.eye(k, dtype=D64) + F)
    if kind == "gen":
        Z = torch.randn(U.shape[0], k, dtype=D64, generator=g)
        Z = Z - U @ (U.mT @ Z)
        Q = Q + (delta / 2) * Z / torch.linalg.svdvals(Z)[0]
    return Q, F


def seed_of(*parts):
    return zlib.crc32("/".join(map(str, parts)).encode())


# ---------------------------------------------------------------- part resid
RESID_METHODS = ("scqr3", "scqr3_raw", "cqr2", "cqr2_raw", "robust", "rcqr",
                 "tf32e_scqr3", "tf32e_scqr3_raw", "tf32e_cqr2_raw", "householder32")


def make64(m, l, kappa, spec, g):
    """Same construction as benchmarks/device_sweep.py (make64); also returns the exact left basis U."""
    U = torch.linalg.qr(torch.randn(m, l, dtype=D64, generator=g))[0]
    V = torch.linalg.qr(torch.randn(l, l, dtype=D64, generator=g))[0]
    if spec == "geometric":
        s = torch.logspace(0, -math.log10(kappa), l, dtype=D64)
    elif spec == "cluster":
        s = torch.full((l,), 1 / kappa, dtype=D64); s[: l // 2] = 1
    else:  # staircase
        s = torch.cat([torch.full((l - 3 * (l // 4),) if b == 3 else (l // 4,), kappa ** (-b / 3), dtype=D64)
                       for b in range(4)])
    return (U * s) @ V.T, U


def part_resid(out, smoke=False):
    shapes = [(1024, 64)] if smoke else [(1024, 64), (4096, 276)]
    kappas = [1e3] if smoke else [10 ** (e / 2) for e in range(4, 17)]
    specs = ("geometric",) if smoke else ("geometric", "cluster", "staircase")
    seeds = (0,) if smoke else (0, 1)
    for (m, l) in shapes:
        for spec in specs:
            for kappa in kappas:
                for seed in seeds:
                    g = torch.Generator().manual_seed(seed_of("resid", m, l, spec, kappa, seed))
                    Y64, U = make64(m, l, kappa, spec, g)
                    Y32 = Y64.float()
                    Yd = Y32.double()
                    Ufl = exact_q(Yd)
                    smin = torch.linalg.svdvals(Yd)[-1].item()
                    nY = torch.linalg.matrix_norm(Yd).item()
                    for meth in RESID_METHODS:
                        t = time.perf_counter()
                        Q, R = qr_variant(meth, Y32, seed)
                        dt = time.perf_counter() - t
                        if finite(Q) and finite(R):
                            D = Yd - Q @ R
                            resid = torch.linalg.matrix_norm(D).item() / nY
                            bound = torch.linalg.svdvals(D)[0].item() / smin
                        else:
                            resid = bound = float("nan")
                        rec = dict(part="resid", m=m, l=l, spec=spec, kappa=kappa, seed=seed, method=meth,
                                   orth=orth_f(Q), resid=resid, sin_fl=sin_theta(Q, Ufl), sin_ex=sin_theta(Q, U),
                                   bound=bound, sigma_min_fl=smin, qr_s=dt)
                        out.write(json.dumps(rec) + "\n")
                    out.flush()


# ---------------------------------------------------------------- matrices with known spectra (rsvd, subiter)
RSVD_SPECS = ("poly1", "poly2", "geo12", "geo6")


def make_A(spec, seed, m=2048, n=1024):
    g = torch.Generator().manual_seed(seed_of("A", spec, seed, m, n))
    U = torch.linalg.qr(torch.randn(m, n, dtype=D64, generator=g))[0]
    V = torch.linalg.qr(torch.randn(n, n, dtype=D64, generator=g))[0]
    j = torch.arange(1, n + 1, dtype=D64)
    s = {"poly1": j ** -1, "poly2": j ** -2,
         "geo12": 10 ** (-(j - 1) / 12), "geo6": 10 ** (-(j - 1) / 6)}[spec].clamp_min(1e-30)
    return (U * s) @ V.T, U, s, g


REAL_VARIANTS = ("scqr3", "cqr2_raw", "tf32e_scqr3", "tf32e_scqr3_raw", "tf32e_cqr2_raw", "householder32")


def part_rsvd(out, smoke=False, l=64, k=48):
    specs = ("poly2",) if smoke else RSVD_SPECS
    qs = (1,) if smoke else (0, 1, 2)
    seeds = (0,) if smoke else (0, 1)
    for spec in specs:
        for seed in seeds:
            A, UA, s, g = make_A(spec, seed)
            nA = torch.linalg.matrix_norm(A).item()
            Om = torch.randn(A.shape[1], l, dtype=D64, generator=g)
            for q in qs:
                Y = A @ Om
                for _ in range(q):
                    Y = A @ (A.mT @ Y)
                Y = Y / torch.linalg.matrix_norm(Y)
                U = exact_q(Y)
                B = U.mT @ A
                sv_U = torch.linalg.svdvals(B)[:k]
                lr_exact = torch.linalg.matrix_norm(A - U @ B).item() / nA

                def record(variant, Q, kind=None, delta=None, F=None):
                    t = time.perf_counter()
                    rec = dict(part="rsvd", spec=spec, seed=seed, q=q, variant=variant, kind=kind, delta=delta,
                               orth=orth_f(Q), sin=sin_theta(Q, U), lr_exact=lr_exact)
                    if finite(Q):
                        BQ = Q.mT @ A
                        sv = torch.linalg.svdvals(BQ)[:k]
                        rec["sv_rel_exact"] = ((sv - sv_U).abs() / sv_U).max().item()
                        rec["sv_rel_true"] = ((sv - s[:k]).abs() / s[:k]).max().item()
                        rec["sv_rel_true_exactQ"] = ((sv_U - s[:k]).abs() / s[:k]).max().item()
                        lr = torch.linalg.matrix_norm(A - Q @ BQ).item() / nA
                        rec["lr"] = lr
                        rec["excess"] = lr / lr_exact - 1
                    if F is not None:
                        ev = torch.linalg.eigvalsh(F)
                        rec["sv_bound"] = max(math.sqrt(1 + ev[-1].item()) - 1, 1 - math.sqrt(1 + ev[0].item()))
                        rec["pyth"] = math.sqrt(lr_exact ** 2 + (torch.linalg.matrix_norm(F @ B).item() / nA) ** 2)
                        rec["F2"] = ev.abs().max().item()
                    rec["metric_s"] = time.perf_counter() - t
                    out.write(json.dumps(rec) + "\n")

                record("exact", U)
                for v in REAL_VARIANTS:
                    Q, _ = qr_variant(v, Y.float(), seed)
                    record(v, Q)
                for kind in ("rand", "align", "gen"):
                    for d in DELTAS:
                        Q, F = inject(U, kind, d, g, B=B)
                        record(f"inj_{kind}", Q, kind, d, None if kind == "gen" else F)
                out.flush()


# ---------------------------------------------------------------- part subiter
SUB_SPECS = ("poly2", "geo12")


def part_subiter(out, smoke=False, l=64, k=48, J=10):
    specs = ("geo12",) if smoke else SUB_SPECS
    seeds = (0,) if smoke else (0, 1)
    Jr = 2 if smoke else J
    for spec in specs:
        for seed in seeds:
            A, UA, s, g = make_A(spec, seed)
            Uk = UA[:, :k]
            Om = torch.randn(A.shape[1], l, dtype=D64, generator=g)
            variants = ["exact", "scqr3", "tf32e_scqr3_raw", "tf32e_cqr2_raw"] + \
                       [f"inj_{kd}_{d:g}" for kd in ("rand", "gen") for d in DELTAS]
            traj = {}
            for v in variants:
                gi = torch.Generator().manual_seed(seed_of("subiter-inj", spec, seed, v))
                Z = A @ Om
                rows = []
                for j in range(Jr + 1):
                    Z = Z / torch.linalg.matrix_norm(Z)
                    if v == "exact" or v.startswith("inj_"):
                        Q = exact_q(Z)
                        if v.startswith("inj_"):
                            _, kd, d = v.split("_")
                            Q, _ = inject(Q, kd, float(d), gi)
                    else:
                        Q, _ = qr_variant(v, Z.float(), seed)
                    rows.append((orth_f(Q), capture(Q, Uk), Q))
                    Z = A @ (A.mT @ Q)
                traj[v] = rows
            for v in variants:
                rec = dict(part="subiter", spec=spec, seed=seed, variant=v,
                           orth=[r[0] for r in traj[v]], cap=[r[1] for r in traj[v]],
                           dev=[sin_theta(traj[v][j][2], traj["exact"][j][2]) for j in range(Jr + 1)])
                out.write(json.dumps(rec) + "\n")
            out.flush()


# ---------------------------------------------------------------- part powersgd
def part_powersgd(out, smoke=False, n=1024, m=512, T=20):
    ranks = (4,) if smoke else (4, 16)
    seeds = (0,) if smoke else (0, 1)
    Tr = 3 if smoke else T
    for r in ranks:
        for seed in seeds:
            g = torch.Generator().manual_seed(seed_of("psgd", r, seed))
            UL = torch.linalg.qr(torch.randn(n, 64, dtype=D64, generator=g))[0]
            VL = torch.linalg.qr(torch.randn(m, 64, dtype=D64, generator=g))[0]
            L = (UL * torch.arange(1, 65, dtype=D64) ** -1.0) @ VL.mT
            L = L / torch.linalg.matrix_norm(L)
            Gs = []
            for _ in range(Tr):
                N = torch.randn(n, m, dtype=D64, generator=g)
                Gs.append(L + 0.5 * N / torch.linalg.matrix_norm(N))
            Q0 = torch.randn(m, r, dtype=D64, generator=g)
            sumG = sum(Gs)
            variants = ["exact", "cqr2", "scqr3", "tf32e_cqr2_raw"] + \
                       [f"inj_{kd}_{d:g}" for kd in ("rand", "gen") for d in DELTAS]
            for v in variants:
                gi = torch.Generator().manual_seed(seed_of("psgd-inj", r, seed, v))
                e = torch.zeros(n, m, dtype=D64)
                Q = Q0.clone()
                errs, orths, sumA, oneshot = [], [], torch.zeros(n, m, dtype=D64), []
                for t in range(Tr):
                    M = Gs[t] + e
                    P = M @ Q
                    if v == "exact" or v.startswith("inj_"):
                        Pe = exact_q(P)
                        Po = Pe
                        if v.startswith("inj_"):
                            _, kd, d = v.split("_")
                            Po, _ = inject(Pe, kd, float(d), gi)
                    else:
                        Po, _ = qr_variant(v, P.float(), seed)
                    Q = M.mT @ Po
                    approx = Po @ Q.mT
                    nM = torch.linalg.matrix_norm(M).item()
                    c = torch.linalg.matrix_norm(M - approx).item() / nM
                    errs.append(c); orths.append(orth_f(Po))
                    if v == "exact":                                  # one-shot excess at the exact trajectory
                        UM = Pe.mT @ M
                        for d in DELTAS:
                            Pd, F = inject(Pe, "rand", d, gi)
                            cd = torch.linalg.matrix_norm(M - Pd @ (Pd.mT @ M)).item() / nM
                            uf = torch.linalg.matrix_norm(UM).item() / nM
                            oneshot.append(dict(t=t, delta=d, excess=cd - c,
                                                bound=math.sqrt(c * c + (d * uf) ** 2) - c))
                    e = M - approx
                    sumA += approx
                rec = dict(part="powersgd", r=r, seed=seed, variant=v, err=errs, orth=orths,
                           cum_err=torch.linalg.matrix_norm(sumG - sumA).item() / torch.linalg.matrix_norm(sumG).item())
                if oneshot:
                    rec["oneshot"] = oneshot
                out.write(json.dumps(rec) + "\n")
            out.flush()


PARTS = dict(resid=part_resid, rsvd=part_rsvd, subiter=part_subiter, powersgd=part_powersgd)


# ---------------------------------------------------------------- judge
def load(d):
    recs = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".jsonl"):
            with open(os.path.join(d, f)) as fh:
                recs += [json.loads(x) for x in fh if x.strip()]
    return recs


def judge(d):
    R = load(d)
    lines, verdict = [], {}

    def say(key, ok, detail):
        verdict[key] = "PASS" if ok is True else ("FAIL" if ok is False else ok)
        lines.append(f"{key}: {verdict[key]} -- {detail}")

    nan = float("nan")
    fin = lambda x: x is not None and math.isfinite(x)
    broken = lambda r: not fin(r["orth"]) or r["orth"] > 0.1

    # ---- resid
    rs = [r for r in R if r["part"] == "resid"]
    H = ("scqr3", "scqr3_raw", "cqr2", "cqr2_raw", "robust", "rcqr")
    h = [r for r in rs if r["method"] in H and not broken(r)]
    if h:
        res = [r["resid"] for r in h]
        say("R1", max(res) <= 1e-4 and statistics.median(res) <= 1e-5,
            f"FP32 cholqr non-broken n={len(h)}: max resid {max(res):.2e}, median {statistics.median(res):.2e}")
        bad = []
        for meth in H:
            lo = [r["resid"] for r in h if r["method"] == meth and r["kappa"] <= 1e3 + 1]
            hi = [r["resid"] for r in h if r["method"] == meth and r["kappa"] >= 1e6 - 1]
            if lo and hi and statistics.median(hi) > 10 * statistics.median(lo):
                bad.append(f"{meth} {statistics.median(lo):.1e}->{statistics.median(hi):.1e}")
        say("R2", not bad, "median resid cond>=1e6 vs cond<=1e3 within 10x" + (f"; violations: {bad}" if bad else ""))
        t = [r for r in rs if r["method"].startswith("tf32e") and not broken(r)]
        if t:
            tr = [r["resid"] for r in t]
            say("R3", all(1e-5 <= x <= 1e-2 for x in tr), f"tf32e non-broken n={len(t)}: resid {min(tr):.2e}..{max(tr):.2e}")
        ok = sum(r["sin_fl"] <= 1e-7 * r["kappa"] for r in h)
        say("R4a", ok >= 0.98 * len(h), f"sin_fl <= 1e-7 kappa in {ok}/{len(h)} (need >= 98%)")
        chk = [r for r in rs if fin(r["bound"]) and r["bound"] < 1 and fin(r["sin_fl"])]
        viol = [r for r in chk if r["sin_fl"] > r["bound"] * (1 + 1e-6) + 1e-15 * r["kappa"] + 1e-13]
        say("R4b", not viol, f"sin_fl <= ||Y-QR||_2/sigma_min in {len(chk) - len(viol)}/{len(chk)}")
        g = [r for r in rs if fin(r["orth"]) and 1e-3 <= r["orth"] <= 0.1 and fin(r["sin_fl"])]
        if g:
            ok = sum(r["sin_fl"] <= r["orth"] for r in g)
            say("R5", ok >= 0.9 * len(g), f"orth in [1e-3, 0.1]: sin_fl <= orth in {ok}/{len(g)} (need >= 90%)")
        else:
            say("R5", "NOT TESTED", "no run with orth in [1e-3, 0.1]")

    # ---- rsvd
    rv = [r for r in R if r["part"] == "rsvd"]
    if rv:
        inj = [r for r in rv if r["variant"] in ("inj_rand", "inj_align")]
        v1 = [r for r in inj if r["sv_rel_exact"] > r["sv_bound"] + 1e-12]
        al = [r for r in inj if r["kind"] == "align"]
        v1b = [r for r in al if r["sv_rel_exact"] < 0.9 * r["sv_bound"]]
        say("D1", not v1 and not v1b, f"sv rel err <= bound in {len(inj) - len(v1)}/{len(inj)}; "
            f"align attains >= 0.9 bound in {len(al) - len(v1b)}/{len(al)}")
        v2 = [r for r in inj if abs(r["lr"] - r["pyth"]) > 1e-6 * r["lr"] + 1e-14]
        say("D2", not v2, f"Pythagoras identity within 1e-6 in {len(inj) - len(v2)}/{len(inj)}")
        a4 = [r for r in al if r["delta"] == 1e-4]
        c1 = all(r["excess"] <= 0.01 for r in a4 if r["lr_exact"] >= 1e-3)
        c2 = all(r["sv_rel_exact"] <= 5e-5 + 1e-12 for r in a4)
        say("V1", c1 and c2, f"align 1e-4: max excess (lr_exact>=1e-3) "
            f"{max([r['excess'] for r in a4 if r['lr_exact'] >= 1e-3] or [nan]):.2e}; max sv err "
            f"{max([r['sv_rel_exact'] for r in a4] or [nan]):.2e}")
        a2 = [r for r in al if r["delta"] == 1e-2]
        c1 = any(r["excess"] >= 0.1 for r in a2 if r["lr_exact"] <= 1e-3)
        c2 = all(r["excess"] <= 0.01 for r in a2 if r["lr_exact"] >= 0.1)
        say("V2", c1 and c2, "align 1e-2: excess >= 10% for some lr_exact <= 1e-3; <= 1% for all lr_exact >= 0.1")
        real = [r for r in rv if r["variant"] in REAL_VARIANTS and not broken(r) and "sv_rel_exact" in r]
        if real:
            ok = sum(r["sv_rel_exact"] <= max(r["orth"], 1e-6) for r in real)
            say("D3", ok >= 0.9 * len(real), f"realistic non-broken: sv err <= max(orth, 1e-6) in {ok}/{len(real)}")

    # ---- subiter
    sv_ = [r for r in R if r["part"] == "subiter"]
    if sv_:
        ex = {(r["spec"], r["seed"]): r for r in sv_ if r["variant"] == "exact"}
        bad1, bad2 = [], []
        for r in sv_:
            o = [x for x in r["orth"][1:]]
            if all(fin(x) for x in o) and o[-1] > 3 * statistics.median(o) + 1e-12:
                bad1.append(f"{r['spec']}/{r['seed']}/{r['variant']}")
            v = r["variant"]
            if v == "exact":
                continue
            lim = float(v.split("_")[2]) if v.startswith("inj_") else (1e-4 if v == "scqr3" else 1e-2)
            ce = ex[(r["spec"], r["seed"])]["cap"][-1]
            if not (fin(r["cap"][-1]) and r["cap"][-1] <= max(2 * ce, lim)):
                bad2.append(f"{r['spec']}/{r['seed']}/{v}: {r['cap'][-1]:.2e} vs exact {ce:.2e}")
        say("S1", not bad1, f"orth(Q_J) <= 3 median_j orth(Q_j); violations: {bad1}")
        say("S2", not bad2, f"capture error at J <= max(2 exact, limit); violations: {bad2}")

    # ---- powersgd
    ps = [r for r in R if r["part"] == "powersgd"]
    if ps:
        ex = {(r["r"], r["seed"]): r for r in ps if r["variant"] == "exact"}
        os_ = [o for r in ex.values() for o in r["oneshot"]]
        bad = [o for o in os_ if o["excess"] > o["bound"] * (1 + 1e-9) + 1e-15]
        say("W1", not bad, f"one-shot excess <= sqrt(c^2+delta^2 u^2)-c in {len(os_) - len(bad)}/{len(os_)}; "
            f"max excess at 1e-2: {max(o['excess'] for o in os_ if o['delta'] == 1e-2):.2e}")
        b2, b3, b4 = [], [], []
        for r in ps:
            v = r["variant"]
            if v == "exact":
                continue
            e0 = ex[(r["r"], r["seed"])]
            dmax = max(abs(a - b) for a, b in zip(r["err"], e0["err"]))
            cum = abs(r["cum_err"] / e0["cum_err"] - 1)
            tag = f"r{r['r']}/s{r['seed']}/{v}: max|derr| {dmax:.1e}, cum {cum:.1e}"
            if v.startswith("inj_rand"):
                d = float(v.split("_")[2])
                if dmax > d or (d == 1e-2 and dmax > 1e-3) or cum > 0.01:
                    b2.append(tag)
            elif v.startswith("inj_gen"):
                if dmax > float(v.split("_")[2]):
                    b4.append(tag)
            elif dmax > (1e-3 if v.startswith("tf32e") else 1e-5):
                b3.append(tag)
        say("W2", not b2, f"range-preserving injected; violations: {b2}")
        say("W3", not b3, f"cholqr FP32 <= 1e-5, tf32e <= 1e-3; violations: {b3}")
        say("W4", not b4, f"range-changing injected <= delta; violations: {b4}")

    # ---- registered overall decision
    if all(k in verdict for k in ("V1", "D1", "W2")):
        say("TOL", verdict["V1"] == "PASS" and verdict["D1"] == "PASS" and verdict["W2"] == "PASS",
            "1e-4 is a sensible default tolerance for the target uses (V1 and D1 and W2)")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all"] + list(PARTS))
    ap.add_argument("--out", default="benchmarks/results/downstream_2026-11")
    ap.add_argument("--smoke", action="store_true", help="tiny run; prints timings only, writes to /tmp")
    ap.add_argument("--judge", metavar="DIR")
    a = ap.parse_args()
    if a.judge:
        txt = judge(a.judge)
        print(txt)
        with open(os.path.join(a.judge, "judge.txt"), "w") as fh:
            fh.write(f"cholqr {cholqr.__version__}, torch {torch.__version__}\n" + txt + "\n")
        sys.exit(0)
    out_dir = "/tmp/downstream_smoke" if a.smoke else a.out
    os.makedirs(out_dir, exist_ok=True)
    for name, fn in PARTS.items():
        if a.part not in ("all", name):
            continue
        t = time.perf_counter()
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as fh:
            fn(fh, smoke=a.smoke)
        print(f"{name}: {time.perf_counter() - t:.1f} s", flush=True)
