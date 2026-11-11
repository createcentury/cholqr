"""Which torch.linalg operations and cholqr entry points work on a PyTorch/XLA device (TPU), and which fall back?

Pre-registration: benchmarks/prereg/PREREG_POWERSGD_TRAIN_2026-11-12.md (section "TPU"). For every item: ok / error
(exception type and first line), the shape and dtype of the result where relevant, the orthogonality
||Q^T Q - I||_F (FP64, on the CPU) of returned Q, and the ATen ops that PyTorch/XLA executed on the CPU
("aten::" fallbacks, from torch_xla.debug.metrics) during the item.

Usage:  python benchmarks/xla_probe.py --device xla [--precision highest] --out probe.json
"""
import argparse, json, time, traceback

import torch


def orth(Q):
    Q = Q.detach().cpu().double()
    return float((Q.mT @ Q - torch.eye(Q.shape[-1], dtype=torch.float64)).norm(dim=(-2, -1)).max())


def make(m, l, kappa, seed, batch=None):
    g = torch.Generator().manual_seed(seed)
    shp = (batch, m, l) if batch else (m, l)
    U = torch.linalg.qr(torch.randn(*shp, dtype=torch.float64, generator=g))[0]
    V = torch.linalg.qr(torch.randn(*shp[:-2], l, l, dtype=torch.float64, generator=g))[0]
    s = torch.logspace(0, -torch.log10(torch.tensor(float(kappa))).item(), l, dtype=torch.float64)
    return ((U * s) @ V.mT).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xla"); ap.add_argument("--precision", default="highest")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    met = None
    if a.device.startswith("xla"):
        import torch_xla, torch_xla.backends
        import torch_xla.debug.metrics as met
        torch_xla.backends.set_mat_mul_precision({"highest": "highest", "high": "high", "medium": "default"}[a.precision])
    torch.set_float32_matmul_precision(a.precision)
    import cholqr
    from cholqr.contrib import powersgd
    dev = torch.device(a.device)
    out = dict(torch=torch.__version__, cholqr=cholqr.__version__, device=str(dev), precision=a.precision, items=[])
    if met is not None:
        import torch_xla
        out["torch_xla"] = torch_xla.__version__
        try:
            import torch_xla.core.xla_model as xm; out["hw"] = str(xm.xla_device_hw(dev))
        except Exception as e:
            out["hw_error"] = repr(e)[:200]

    def item(name, fn):
        if met is not None:
            met.clear_counters()
        rec = dict(name=name)
        t0 = time.time()
        try:
            r = fn()
            rec.update(ok=True, **(r or {}))
        except Exception as e:
            rec.update(ok=False, error=f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}",
                       where=traceback.format_exc().splitlines()[-3][:200])
        rec["seconds"] = round(time.time() - t0, 2)
        if met is not None:
            rec["cpu_fallback_ops"] = sorted(met.executed_fallback_ops())
        out["items"].append(rec); print(json.dumps(rec), flush=True)

    Y = make(1024, 64, 1e3, 1).to(dev); Yb = make(1024, 64, 1e3, 2, batch=4).to(dev)
    Yill = make(1024, 64, 1e9, 3).to(dev)

    def chol(A):
        L, info = torch.linalg.cholesky_ex(A)
        return dict(info_shape=list(info.shape), info=info.cpu().reshape(-1).tolist()[:8], L_shape=list(L.shape),
                    L_finite=bool(torch.isfinite(L).all().cpu()))
    item("cholesky_ex 2-D (64x64, SPD)", lambda: chol(Y.mT @ Y))
    item("cholesky_ex batched (4x64x64, SPD)", lambda: chol(Yb.mT @ Yb))
    item("cholesky_ex 2-D non-PD (G - 2I)", lambda: chol(Y.mT @ Y - 2 * torch.eye(64, device=dev)))
    item("cholesky_ex 2-D, float64", lambda: chol((Y.mT @ Y).double()))
    item("solve_triangular", lambda: dict(shape=list(torch.linalg.solve_triangular(
        torch.linalg.cholesky(Y.mT @ Y), torch.eye(64, device=dev), upper=False).shape)))
    item("linalg.qr (Householder) 2-D", lambda: dict(orth=orth(torch.linalg.qr(Y)[0])))
    item("linalg.qr batched", lambda: dict(orth=orth(torch.linalg.qr(Yb)[0])))
    item("float64 matmul", lambda: dict(dtype=str((Y.double().mT @ Y.double()).dtype)))
    item("eigvalsh", lambda: dict(v=float(torch.linalg.eigvalsh(Y.mT @ Y)[0].cpu())))
    for meth in ("scqr3", "cqr2", "rcqr", "auto"):
        item(f"cholesky_qr {meth} on_breakdown=ignore", lambda m=meth: dict(
            orth=orth(cholqr.cholesky_qr(Yb, method=m, on_breakdown="ignore"))))
        item(f"cholesky_qr {meth} default", lambda m=meth: dict(orth=orth(cholqr.cholesky_qr(Yb, method=m))))
    item("cholesky_qr scqr3 robust=True", lambda: dict(orth=orth(cholqr.cholesky_qr(Yb, robust=True))))
    item("cholesky_qr scqr3 verify=True", lambda: dict(orth=orth(cholqr.cholesky_qr(Yb, verify=True))))
    item("cholesky_qr scqr3 default, cond 1e9 (breakdown -> fallback)",
         lambda: dict(orth=orth(cholqr.cholesky_qr(Yill)),
                      orth_raw=orth(cholqr.cholesky_qr(Yill, on_breakdown="ignore"))))
    item("cholesky_qr verify=True, cond 1e9", lambda: dict(orth=orth(cholqr.cholesky_qr(Yill, verify=True))))
    item("orthogonality_bound", lambda: dict(bound=float(cholqr.orthogonality_bound(
        cholqr.cholesky_qr(Y)).reshape(-1)[0].cpu())))

    def psgd():
        X = make(300, 8, 1e2, 4, batch=5).to(dev)
        powersgd.orthogonalize_(X)
        return dict(orth=orth(X))
    item("contrib.powersgd.orthogonalize_ batched", psgd)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
