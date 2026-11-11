"""How does cholqr behave on THIS device and THESE settings? (a few seconds; torch only)

    python -m cholqr            # default device: cuda > mps > cpu
    python -m cholqr cpu --full # also the 4096 x 276 breakdown sweep

Reports the measured Gram-product error (eps_2/u; TF32 shows up as eps_2/u >> 100), the first condition number at
which FP32 scqr3 / cqr2 break down without the check (on_breakdown="ignore"), whether the default (check + fallback)
keeps orthogonality below 1e-4 over that range, whether robust=True (FP64 first pass) is available, and a timing.
References are computed in FP64 on the CPU.
"""
import math
import platform
import sys
import time
import warnings
from typing import Optional

import torch

from .core import _gram, cholesky_qr

_U = 2.0 ** -24                                                           # FP32 unit roundoff


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def _device_name(dev: torch.device) -> str:
    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev)
    if dev.type == "mps":
        return f"Apple GPU (MPS), {platform.machine()}"
    return f"CPU {platform.processor() or platform.machine()}"


def _orth(Q: torch.Tensor) -> float:
    Q = Q.cpu().double()
    return float(torch.linalg.matrix_norm(Q.mT @ Q - torch.eye(Q.shape[-1], dtype=torch.float64), ord=2))


def _gram_eps(dev, m, l, g) -> float:
    """||fl(X^T X) - X^T X||_2 / u for a random unit-column FP32 X, Gram computed on `dev` as cholqr does."""
    X = torch.randn(m, l, dtype=torch.float64, generator=g)
    X = (X / X.norm(dim=0)).float()
    G = _gram(X.to(dev)).cpu().double()
    Xd = X.double()
    return float(torch.linalg.matrix_norm(G - Xd.mT @ Xd, ord=2)) / _U


def _make(m, l, kappa, g) -> torch.Tensor:
    """FP64 matrix with geometric singular values 1 .. 1/kappa and random singular vectors."""
    U = torch.linalg.qr(torch.randn(m, l, dtype=torch.float64, generator=g))[0]
    V = torch.linalg.qr(torch.randn(l, l, dtype=torch.float64, generator=g))[0]
    return (U * torch.logspace(0, -math.log10(kappa), l, dtype=torch.float64)) @ V.mT


def _sweep(dev, m, l, g) -> tuple[dict[str, Optional[float]], float]:
    """First half-decade kappa where FP32 scqr3 / cqr2 lose orthogonality (> 1e-2) without the check, and the worst
    orthogonality of the default (check + fallback) over 1e2 .. 1e10."""
    first: dict[str, Optional[float]] = {"scqr3": None, "cqr2": None}
    worst_default = 0.0
    for e in range(4, 21):
        kappa = 10 ** (e / 2)
        Y = _make(m, l, kappa, g).float().to(dev)
        for method in first:
            if first[method] is None and _orth(cholesky_qr(Y, method=method, on_breakdown="ignore")) > 1e-2:
                first[method] = kappa
            worst_default = max(worst_default, _orth(cholesky_qr(Y, method=method)))
    return first, worst_default


def _time_ms(fn, dev, reps=5) -> float:
    for _ in range(2):
        fn()
    _sync(dev)
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync(dev)
    return (time.perf_counter() - t) / reps * 1e3


def diagnose(device: Optional[str] = None, quick: bool = True, verbose: bool = False) -> dict:
    """Measure cholqr's numerical behaviour on `device` (default: cuda > mps > cpu) under the current
    torch.get_float32_matmul_precision(). quick=True: breakdown sweep on 1024 x 64 only (full adds 4096 x 276).
    Returns a dict; verbose=True also prints a summary."""
    from . import __version__
    t0 = time.perf_counter()
    dev = torch.device(device or _default_device())
    g = torch.Generator().manual_seed(0)
    out: dict[str, object] = dict(torch=torch.__version__, cholqr=__version__, device=str(dev),
                                  device_name=_device_name(dev),
                                  float32_matmul_precision=torch.get_float32_matmul_precision())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                   # the TF32 warning is reported below instead
        eps = {f"{m}x{l}": _gram_eps(dev, m, l, g) for m, l in ((1024, 64), (4096, 276))}
        out["gram_eps_over_u"] = eps
        out["tf32_in_effect"] = max(eps.values()) > 100                   # FP32 accumulation measures ~1-60 u
        sizes = ((1024, 64),) if quick else ((1024, 64), (4096, 276))
        breakdown: dict[str, dict[str, Optional[float]]] = {}
        out["first_breakdown_kappa"], worst = breakdown, 0.0
        for m, l in sizes:
            first, w = _sweep(dev, m, l, g)
            breakdown[f"{m}x{l}"] = first
            worst = max(worst, w)
        out["default_max_orth"] = worst
        out["default_ok"] = worst < 1e-4
        try:
            torch.zeros(1, dtype=torch.float64, device=dev)
            Y = _make(1024, 64, 1e9, g).float().to(dev)
            out["robust_available"] = True
            out["robust_orth_at_1e9"] = _orth(cholesky_qr(Y, robust=True, on_breakdown="ignore"))
        except (RuntimeError, TypeError):
            out["robust_available"], out["robust_orth_at_1e9"] = False, None
        Yb = torch.randn(16, 2048, 128, generator=g).to(dev)
        out["timing_ms"] = dict(shape="16x2048x128",
                                cholesky_qr=_time_ms(lambda: cholesky_qr(Yb), dev),
                                torch_linalg_qr=_time_ms(lambda: torch.linalg.qr(Yb, mode="reduced"), dev))
    out["runtime_s"] = time.perf_counter() - t0
    if verbose:
        _print(out)
    return out


def _fmt(k):
    return "none <= 1e10" if k is None else f"{k:.0e}"


def _print(d: dict) -> None:
    print(f"cholqr {d['cholqr']} / torch {d['torch']} on {d['device']} ({d['device_name']})")
    print(f"float32_matmul_precision = {d['float32_matmul_precision']}")
    eps = ", ".join(f"{k}: {v:.1f}" for k, v in d["gram_eps_over_u"].items())
    print(f"Gram error eps_2/u: {eps}  ->  " + ("TF32 / reduced-precision matmul IN EFFECT (orthogonality ~1e-4 at "
          "best; use torch.set_float32_matmul_precision('highest'))" if d["tf32_in_effect"] else "FP32 accumulation"))
    for size, first in d["first_breakdown_kappa"].items():
        print(f"breakdown without check ({size}): scqr3 from cond {_fmt(first['scqr3'])}, cqr2 from cond {_fmt(first['cqr2'])}")
    print(f"default (check + fallback), cond 1e2..1e10: max ||Q^T Q - I||_2 = {d['default_max_orth']:.1e} "
          f"({'OK' if d['default_ok'] else 'ABOVE 1e-4'})")
    if d["robust_available"]:
        print(f"robust=True available; orthogonality at cond 1e9: {d['robust_orth_at_1e9']:.1e}")
    else:
        print("robust=True not available on this device (no FP64); the fallback uses Householder QR instead")
    t = d["timing_ms"]
    print(f"timing {t['shape']}: cholesky_qr {t['cholesky_qr']:.2f} ms, torch.linalg.qr {t['torch_linalg_qr']:.2f} ms")
    print(f"(diagnose took {d['runtime_s']:.1f} s)")


def _main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = [a for a in argv if not a.startswith("--")]
    diagnose(args[0] if args else None, quick="--full" not in argv, verbose=True)


if __name__ == "__main__":
    _main()
