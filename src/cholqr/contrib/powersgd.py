"""Faster orthogonalization for PyTorch's PowerSGD DDP communication hook.

PowerSGD (``torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook``) orthogonalizes its low-rank factors at every
step with ``_orthogonalize``. In current PyTorch:

- fp16/bf16 inputs (and rank <= 2) use a per-column Python Gram-Schmidt loop;
- fp32 inputs use ``torch.linalg.qr``, which is slow for batches on CUDA.

``patch()`` replaces it with:

- single matrices (the default, ``batch_tensors_with_same_shape=False``): upcast to FP32 and Householder QR;
- batches: CholeskyQR2 computed in FP32. Any matrix whose result is not orthonormal to 1e-3 is recomputed with
  Householder QR, which costs one host sync per call.

A100 measurements (ms per call, bf16): 1x768x32 3.25 -> 0.24, 1x3072x64 6.57 -> 0.72, 12x3072x64 6.74 -> ~1.9.
See ``benchmarks/results/a100_powersgd_2026-11-07.txt``.

Usage::

    from cholqr.contrib import powersgd
    powersgd.patch()          # before training; powersgd.unpatch() restores the original

Notes:
- The result differs from the original only by column signs, and the PowerSGD approximation P Q^T does not depend on
  them.
- ``epsilon`` is accepted for API compatibility. Zero columns are handled by the column scaling, as in ``cholesky_qr``.
- Disable TF32 for the Gram products (``torch.set_float32_matmul_precision("highest")``) if you need full accuracy.
"""
import torch
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as _psh

from ..core import cholesky_qr

__all__ = ["orthogonalize_", "patch", "unpatch", "is_patched"]

_ORIGINAL = _psh._orthogonalize
_TOL = 1e-3


def orthogonalize_(matrices: torch.Tensor, epsilon: float = 0) -> torch.Tensor:
    """In-place replacement for ``powerSGD_hook._orthogonalize`` on a (b, n, r) tensor with r <= n."""
    if not (matrices.dim() == 3 and matrices.shape[2] <= matrices.shape[1]):
        raise ValueError("expected a (b, n, r) tensor with r <= n")
    b, _, r = matrices.shape
    X = matrices.float()
    if b == 1:
        Q = torch.linalg.qr(X, mode="reduced")[0]
    else:
        Q = cholesky_qr(X, method="cqr2")
        eye = torch.eye(r, device=X.device, dtype=X.dtype)
        bad = (Q.mT @ Q - eye).norm(dim=(-2, -1)) > _TOL
        if bool(bad.any()):                                                # rare: ill-conditioned input
            Q[bad] = torch.linalg.qr(X[bad], mode="reduced")[0]
    matrices.copy_(Q)
    return matrices


def patch() -> None:
    """Make PowerSGD use :func:`orthogonalize_`. Affects every PowerSGD hook in this process."""
    _psh._orthogonalize = orthogonalize_


def unpatch() -> None:
    """Restore PyTorch's original ``_orthogonalize``."""
    _psh._orthogonalize = _ORIGINAL


def is_patched() -> bool:
    return _psh._orthogonalize is orthogonalize_
