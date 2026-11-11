# PowerSGD drop-in (`cholqr.contrib.powersgd`)

PyTorch's PowerSGD DDP hook orthogonalizes its low-rank factors at every step. It uses a per-column Python
Gram-Schmidt loop for fp16/bf16 and batched Householder QR for fp32. One call replaces this:

```python
from cholqr.contrib import powersgd
powersgd.patch()      # before training; powersgd.unpatch() restores PyTorch's routine
```

The replacement works as follows:
- **Single matrices:** upcast to FP32, then Householder QR.
- **Batches** (`batch_tensors_with_same_shape=True`): CholeskyQR2 in FP32, with a verified fallback to Householder QR.

A100, ms per call:

| b x n x r | dtype | PyTorch | patched | orthogonality |
|---|---|---|---|---|
| 1 x 768 x 32 | bf16 | 3.38 | 0.24 (**14x**) | 1.7e-2 → 2.8e-3 |
| 1 x 3072 x 64 | bf16 | 6.31 | 0.71 (**8.9x**) | 2.5e-2 → 2.8e-3 |
| 12 x 3072 x 64 | bf16 | 6.55 | 2.42 (**2.7x**) | 2.6e-2 → 2.9e-3 |
| 12 x 3072 x 64 | fp32 | 8.17 | 2.39 (**3.4x**) | 1.3e-6 → 6.5e-6 |
| 1 x 768 x 32 | fp32 | 0.24 | 0.23 (same path) | unchanged |

The result equals PyTorch's up to column signs, which the PowerSGD approximation does not depend on. A test checks
that DDP gradients match with and without the patch.

[Back to README](../README.md)

## Distributed training: keep decisions identical on every rank

In PowerSGD the factor is all-reduced before it is orthogonalized, so every rank normally sees bit-identical input and
makes the same decision. That holds only if the orthogonalization is deterministic and the GPUs are identical. If ranks
can differ (mixed GPU types, non-deterministic kernels), a fallback taken on one rank but not on another gives the ranks
different factors, and the model replicas drift apart without any error. Recommendations, for this hook and for any use
of `cholesky_qr` inside data-parallel training (DDP, FSDP, ZeRO):

- Do not use `on_breakdown="raise"` in multi-process training: a rank that raises leaves the others waiting at the next
  collective (a hang).
- If you check or fall back, make the decision collective: reduce the per-rank "bad" flag with
  `torch.distributed.all_reduce(flag, op=ReduceOp.MAX)` before branching, so that all ranks recompute the same matrices;
  or orthogonalize on one rank and broadcast the result.
- For the periodic check outside a compiled graph (README), reduce the result of `cholqr.orthogonality_bound(Q)` with
  `ReduceOp.MAX` before deciding to recompute.

