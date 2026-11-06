# cholqr

Fast, compile-friendly **shifted CholeskyQR3** for batched tall-skinny matrices in PyTorch.
It is a drop-in for `torch.linalg.qr(Y, mode="reduced")[0]` when you only need an orthonormal basis of `range(Y)`,
for example in randomized SVD and sketching, low-rank optimizers such as randomized Muon, or subspace iteration.

```python
import torch
from cholqr import cholesky_qr

Y = torch.randn(40, 768, 210, device="cuda")       # batch of tall-skinny matrices
Q = cholesky_qr(Y)                                 # orthonormal columns, same span as Y
fast = torch.compile(cholesky_qr, fullgraph=True)  # single graph, no host synchronization
```

## Why

On GPUs, batched Householder QR (cuSOLVER `geqr2`) processes the batch one matrix at a time.

On an NVIDIA A100, for the sketches of randomized low-rank Muon on nanoGPT (two weight banks, sketch width 210):

| | Householder QR | `cholqr` | full Newton–Schulz (reference) |
|---|---|---|---|
| optimizer time per step, both banks, one GPU | 117.6 ms | **15.2 ms** | 17.1 ms |

- `torch.linalg.qr` reaches about **1.6% of FP32 peak** here.
- On the QR step alone, `cholqr` is 3–14x faster.
- End to end (1490 steps, A100-80GB), replacing the QR removes about 100 ms per step. Randomized Muon then trains as fast as full Muon at unchanged perplexity.

Run `python benchmarks/bench_qr.py --compile` on your GPU to measure your own shapes.

## How

Shifted CholeskyQR3 (Fukaya et al., 2020) with three implementation choices:

1. **Column scaling before every Gram product.** The shift is scale invariant, and unnormalized inputs do not overflow FP32.
2. **Explicit r x r triangular inverse applied as one batched GEMM.** This replaces a slow triangular solve over m rows.
3. **`cholesky_ex` with a per-matrix `torch.where` fallback.** There is no host synchronization, and the routine compiles with zero graph breaks.

## Accuracy and range of validity

| input condition number | default (FP32) | `robust=True` (FP64 first pass) |
|---|---|---|
| up to 1e6 | ‖QᵀQ − I‖_F ≈ 1e-5 | ‖QᵀQ − I‖_F ≈ 1e-5 |
| 5e6 – 1e8 | **breaks down** (orthogonality lost) | ‖QᵀQ − I‖_F ≈ 1e-5, subspace error ≈ 2e-6 |

These numbers were tested on 3072 x 210 inputs with geometric and power-law spectra (`tests/`).

- The default first-pass shift `1e-6 * tr(G)` is a heuristic. The a-priori shift of Fukaya et al. is impractically large in FP32 for large m·r.
- `robust=True` costs one FP64 Gram product. That is cheap on data-center GPUs (A100/H100) and slow on consumer GPUs.
- **Real optimizer states** (27 nanoGPT momentum snapshots, sketch width 210, bf16 power iteration as in training):
  - κ(Y) ≤ 4.5×10⁴.
  - No fallback was triggered.
  - The final low-rank update differs from an FP64 Householder reference by at most 7.6×10⁻⁵ (relative).
  - Sketches computed in bf16 are typically well inside the default range, because bf16 rounding caps their effective condition number.
- Exactly rank-deficient or zero inputs return finite outputs (not an orthonormal basis of a larger space).

```python
Q = cholesky_qr(Y)               # fast default, cond(Y) <~ 1e6
Q = cholesky_qr(Y, robust=True)  # ill-conditioned inputs, cond(Y) up to >= 1e8
```

## Install

```bash
pip install cholqr        # or: pip install git+https://github.com/createcentury/cholqr
```

## Citation

Please cite the algorithm:

> T. Fukaya, R. Kannan, Y. Nakatsukasa, Y. Yamamoto, Y. Yanagisawa. *Shifted Cholesky QR for computing the QR
> factorization of ill-conditioned matrices.* SIAM J. Sci. Comput. 42(1):A477–A503, 2020.

> Y. Yamamoto, Y. Nakatsukasa, Y. Yanagisawa, T. Fukaya. *Roundoff error analysis of the CholeskyQR2 algorithm.*
> Electron. Trans. Numer. Anal. 44:306–326, 2015.

## License

MIT
