# cholqr

Fast, compile-friendly **shifted CholeskyQR3** for batched tall-skinny matrices in PyTorch.
It is a drop-in for `torch.linalg.qr(Y, mode="reduced")[0]` when you only need an orthonormal basis of `range(Y)`,
for example in randomized SVD and sketching, low-rank optimizers such as randomized Muon, or subspace iteration.

```python
import torch
from cholqr import cholesky_qr

Y = torch.randn(40, 768, 210, device="cuda")       # batch of tall-skinny matrices
Q = cholesky_qr(Y)                                 # shifted CholeskyQR3 (default, safe)
Q = cholesky_qr(Y, method="cqr2")                  # CholeskyQR2: ~1/3 cheaper, only for cond(Y) <~ 4e3
Q = cholesky_qr(Y, robust=True)                    # FP64 first pass: cond(Y) up to >= 1e8
Q, R = cholesky_qr(Y, return_r=True)               # also R (upper triangular), Y = Q R
fast = torch.compile(cholesky_qr, fullgraph=True)  # single graph, no host synchronization
```

| method | passes | reliable range in FP32 (tested) | use when |
|---|---|---|---|
| `"scqr3"` (default) | 3 | cond(Y) ≲ 1e6 | general use |
| `"scqr3"`, `robust=True` | 3 (first in FP64) | cond(Y) up to ≥ 1e8 | ill-conditioned inputs, data-center GPUs |
| `"cqr2"` | 2 | cond(Y) ≲ 4e3 (= u^{-1/2}; 1e4 measured on A100) | known well-conditioned inputs, e.g. sketches without power iteration |
| `"auto"` | — | any | **recommended when shapes vary**: Householder (`torch.linalg.qr`) for a single matrix with r ≤ 64, otherwise `"scqr3"` |
| `"rcqr"` | sketch + 1 | cond(Y) up to ≥ 1e8 (orth ≈ 7e-6) | single large matrices; **slow when batched** (its preconditioner is a batched Householder QR) |

`"cqr2"` breaks down abruptly beyond its range: orthogonality is lost, and no error is raised. Use it only when you know the conditioning.

**Example (randomized low-rank Muon on nanoGPT):** the measured cond(Y) is at most 70 without power iteration (h=0), so `"cqr2"` is safe there. With one power iteration (h=1) it reaches 4.5×10⁴, which needs `"scqr3"`.

## Why

On GPUs, batched Householder QR (cuSOLVER `geqr2`) processes the batch one matrix at a time.

On an NVIDIA A100, for the sketches of randomized low-rank Muon on nanoGPT (two weight banks, sketch width 210):

| | Householder QR | `cholqr` | full Newton–Schulz (reference) |
|---|---|---|---|
| optimizer time per step, both banks, one GPU | 117.6 ms | **15.2 ms** | 17.1 ms |

- `torch.linalg.qr` reaches about **1.6% of FP32 peak** here.
- On the QR step alone, `cholqr` is 3–14x faster.
- End to end (1490 steps, A100-80GB), replacing the QR removes about 100 ms per step. Randomized Muon then trains as fast as full Muon at unchanged perplexity.

Library benchmark on an **NVIDIA A100-SXM4-80GB** (torch 2.7.1+cu126, FP32 with `float32_matmul_precision("highest")`, compiled with `fullgraph=True`; mean of 20 calls):

| batch x m x r | `torch.linalg.qr` | `"scqr3"` | `"cqr2"` | `robust=True` | speedup scqr3 / cqr2 |
|---|---|---|---|---|---|
| 40 x 768 x 210 | 50.5 ms | 3.58 ms | 2.39 ms | 3.72 ms | **14.1x / 21.1x** |
| 10 x 3072 x 210 | 25.5 ms | 4.20 ms | 2.80 ms | 3.99 ms | **6.1x / 9.1x** |
| 16 x 4096 x 256 | 63.2 ms | 6.61 ms | 5.08 ms | 6.40 ms | **9.6x / 12.4x** |
| 4 x 768 x 64 | 1.48 ms | 1.36 ms | 0.98 ms | 1.41 ms | 1.1x / 1.5x |

- `"rcqr"` (randomized CholeskyQR, as in Dion): 37.4 / 11.4 / 21.6 / 2.50 ms for the four shapes above. Most of this is the batched Householder QR of the sketch (34.9 ms for 40 x 384 x 210). On a single 8192 x 512 matrix it takes 4.5 ms (scqr3: 3.3 ms). For batches, `robust=True` is both more accurate and about 10x faster. Details: `benchmarks/results/a100_rcqr_2026-11-07.txt`.
- On the A100, `robust=True` costs about the same as the default, because its FP64 throughput is high.
- Without compilation (eager), `"scqr3"` takes 3.77 / 4.33 / 6.58 / 1.85 ms for the four shapes. Compilation matters mainly for small problems.

Library benchmark on an **NVIDIA Tesla T4** (torch 2.14.1, CUDA 13.0; mean of 20 calls; `benchmarks/bench_qr.py`):

| batch x m x r | `torch.linalg.qr` | `cholqr` (compiled) | speedup | `robust=True` | max ‖QᵀQ − I‖_F |
|---|---|---|---|---|---|
| 40 x 768 x 210 | 51.7 ms | 10.8 ms | **4.8x** | 45.1 ms | 3.5e-6 |
| 10 x 3072 x 210 | 23.0 ms | 9.7 ms | **2.4x** | 40.4 ms | 3.5e-6 |
| 16 x 4096 x 256 | 62.4 ms | 25.5 ms | **2.4x** | 104.2 ms | 4.3e-6 |
| 4 x 768 x 64 | 1.55 ms | 1.68 ms | 0.9x | 6.2 ms | 1.5e-6 |

- The gain comes from batches of moderately wide sketches. For small problems (e.g. 4 x 768 x 64), Householder QR is already fast, so do not expect a speedup there.
- `robust=True` is slow on the T4 because the T4 has low FP64 throughput. Use it only for ill-conditioned inputs, preferably on A100/H100-class GPUs.

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
- **TF32:** if TF32 matmul is enabled (`torch.set_float32_matmul_precision("high")`, common in training scripts on Ampere and newer GPUs), the Gram products lose precision.
  - Measured on an A100-SXM4-80GB (torch 2.7.1, 3072 x 210, ‖QᵀQ − I‖_F):

    | cond(Y) | 1e1 | 1e3 | 4e3 | 1e4 | 1e6 | 1e8 |
    |---|---|---|---|---|---|---|
    | `scqr3`, highest | 1.5e-6 | 1.6e-6 | 1.6e-6 | 1.5e-6 | 2.4e-6 | breaks down |
    | `scqr3`, TF32 | 1.4e-4 | 1.9e-4 | 1.9e-4 | 2.1e-4 | 2.4e-4 | 2.8e-4 |
    | `robust`, highest | 1.2e-6 | 1.1e-6 | 1.6e-6 | 1.5e-6 | 1.6e-6 | 1.5e-6 |
    | `robust`, TF32 | 1.8e-4 | 1.8e-4 | 2.0e-4 | 1.8e-4 | 1.8e-4 | 1.8e-4 |
    | `cqr2`, highest | 1.6e-6 | 2.4e-6 | 2.4e-6 | 2.9e-6 | breaks down | breaks down |
    | `cqr2`, TF32 | 1.9e-4 | 2.0e-3 | 6.8e-3 | breaks down | breaks down | breaks down |

    TF32 costs about two orders of magnitude in orthogonality for every method, and `"cqr2"` loses accuracy from cond(Y) ~ 1e3 on.
    This agrees with our earlier CPU emulation.
  - `cholqr` warns in that case. Wrap the call with `torch.set_float32_matmul_precision("highest")` if you need full accuracy.
- Real dtypes only; complex inputs raise `TypeError`. Autograd works (gradients agree with Householder QR to ~1e-7 in tests). Runs on CUDA, CPU and Apple MPS.
- Status: **experimental (0.x)**. The API may change. Issues and benchmarks on other GPUs are welcome.

```python
Q = cholesky_qr(Y)               # fast default, cond(Y) <~ 1e6
Q = cholesky_qr(Y, robust=True)  # ill-conditioned inputs, cond(Y) up to >= 1e8
```

## `method="auto"`

CholeskyQR costs ~45 small kernel launches, so a single small matrix takes about 1 ms eager and 0.3 ms or more compiled.
`torch.linalg.qr` takes 0.2–0.9 ms there. `"auto"` picks the faster one: on an A100 it is within a few percent of the
better of `torch.linalg.qr` and `"scqr3"` for every shape we measured (`benchmarks/results/a100_v03_2026-11-07.txt`).
On the Householder path, R is normalized to a non-negative diagonal, as in the CholeskyQR methods.

## PowerSGD drop-in (`cholqr.contrib.powersgd`)

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
