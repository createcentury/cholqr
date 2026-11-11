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
fast = torch.compile(cholesky_qr, fullgraph=True)  # single graph, no host synchronization (no breakdown check)
```

**Breakdown detection and fallback (0.5).** CholeskyQR breaks down when the input is too ill-conditioned for the
chosen method. Before 0.5 the result then silently lost orthogonality. In eager mode, `cholqr` now detects this and
recomputes only the affected matrices: first with an FP64 first pass, then with Householder QR. The check (a failed factorization, or a last-pass Cholesky
diagonal below 0.195: a pass on X leaves orthogonality ~ c·u/Lmin² with c ≈ 18–33 measured, so 0.195 targets 1e-4 with
a 2x margin on c) reuses the small r x r Cholesky factors the passes already compute, so it adds no Gram product and no extra pass over
the data. It does cost one host synchronization per call. Choose the behaviour with `on_breakdown`:
`"fallback"` (default), `"warn"`, `"raise"` or `"ignore"` (the 0.4 behaviour, no synchronization). Inside
`torch.compile` the check is skipped.

| method | passes | reliable range in FP32 (tested) | use when |
|---|---|---|---|
| `"scqr3"` (default) | 3 | cond(Y) ≲ 1e6 | general use |
| `"scqr3"`, `robust=True` | 3 (first in FP64) | cond(Y) up to ≥ 1e8 | ill-conditioned inputs, data-center GPUs |
| `"cqr2"` | 2 | cond(Y) ≲ 4e3 guaranteed (= u^{-1/2}); with the chunked Gram of v0.4 it survives 1e4 and breaks between 1e4 and 3e4 (CPU) | known well-conditioned inputs, e.g. sketches without power iteration |
| `"auto"` | — | any | **recommended when shapes vary**: Householder (`torch.linalg.qr`) for a single matrix with r ≤ 64, otherwise `"scqr3"` |
| `"rcqr"` | sketch + 1 | cond(Y) up to ≥ 1e8 (orth ≈ 7e-6) | single large matrices; **slow when batched** (its preconditioner is a batched Householder QR) |

Beyond its range, a method breaks down abruptly. Since 0.5 this is detected and repaired in eager mode (see above),
but every repaired call pays for a second factorization. Pick the method for the conditioning you expect. Under
`torch.compile` there is no check, so use `robust=True` there if the conditioning is unknown.

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

1. **Split-K Gram products (v0.4).** For m >= 2048, the Gram product Q^T Q is computed as a batched GEMM over
   256-row chunks plus a sum. cuBLAS otherwise runs this small-output, long-reduction GEMM almost serially (A100:
   1.13 ms -> 0.08 ms for 4 x 6144 x 8 fp32). Summing shorter partial products also improves fp32 accuracy.
1. **Column scaling before every Gram product.** The shift is scale invariant. If a first-pass column norm overflows in FP32, that column is first divided by its maximum (branch-free; other columns are untouched, so results are bit-identical otherwise). Q is then correct for any finite input. R overflows when the true column norm exceeds the FP32 maximum, because it is not representable. Fixed in 0.5: before, entries above ~1e19 gave a zero Q.
2. **Explicit r x r triangular inverse applied as one batched GEMM.** This replaces a slow triangular solve over m rows.
3. **`cholesky_ex` with a per-matrix `torch.where` fallback.** The routine compiles with zero graph breaks. The
   `info` flags of these factorizations, plus the smallest diagonal entry of the last pass's factor, are the breakdown
   test of 0.5. Reading them is the only host synchronization, and it happens in eager mode only.

## Accuracy and range of validity

| input condition number | FP32 `scqr3` without check (`on_breakdown="ignore"`, = 0.4) | default (0.5, eager) | `robust=True` (FP64 first pass) |
|---|---|---|---|
| up to 1e6 | ‖QᵀQ − I‖_F ≈ 1e-5 | identical (no fallback triggered) | ‖QᵀQ − I‖_F ≈ 1e-5 |
| 5e6 – 1e8 | **breaks down** (‖QᵀQ − I‖_F ≈ 20–90) | ‖QᵀQ − I‖_F ≈ 1e-5 (repaired) | ‖QᵀQ − I‖_F ≈ 1e-5, subspace error ≈ 2e-6 |

These numbers were tested on 3072 x 210 inputs with geometric and power-law spectra (`tests/`).

Where the cliff is (CPU, FP32 input, 276 columns, geometric spectrum, one draw per point; `on_breakdown="ignore"`):
for 1024 rows it lies between cond(Y) 1e6 and 2e6. For 4096 rows it sits at about 3e6: there, 2 of 4 random draws
broke down and the other 2 did not, and all draws broke at 1e7. Orthogonality jumps from ≈ 1e-5 to ≈ 25 within
that range. In every broken case we measured, a Cholesky factorization in pass 2 or 3
failed (`info > 0`), and the 0.5 check catches exactly this. With the default fallback, orthogonality stays at
≈ 2e-6–8e-6 up to cond(Y) = 1e10. Q is then an accurate basis of the FP32-rounded Y. For large cond(Y) that range
can differ substantially from the range of the original higher-precision matrix, and no QR can undo that.

- **Theory vs. practice.** The guarantee of Fukaya et al. (2020, Thm. 4.1) needs m·n·u ≤ 1/64 and
  cond(Y) ≤ u⁻¹ / (96 (mn + n(n+1))). In FP32 (u = 2⁻²⁴), neither condition holds already at 4096 x 276: the second
  bound is below 1. So the FP32 range above is empirical, not a theorem. The breakdown check is what makes the
  default safe outside that range.
- **Power iteration** raises the condition number of a sketch roughly to the power 2h+1 for h steps. This is a
  heuristic: it can be badly off in both directions for individual matrices. Sketches that are harmless at h = 0 can
  therefore cross the cliff at h = 1. Use `robust=True` or keep the default check, and don't predict cond(Y) from h = 0.

- The default first-pass shift `1e-6 * tr(G)` is a heuristic. The a-priori shift of Fukaya et al. is impractically large in FP32 for large m·r.
- `robust=True` costs one FP64 Gram product. That is cheap on data-center GPUs (A100/H100) and slow on consumer GPUs.
- **Real optimizer states** (27 nanoGPT momentum snapshots, sketch width 210, bf16 power iteration as in training):
  - κ(Y) ≤ 4.5×10⁴.
  - No fallback was triggered.
  - The final low-rank update differs from an FP64 Householder reference by at most 7.6×10⁻⁵ (relative).
  - Sketches computed in bf16 are typically well inside the default range, because bf16 rounding caps their effective condition number.
- Exactly rank-deficient or zero inputs (including zero columns) are flagged as breakdowns. With the default they
  are recomputed by Householder QR, so Q has orthonormal columns and Y = Q R still holds (R then has zero diagonal
  entries). With `on_breakdown="ignore"` they return finite outputs that are not orthonormal (0.4 behaviour).
- **What the check does not catch.** It detects failed Cholesky factorizations and numerically rank-deficient
  last-pass inputs. It does not catch gradual loss of orthogonality:
  - `"rcqr"` above cond(Y) ~ 1e8: ≈ 3e-4 at 1e9.
  - (`"cqr2"` near its limit on some spectra, which used to degrade to ≈ 1e-2 without a failed factorization, is now
    caught by the last-pass threshold below.)
  - Custom `passes` (e.g. `passes=1`): the check is calibrated for the default pass counts only.
- **Cost for rank-deficient inputs.** If your inputs are routinely rank deficient (e.g. a sketch wider than the
  matrix rank), every such call takes the fallback: one host synchronization plus a Householder QR. Use
  `on_breakdown="ignore"` if you only need the range and not orthonormal extra columns.
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
Q = cholesky_qr(Y)               # fast default; beyond cond(Y) ~ 1e6 the affected matrices are recomputed
Q = cholesky_qr(Y, robust=True)  # ill-conditioned inputs, cond(Y) up to >= 1e8 at full speed (no fallback needed)
```

## What is proved, what is measured, and how to check your device

- **Proved (exact arithmetic; Lean 4 proofs in [`lean/`](lean/)):** the spectrum of the shifted first pass
  (singular values σᵢ/√(σᵢ²+s)); a positive-definiteness criterion for the shifted computed Gram matrix; and the
  a-posteriori orthogonality bound ‖QᵀQ − I‖ ≤ (s + e)‖L⁻¹‖² (+ a forming term). The last one is the bound of Fukaya et
  al. (2020, eq. 3.33), restated with every rounding term collected in e.
- **Measured (pre-registered experiments; details in the technical note):** where FP32 CholeskyQR breaks for several
  spectra and sizes; that the bound holds with the measured rounding terms and no safety margin; that the breakdown
  check flags every run whose orthogonality exceeded 1e-4 in validation data it was not tuned on; how inputs in
  bfloat16 / float16 / float8 / float64 behave; and the behaviour on several devices (table below).
- **Not covered:** the floating-point layer is not formalized; devices and settings not in the table; `torch.compile`
  (the check is skipped there); gradual loss of orthogonality for `"rcqr"` and custom `passes`.

Check your own device in about a second:

```bash
python -m cholqr.diagnose            # cuda, then mps, then cpu;  --full adds 4096 x 276
```

It reports the Gram-product error class (FP32 or TF32), the condition numbers where `scqr3` / `cqr2` break, whether
the default stays orthogonal, whether `robust=True` is available, and a timing against `torch.linalg.qr`.

<!-- DEVICE TABLE: filled from the device sweep (benchmarks/device_sweep.py) -->
| device | Gram error ε/u (FP32) | first breakdown, scqr3 / cqr2 (1024 x 64, geometric) | checks |
|---|---|---|---|
| Apple M4 CPU | 27–31 (m 1024), 5–6 (m 4096) | pending | all pass |
| Apple M4 GPU (MPS) | 27–30 (m 1024), 5–6 (m 4096) | pending | all pass; the TF32 setting has no effect |
| NVIDIA (several) | pending | pending | pending |

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
