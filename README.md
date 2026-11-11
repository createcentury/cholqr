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
`torch.compile` the check is skipped. This detector is **empirical** (calibrated and validated, not proved; under
TF32 it misses most degraded runs). For a guarantee use `verify=True` (below).

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

Shifted CholeskyQR3 (Fukaya et al., 2020) with four implementation choices:

1. **Split-K Gram products (v0.4).** For m >= 2048, the Gram product Q^T Q is computed as a batched GEMM over
   256-row chunks plus a sum. cuBLAS otherwise runs this small-output, long-reduction GEMM almost serially (A100:
   1.13 ms -> 0.08 ms for 4 x 6144 x 8 fp32). Summing shorter partial products also improves fp32 accuracy.
2. **Column scaling before every Gram product.** The shift is scale invariant. If a first-pass column norm overflows in FP32, that column is first divided by its maximum (branch-free; other columns are untouched, so results are bit-identical otherwise). Q is then correct for any finite input. R overflows when the true column norm exceeds the FP32 maximum, because it is not representable. Fixed in 0.5: before, entries above ~1e19 gave a zero Q.
3. **Explicit r x r triangular inverse applied as one batched GEMM.** This replaces a slow triangular solve over m rows.
4. **`cholesky_ex` with a per-matrix `torch.where` fallback.** The routine compiles with zero graph breaks. The
   `info` flags of these factorizations, plus the smallest diagonal entry of the last pass's factor, are the breakdown
   test of 0.5. Reading them is the only host synchronization, and it happens in eager mode only.

## Accuracy and range of validity

| input condition number | FP32 `scqr3` without check (`on_breakdown="ignore"`, = 0.4) | default (0.5, eager) | `robust=True` (FP64 first pass) |
|---|---|---|---|
| up to 1e6 | ‖QᵀQ − I‖_F ≈ 1e-5 | identical (no fallback triggered) | ‖QᵀQ − I‖_F ≈ 1e-5 |
| 5e6 – 1e8 | **can break down** (‖QᵀQ − I‖_F ≈ 20–90 when it does) | ‖QᵀQ − I‖_F ≈ 1e-5 (repaired) | ‖QᵀQ − I‖_F ≈ 1e-5, subspace error ≈ 2e-6 |

These numbers were tested on 3072 x 210 inputs with geometric and power-law spectra (`tests/`). The onset of the
breakdown is not a fixed number: in the device sweep (single draws, `scqr3`, FP32, `on_breakdown="ignore"`) the first
broken cond(Y) ranged from about 3e5 (clustered or staircase spectra) to about 3e7 (geometric and power-law at 64
columns), and spike spectra often survived to 1e9 or beyond. The same input can break on one platform and not on
another: 3072 x 210, geometric, cond 1e7 breaks on an Apple M4 CPU but not on an x86 CPU (AMD Ryzen 9 9950X); at
1e8 it breaks on both. The check below does not depend on where the cliff is.

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
  are recomputed by Householder QR, so Q has orthonormal columns and Y = Q R still holds (R then has numerically zero diagonal
  entries). With `on_breakdown="ignore"` they return finite outputs that are not orthonormal (0.4 behaviour).
- **What the check does not catch.** It detects failed Cholesky factorizations and numerically rank-deficient
  last-pass inputs. It does not catch gradual loss of orthogonality:
  - `"rcqr"` above cond(Y) ~ 1e8: ≈ 3e-4 at 1e9.
  - (`"cqr2"` near its limit on some spectra, which used to degrade to ≈ 1e-2 without a failed factorization, is now
    caught by the last-pass threshold described above.)
  - Custom `passes` (e.g. `passes=1`): the check is calibrated for the default pass counts only.
- **Cost for rank-deficient inputs.** If your inputs are routinely rank deficient (e.g. a sketch wider than the
  matrix rank), every such call takes the fallback: one host synchronization plus a Householder QR. Use
  `on_breakdown="ignore"` if you only need the range and not orthonormal extra columns.
- **TF32** (`torch.set_float32_matmul_precision("high")`, common in training scripts on Ampere and newer GPUs).
  Plain TF32 Gram products have errors of ~3000-6700 u instead of ~5 u. Measured with 0.4-style plain products (device
  sweep, 1632 runs per GPU, torch 2.8): scqr3 then never breaks down up to cond(Y) = 1e10, but the default's
  orthogonality degrades to 1.7e-3 - 3.1e-3, 15-30% of runs exceed 1e-4, and the breakdown check cannot see it.
  Since 0.5, cholqr **compensates** TF32 on Ampere-or-newer NVIDIA GPUs with the 3xTF32 scheme (exact split of each
  operand into a TF32 part and a remainder; two or three TF32 GEMMs; no global setting is changed). Measured (same
  sweep): median Gram error 25-119 u on an RTX 4090 and 16-76 u on an H100; runs above 1e-4: 24 of 1632 (worst 7.3e-4)
  on the RTX 4090 and 35 of 1632 (worst 1.0e-3) on the H100; compiled path unchanged. This is **not full FP32**: the
  breakdown point moves from ~1e7 to ~1e6 and the check catches only a few of the remaining degraded runs. It is also
  **slower than full FP32**: the compensated call takes about 1.3-1.7x (H100) and ~2x (RTX 4090) the time of the same
  call at `"highest"`, so under TF32 cholqr 0.5 is slower than 0.4 was. There is no switch to get 0.4's plain-TF32
  behaviour back. **If you need full accuracy, wrap the call with `torch.set_float32_matmul_precision("highest")`.**
- Real dtypes only; complex inputs raise `TypeError`. Autograd works (gradients agree with Householder QR to ~1e-7 in tests). Runs on CUDA, CPU and Apple MPS.
- Status: **experimental (0.x)**. The API may change. Issues and benchmarks on other GPUs are welcome.

```python
Q = cholesky_qr(Y)               # fast default; beyond cond(Y) ~ 1e6 the affected matrices are recomputed
Q = cholesky_qr(Y, robust=True)  # ill-conditioned inputs, cond(Y) up to >= 1e8 at full speed (no fallback needed)
```

## Certified orthogonality (`verify=True`)

```python
Q = cholesky_qr(Y, verify=True)            # every returned matrix satisfies ||Q^T Q - I||_F <= 1e-4 (proved under IEEE FP64 assumptions H0-H2)
Q = cholesky_qr(Y, verify=True, tol=1e-6)
b = cholqr.orthogonality_bound(Q)           # the bound itself, per matrix (float64)
```

**What is guaranteed.** With `verify=True`, a returned Q (each matrix of a batch) satisfies ‖QᵀQ − I‖_F ≤ `tol` for
the *exact* values of the returned tensor, not for a computed estimate. The check computes the Gram matrix Ĝ of the
returned Q in FP64 and accepts it iff `fl(α·ê + β·tr̂) ≤ tol`, where ê is the computed ‖Ĝ − I‖_F,
α = 1 + (r² + 4)u + O(u²), β = m·u·(1 + O((m + r)u)), u = 2^-53 (constants computed in exact rational arithmetic
and rounded up). In words: the computed error plus a slack of about m·u·tr(Ĝ) ≈ m·r·u, e.g. 1.3e-10 for 4096 x 276.
The proof uses that every value cholqr returns is an FP32 value (float16/bfloat16/float8 values are too), so it is exact
in FP64, products of two entries are exact, and nothing underflows; the Gram rounding is then bounded by
γ_m·Σ|q_ki||q_kj| for *any* summation order, blocking or split-K, with or without FMA; Cauchy–Schwarz and the computed
diagonal bound the true column norms; the rounding of the norm, the trace and the final multiply-add are included.
The algebraic part is proved in Lean ([`lean/CholQR/Verify.lean`](lean/CholQR/Verify.lean)); the floating-point
facts are proved on paper. It does not matter how Q was computed (TF32, a broken-down factorization, another device).

**Assumptions.** (H0) the checked entries are finite, normal-or-zero FP32 values (true for every value cholqr returns,
except FP32 subnormals); (H1) IEEE 754 binary64 with round-to-nearest (standard model; correctly rounded sqrt);
(H2) the FP64 GEMM sums the products in any order, with or without FMA, without Strassen-type algorithms, FP64
emulation on lower-precision units or x87 extended-precision accumulation. Where the check runs: cuBLAS DGEMM for
NVIDIA CUDA tensors; the CPU BLAS for CPU, MPS and ROCm tensors (ROCm's rocBLAS/hipBLASLt are excluded, the tensor is
copied to the CPU). That the CPU BLAS satisfies (H2) — e.g. that macOS Accelerate, OpenBLAS or MKL do not use
Strassen-type DGEMM — is assumed, not checked. If `CUBLAS_EMULATE_DOUBLE_PRECISION` or `CUBLAS_EMULATION_STRATEGY` is
set (cuBLAS FP64 emulation, 13.0u2+), CUDA checks return `inf` with a warning. Matrices with an FP32 subnormal entry
get `inf` (not certified): with flush-to-zero / denormals-are-zero modes (`torch.set_flush_denormal(True)`) the FP64
copy would differ from the returned values; detection uses bit patterns, so it works in every mode, and no other
intermediate of the check is ever subnormal. float64 tensors passed to `orthogonality_bound` must hold FP32 values
(otherwise `inf`); non-finite matrices are never certified.

**What happens when a matrix is not certified** (`on_breakdown`): `"fallback"` (default) recomputes only those matrices
(scqr3 with an FP64 first pass where the device has FP64, then Householder QR, in FP64 where available, rounded through
FP32), re-verifies, and raises `torch.linalg.LinAlgError` if one still fails, so a returned Q is always certified;
`"warn"` warns and returns the uncertified result (a Python `RuntimeWarning`: under the default warning filter it is
shown once per call site, repeated failures from the same line are silent; use `"raise"` or
`warnings.simplefilter("always")` if every failure must be seen); `"raise"` raises; `"ignore"` is rejected (`ValueError`; use
`orthogonality_bound`). With `verify=True` the certificate replaces the empirical flags (a matrix the detector flags
but the certificate accepts is returned as is).

| mode | `verify=True` |
|---|---|
| eager CPU / NVIDIA CUDA | check in FP64 on the tensor's device; one host synchronization |
| eager ROCm | Q is copied to the CPU and checked there (rocBLAS/hipBLASLt not assumed) |
| eager MPS | Q is copied to the CPU (m·r·4 bytes per matrix) and checked there; fallback runs on MPS (Householder in FP32) |
| `torch.compile`, `torch.export` | `RuntimeError` at call time (no data-dependent fallback inside a graph); compile with `verify=False` and call `orthogonality_bound(Q)` outside |
| `torch.func.vmap` / `grad`, `torch.jit.trace` | `RuntimeError` at call time; same workaround |
| autograd | unchanged: the check runs under `no_grad` on a detached copy |
| CUDA graphs | not capturable (host synchronization) |

**Cost** (median, verify=True vs the default, fp32, no fallback triggered; `benchmarks/verify_eval.py --part v3`):
| device (torch 2.15) | 1 x 4096 x 276 | 8 x 4096 x 276 | 16 x 1024 x 64 | 4 x 8192 x 128 |
|---|---|---|---|---|
| Apple M4 CPU | +27% (6.0 → 7.6 ms) | +25% (38 → 47 ms) | +27% (3.4 → 4.3 ms) | +35% (12.0 → 16.2 ms) |
| Apple M4 GPU (MPS; check on CPU) | +75% (3.2 → 5.6 ms) | +83% (13.5 → 24.7 ms) | +38% (1.5 → 2.1 ms) | +134% (3.3 → 7.7 ms) |

Single run on a loaded machine; CUDA not measured yet. The cost is one FP64 Gram product (2·m·r² flops per matrix);
on GPUs with slow FP64 (consumer cards run FP64 at 1/64 of FP32) expect a larger relative cost.

**What is not covered.** Orthogonality only: not the accuracy of the subspace range(Q) vs range(Y), nor of R
(Y = QR). A residual certificate for ‖Y − QR‖ is future work. The default `tol=None` means 1e-4 for float32/float64
output; for Q requested in float16/bfloat16/float8 a `tol` must be given (`ValueError` at call time otherwise), because
rounding Q alone moves the error by up to ~2√r·u_d (u_d the format's unit roundoff): use `out_dtype=torch.float32`, or
tol ≳ 1e-4 + 2√r·u_d + r·u_d². The tolerance is never relaxed silently. Why FP64: the same argument run in FP32 gives a slack of ~m·r·2^-24 ≈ 0.07 for 4096 x 276, far above 1e-4.

## What is proved, what is measured, and how to check your device

- **Proved (exact arithmetic; Lean 4 proofs in [`lean/`](lean/)):** the spectrum of the shifted first pass
  (singular values σᵢ/√(σᵢ²+s)); a positive-definiteness criterion for the shifted computed Gram matrix; and the
  a-posteriori orthogonality bound ‖QᵀQ − I‖ ≤ (s + e)‖L⁻¹‖² (+ a forming term). The last one is the bound of Fukaya et
  al. (2020, eq. 3.33), restated with every rounding term collected in e.
- **Measured (pre-registered experiments; scripts in `benchmarks/`, a technical note is in preparation):** where FP32 CholeskyQR breaks for several
  spectra and sizes; that the bound holds with the measured rounding terms and no safety margin; that the breakdown
  check flags every run whose orthogonality exceeded 1e-4 in validation data it was not tuned on; how inputs in
  bfloat16 / float16 / float8 / float64 behave; and the behaviour on several devices (table below).
- **Proved (floating point):** the `verify=True` certificate (section above): a returned Q satisfies
  ‖QᵀQ − I‖_F ≤ tol under IEEE FP64 arithmetic in any summation order (algebraic part in Lean, rounding facts on paper).
- **Not covered:** the floating-point layer of the CholeskyQR analysis is not formalized (the default detector is
  empirical); devices and settings not in the table; `torch.compile`
  (the check is skipped there); gradual loss of orthogonality for `"rcqr"` and custom `passes`.

Check your own device in a few seconds:

```bash
python -m cholqr            # cuda, then mps, then cpu;  --full adds 4096 x 276
```

It reports the Gram-product error class (FP32 or TF32), the condition numbers where `scqr3` / `cqr2` break, whether
the default stays orthogonal, whether `robust=True` is available, and a timing against `torch.linalg.qr`.

**Cost of the breakdown check** (`on_breakdown="fallback"`, the default, vs `"ignore"`; well-conditioned inputs, so no
fallback runs; median of 50 calls; torch 2.8, CUDA 12.8; `benchmarks/bench_overhead.py`):

| GPU | 1 x 1024 x 64 | 1 x 4096 x 276 | 16 x 2048 x 128 | 40 x 768 x 210 | 8 x 4096 x 276 | speed-up vs `torch.linalg.qr`, 40 x 768 x 210 |
|---|---|---|---|---|---|---|
| H100 80GB HBM3 | +5% | +4% | +3% | +1% | +1% | 14.4x |
| A100-SXM4-80GB | +3% | +2% | +2% | +1% | +0% | 12.4x |
| A40 | +3% | +2% | +2% | +1% | +0% | 10.5x |
| RTX 4090 | +2% | +2% | +2% | +1% | +1% | 12.0x |
| RTX 5090 | +7% | +2% | +3% | +1% | +0% | 16.9x |
| RTX PRO 6000 Blackwell | +7% | +2% | +2% | +1% | +0% | 18.5x |
| L4 | +3% | +2% | +1% | +0% | +0% | 6.1x |

The check costs one host synchronization (0.02-0.04 ms on these GPUs). On Apple GPUs (MPS) a synchronization is more
expensive: up to +90% for a single small matrix, about 0% for large batches. For a single small matrix
`torch.linalg.qr` is faster than any CholeskyQR, which is what `method="auto"` uses.

Device sweep (`benchmarks/device_sweep.py`; FP32 inputs, true FP32 matmul; geometric, power-law, spike, cluster, linear
and staircase spectra; condition numbers 1e2-1e10 in half-decades; 2 draws; scqr3 and cqr2). ε/u is the median Gram
error of pass 2 in units of u = 2^-24 (m = 1024 / 4096, r = 276); "breaks at" is the smallest condition number of the grid
(geometric spectrum, 4096 x 276) where a factorization fails with `on_breakdown="ignore"`. Checks = the a-posteriori
bound holds with measured terms; the default returns ‖QᵀQ − I‖ < 1e-4 in every run; the last-pass threshold flags every
run with orthogonality > 1e-4 (false alarms in parentheses).

| device (torch) | ε/u | breaks at: scqr3 / cqr2 | default orthogonality < 1e-4 | flagged / degraded (false alarms) |
|---|---|---|---|---|
| Apple M4 CPU (2.15) | 30.6 / 6.0 | 1e7 / 3e4 | 1632 / 1632 | 60 / 60 (2 of 753) |
| Apple M4 GPU, MPS (2.15) | 30.4 / 5.5 | 1e7 / 3e4 | 1632 / 1632 | 61 / 61 (2 of 761) |
| NVIDIA H100 80GB HBM3 (2.8) | 5.1 / 5.2 | 1e7 / 3e4 | 1632 / 1632 | 58 / 58 (2 of 809) |
| NVIDIA A40 (2.8) | 4.1 / 5.3 | 1e7 / 3e4 | 1632 / 1632 | 58 / 58 (4 of 819) |
| NVIDIA RTX 4090 (2.8) | 4.1 / 5.3 | 1e7 / 3e4 | 1632 / 1632 | 58 / 58 (4 of 818) |
| NVIDIA RTX 5090 (2.8) | 5.1 / 5.3 | 1e7 / 3e4 | 1632 / 1632 | 60 / 60 (1 of 815) |
| NVIDIA RTX PRO 6000 Blackwell (2.8) | 5.2 / 5.3 | 1e7 / 3e4 | 1632 / 1632 | 60 / 60 (1 of 813) |
| NVIDIA L4 (2.8) | 5.5 / 5.3 | 1e7 / 3e4 | 1632 / 1632 | 57 / 57 (3 of 816) |
| NVIDIA T4 (2.14, Colab) | 7.8 / 5.3 | 1e7 / 3e4 | 1632 / 1632 | 56 / 56 (3 of 817) |

The Apple numbers are larger at m = 1024 because that size uses a plain GEMM (split-K starts at m = 2048).
`python -m cholqr` reports the same Gram-error class for a random well-conditioned matrix, so its numbers are smaller.
T4 (Turing) has no TF32, so `"high"` gives the same results as `"highest"` there (and no compensation is applied).
Not measured yet: NVIDIA A100 (run incomplete). Google TPU v5e (torch_xla 2.8): a sanity check runs with 0
mismatches, but the full sweep has not run (the benchmark assumed a scalar `info` from `cholesky_ex`); not a target yet. Results from other hardware are welcome (`benchmarks/device_sweep.py`).

## Compatibility

| | tested | notes |
|---|---|---|
| Python | 3.9-3.15 | CI |
| PyTorch | 2.1 to latest; nightly in CI (allowed to fail) | on torch < 2.4, compiling `method="rcqr"` fails or hangs in Inductor |
| OS / CPU | Linux x86-64 and ARM64, macOS (Apple silicon), Windows | CI |
| NumPy | 1.x and 2.x | only the PowerSGD DDP test uses it |
| NVIDIA GPUs | Turing, Ampere, Ada, Hopper, Blackwell (T4, A40, L4, RTX 4090/5090, RTX PRO 6000, H100) | device table above |
| Apple GPUs | MPS | no FP64: the fallback uses Householder directly |
| Input dtypes | float64, float32, bfloat16, float16, float8_e4m3fn | computation is FP32 (FP64 first pass with `robust=True`) |
| Matmul precision | `highest` (full accuracy); `high` (TF32: partly compensated, see Accuracy) | `medium` may use bfloat16 products on some backends: not compensated |
| Execution modes | eager, `torch.compile` (inductor), `torch.func.vmap`, `torch.export` (with `on_breakdown="ignore"`), autograd incl. double backward, `inference_mode`, deterministic algorithms (CPU), threads, spawned processes | the breakdown check is skipped under `torch.compile`, `vmap` and tracing; `verify=True` raises there |
| CUDA execution modes | deterministic mode, CUDA graphs, reduce-overhead compile, second GPU (2x RTX 4090, torch 2.8: full test suite 124 passed) | tests |
| Not yet verified | Volta GPUs (V100), A100, AMD ROCm, Intel XPU, TPU (PyTorch/XLA: sanity only), FSDP/DTensor inputs | reports welcome |
| Not supported | complex dtypes (raise `TypeError`), TorchScript | |

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
