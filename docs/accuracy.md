# Accuracy and range of validity

| input condition number | FP32 `scqr3` without check (`on_breakdown="ignore"`, = 0.4) | default (0.5, eager) | `robust=True` (FP64 first pass) |
|---|---|---|---|
| up to 1e6 | ‖QᵀQ − I‖_F ≈ 1e-5 | identical (no fallback triggered) | ‖QᵀQ − I‖_F ≈ 1e-5 |
| 5e6 – 1e8 | **can break down** (‖QᵀQ − I‖_F ≈ 20–90 when it does) | ‖QᵀQ − I‖_F ≈ 1e-5 (repaired) | ‖QᵀQ − I‖_F ≈ 1e-5, subspace error ≈ 2e-6 |

These numbers were tested on 3072 x 210 inputs with geometric and power-law spectra (`tests/`). The onset of the
breakdown is not a fixed number: in the device sweep (single draws, `scqr3`, FP32, `on_breakdown="ignore"`) the first
broken cond(Y) ranged from about 3e5 (clustered or staircase spectra) to about 3e7 for most geometric and power-law
cases (up to 1e8 for power-law spectra on the Apple M4 CPU), while linear and spike spectra often survived to 1e10 or
did not break at all. The same input can break on one platform and not on
another: 3072 x 210, geometric, cond 1e7 breaks on an Apple M4 CPU but not on an x86 CPU (AMD Ryzen 9 9950X); at
1e8 it breaks on both. The check below does not depend on where the cliff is.

Where the cliff is (CPU, FP32 input, 276 columns, geometric spectrum, one draw per point; `on_breakdown="ignore"`):
for 1024 rows it lies between cond(Y) 1e6 and 2e6. For 4096 rows it sits at about 3e6: there, 2 of 4 random draws
broke down and the other 2 did not, and all draws broke at 1e7. Orthogonality jumps from ≈ 1e-5 to ≈ 25 within
that range. In every broken case we measured, the first Cholesky factorization to fail (`info > 0`) was in
pass 2, never in pass 3, and the 0.5 check catches exactly this. With the default fallback, orthogonality stays at
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
  - **Wide matrices (r > 276).** The 0.195 last-pass threshold was validated for up to 276 columns. In a
    pre-registered test at 4096 x 512 and 8192 x 1024 (`benchmarks/results/thresh_wide_2026-11/`) it missed
    1 of 69 degraded runs at r = 512 and 6 of 84 at r = 1024; every missed run had orthogonality between 1.05e-4 and
    1.8e-4. There were no false alarms (0/546, 0/495). The threshold is unchanged; if wide matrices must stay below
    1e-4, use `verify=True` ([details](verify.md)).
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

[Back to README](../README.md)
