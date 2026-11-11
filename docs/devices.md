# What is proved, what is measured, and how to check your device

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
| AMD Instinct MI350X (torch 2.10, ROCm 7.1.1) | +5% | +2% | +2% | +1% | n/a¹ | 12.1x |

¹ Not measured: batched `torch.linalg.cholesky_ex` with r > 256 fails on this ROCm stack (see
[compatibility](compatibility.md#rocm-known-issues)).

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
| AMD Instinct MI350X (2.10, ROCm 7.1.1) | 16.3 / 3.9 | 1e7 / 3e4 | 1632 / 1632 | 62 / 62 (1 of 782) |

The Apple numbers are larger at m = 1024 because that size uses a plain GEMM (split-K starts at m = 2048).
`python -m cholqr` reports the same Gram-error class for a random well-conditioned matrix, so its numbers are smaller.
On the MI350X, `verify=True` was also checked on 1196 matrices (small set, precision "highest" and "high"): 0 false
acceptances (no certified Q had exact orthogonality above 1e-4, and the bound was never below the exact error); with the fallback every
returned Q had orthogonality ≤ 8.7e-5.
T4 (Turing) has no TF32, so `"high"` gives the same results as `"highest"` there (and no compensation is applied).
Not measured yet: NVIDIA A100 (run incomplete). Google TPU v5e (torch_xla 2.8): a sanity check runs with 0
mismatches, but the full sweep has not run (the benchmark assumed a scalar `info` from `cholesky_ex`); not a target yet. Results from other hardware are welcome (`benchmarks/device_sweep.py`).

[Back to README](../README.md)
