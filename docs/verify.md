# Certified orthogonality (`verify=True`)

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

[Back to README](../README.md)
