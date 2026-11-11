# Changelog

## 0.5.0 (unreleased)
- **Breakdown detection and fallback.** Beyond its range (FP32 `scqr3`: cond(Y) ~ 1e6–1e7), CholeskyQR used to return
  a non-orthonormal Q without any error. In every case we measured, a Cholesky factorization in a later pass failed;
  0.4 skipped that pass silently. Matrices are now flagged when any pass's `cholesky_ex` fails, or when the last
  pass's Cholesky factor has a diagonal entry below 1e-2 (numerically rank-deficient input, e.g. zero columns). The
  flags reuse the r x r factors, so there is no extra Gram product.
  - New argument `on_breakdown`: `"fallback"` (default) recomputes only the flagged matrices, first with an FP64
    first pass and then with Householder QR; `"warn"`; `"raise"` (`torch.linalg.LinAlgError`); `"ignore"` (0.4
    behaviour).
  - Every option except `"ignore"` costs one host synchronization per call in eager mode. Under `torch.compile` the
    check is skipped, so compiled graphs are unchanged.
  - Inputs inside the range give bit-identical results to 0.4 (checked for Q and R, all methods).
  - Not caught: gradual loss of orthogonality without a failed factorization (rcqr above ~1e8, cqr2 near its
    limit on some spectra, custom `passes`). See the README.
  - Exactly rank-deficient inputs (zero columns) now get an orthonormal Q with the default.
- **Fix:** FP32 inputs with entries above ~1e19 overflowed the first column-norm computation and returned Q = 0.
  Columns whose norm overflows are now divided by their maximum first (branch-free; results for other inputs are
  unchanged). This applies to all methods, including the pre-scaling of `rcqr`.
- **Fix:** `passes=0, return_r=True` raised `AttributeError`; R is now the identity.
- `contrib.powersgd` passes `on_breakdown="ignore"` because it has its own orthogonality check (one host sync per
  call, as before).
- README: measured position of the FP32 cliff (QRBOUND sweep), why the Fukaya et al. guarantee does not cover FP32
  at these sizes, and a caution on power-iterated sketches.

## 0.4.0 (2026-11-08)
- **Faster Gram products.** For m >= 2048 rows, Q^T Q is now computed with a manual split-K: the m rows are chunked
  into blocks of 256, then a batched GEMM, then a sum. Smaller m keeps the plain GEMM, where chunking only adds
  launches.
  - cuBLAS runs this small-output, long-reduction GEMM nearly serially. On an A100, 4 x 6144 x 8 fp32 took 1.13 ms;
    chunked it takes 0.08 ms.
  - The result is mathematically identical. The shorter summations also make fp32 Gram products more accurate:
    `cqr2` now survives cond(Y) ~ 1e4 on CPU (previously it broke down at ~4e3).
  - The speed tables in the README were measured with 0.3.0 and are not refreshed. The change affects the Gram passes
    only, which were the slowest part for tall inputs.
- Tests: the DDP-hook test is skipped when numpy is missing (PyTorch's PowerSGDState imports it). New `test` extra.

## 0.3.0 (2026-11-07)
- `method="auto"`: Householder `torch.linalg.qr` for a single matrix with r <= 64 (where CholeskyQR's launch overhead dominates), `"scqr3"` otherwise. Accepts `robust`.
- `cholqr.contrib.powersgd`: drop-in for PyTorch's PowerSGD `_orthogonalize` (`patch()` / `unpatch()`). Single matrices: upcast to FP32 + Householder. Batches: CholeskyQR2 with a verified fallback. A100 bf16: up to 14x faster and about 10x more orthogonal.
- Fewer kernel launches per pass: the identity is built once, and the trace is not recomputed (columns are unit-norm).
- `method="rcqr"`: randomized (sketch-preconditioned) CholeskyQR (Balabanov 2022; Garrison & Ipsen 2024; as in Dion). Robust in FP32 up to cond(Y) >= 1e8; slow for batches on GPU (batched Householder QR of the sketch). Options `oversample`, `generator`.
- A100 comparison of all methods: `benchmarks/results/a100_rcqr_2026-11-07.txt`.

## 0.2.0 (2026-11-07)
- `method="cqr2"`: CholeskyQR2 (two passes), about 1/3 cheaper; for well-conditioned inputs only (cond(Y) <~ 4e3 in FP32).
- `return_r=True`: also return the upper-triangular R with Y = Q R (accumulated exactly across passes, including column scaling).
- `orthonormal_basis` accepts the same options.
- Runtime warning when TF32 matmul is enabled on CUDA (degrades accuracy; measured on A100, see README).
  The check is skipped while compiling, so `torch.compile(fullgraph=True)` also works on older torch (fixes a Dynamo `Unsupported` error on torch 2.7).

## 0.1.0 (2026-11-07)
- Shifted CholeskyQR3 for batched tall-skinny matrices; `robust=True` (FP64 first pass); real dtypes only; compile-friendly.
