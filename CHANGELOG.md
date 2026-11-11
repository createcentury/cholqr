# Changelog

## 0.5.0 (unreleased)
- **Breakdown detection and fallback (new default).** Beyond its range (FP32 `scqr3`: cond(Y) about 1e7; `cqr2`: about
  3e4), CholeskyQR used to return a non-orthonormal Q without any error. A matrix is now flagged when a pass's
  `cholesky_ex` fails, when its Cholesky factor has a non-finite diagonal, or when the last pass's Cholesky factor has a
  diagonal entry below 0.195. The last test catches gradual loss of orthogonality without a failed factorization, which
  occurs for `cqr2` near its limit. A pass on X leaves orthogonality of about c·u/Lmin² (c = 18-33 measured for FP32
  Gram orders), and 0.195 = sqrt(64 u / 1e-4) targets 1e-4 with a 2x margin on c. In a pre-registered validation on
  data not used to choose it, the threshold flagged every run whose orthogonality exceeded 1e-4 (155/155 on CPU,
  56/56 on an Apple GPU) with 4/1931 and 0/760 false alarms; an earlier draft value of 1e-2 caught 65/155.
  All tests reuse the r x r factors; there is no extra Gram product.
  - New argument `on_breakdown`: `"fallback"` (default) recomputes only the flagged matrices, first with an FP64
    first pass and then with Householder QR; `"warn"`; `"raise"` (`torch.linalg.LinAlgError`); `"ignore"` (0.4
    behaviour).
  - **Behaviour change:** every option except `"ignore"` costs one host synchronization per call in eager mode
    (measured +0-7% on NVIDIA GPUs, up to +90% for a single small matrix on Apple GPUs). The synchronization also means
    the default cannot be captured in a raw CUDA graph; use `on_breakdown="ignore"` or `torch.compile` there.
  - Under `torch.compile` and inside `torch.func` transforms (`vmap`, `grad`) the check is skipped, so compiled graphs
    are unchanged and vmap works.
  - Inputs inside the range give bit-identical results to 0.4 (checked for Q and R, all methods).
  - Exactly rank-deficient inputs (zero columns) now get an orthonormal Q with the default.
  - Not caught: gradual loss of orthogonality of `"rcqr"` above cond ~1e8, and custom `passes` (the threshold is
    calibrated for the default pass counts; e.g. `robust=True, passes=1` is judged with the FP32 threshold).
- **Fix:** FP32 inputs with entries above ~1e19 overflowed the first column-norm computation and returned Q = 0.
  Columns whose norm overflows are now divided by their maximum first (branch-free; results for other inputs are
  unchanged). This applies to all methods, including the pre-scaling of `rcqr`.
- **Fix:** `passes=0, return_r=True` raised `AttributeError`; R is now the identity.
- **Fix:** a factorization whose Cholesky factor has a non-finite diagonal now counts as failed even if `cholesky_ex`
  reports success (observed once on an RTX 4090 with torch 2.8).
- **Fix:** the fallback merge works for every dtype (it failed for float8 inputs).
- **Compatibility:** torch 2.1-2.3 work again (`torch.compiler.is_compiling` fallback; `dtype.is_complex` under Dynamo).
  0.4.0 declared `torch>=2.1` but failed on torch < 2.3. On torch < 2.4, compiling `method="rcqr"` fails (2.1) or hangs (2.3) in Inductor.
- `contrib.powersgd` passes `on_breakdown="ignore"` because it has its own orthogonality check.
- `cholqr.diagnose()` / `python -m cholqr`: measures in a few seconds the Gram error class (FP32 vs TF32), the
  breakdown points, the default's orthogonality, robust availability and timing on the current device.
- `benchmarks/device_sweep.py`, `benchmarks/bench_overhead.py`: the scripts behind the README tables.
- `lean/`: Lean 4 proofs of the exact-algebra results (shifted-pass spectrum, positive-definiteness criterion, the
  a-posteriori orthogonality bound of Fukaya et al. 2020 eq. 3.33); the floating-point layer is not formalized.
- **TF32:** when TF32 is active on an Ampere-or-newer NVIDIA GPU, the Gram and forming products use the 3xTF32 scheme
  (exact split into a TF32 part and a remainder; two or three TF32 GEMMs) without changing global settings. Measured on an
  RTX 4090: Gram error 25-119 u instead of 3000-6700 u; runs with orthogonality above 1e-4 down from 399 to 24 of 1632
  (worst 7.3e-4). Not full FP32: the breakdown point moves from ~1e7 to ~1e6, the check catches few of the remaining
  degraded runs, and on consumer GPUs the call is ~2x slower than with "highest". Use "highest" for full accuracy.
- CI: Python 3.9-3.15, torch 2.1 to latest and nightly (Linux x86 and ARM), numpy 1 and 2, macOS, Windows.
- Tests for execution modes: deterministic algorithms, inference_mode, vmap, double backward, default dtype float64,
  edge shapes, non-contiguous inputs, CUDA graph capture (with `on_breakdown="ignore"`), second GPU.
- README: measured FP32 breakdown points and device table, cost of the check, why the Fukaya et al. guarantee does
  not cover FP32 at these sizes, a caution on power-iterated sketches.

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
