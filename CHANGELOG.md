# Changelog

## Unreleased (0.4.0)
- **Faster Gram products.** Q^T Q is now computed with a manual split-K: the m rows are chunked into blocks of 256, then
  a batched GEMM, then a sum.
  - cuBLAS runs this small-output, long-reduction GEMM nearly serially. On an A100, 4 x 6144 x 8 fp32 took 1.13 ms;
    chunked it takes 0.08 ms.
  - The result is mathematically identical. The shorter summations also make fp32 Gram products more accurate:
    `cqr2` now survives cond(Y) ~ 1e4 on CPU (previously it broke down at ~4e3).

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
