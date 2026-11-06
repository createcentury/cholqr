# Changelog

## Unreleased
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
