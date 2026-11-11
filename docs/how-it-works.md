# How cholqr works

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

[Back to README](../README.md)
