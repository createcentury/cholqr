# Compatibility

| | tested | notes |
|---|---|---|
| Python | 3.9-3.15 | CI |
| PyTorch | 2.1 to latest; nightly in CI (allowed to fail) | on torch < 2.4, compiling `method="rcqr"` fails or hangs in Inductor |
| OS / CPU | Linux x86-64 and ARM64, macOS (Apple silicon), Windows | CI |
| NumPy | 1.x and 2.x | only the PowerSGD DDP test uses it |
| NVIDIA GPUs | Turing, Ampere, Ada, Hopper, Blackwell (T4, A40, L4, RTX 4090/5090, RTX PRO 6000, H100) | [device table](devices.md) |
| AMD GPUs (ROCm) | Instinct MI350X (torch 2.10, ROCm 7.1.1) | device sweep, `verify=True` and speed in the [device table](devices.md); see [known issues](#rocm-known-issues) |
| Apple GPUs | MPS | no FP64: the fallback uses Householder directly |
| Input dtypes | float64, float32, bfloat16, float16, float8_e4m3fn | computation is FP32 (FP64 first pass with `robust=True`) |
| Matmul precision | `highest` (full accuracy); `high` (TF32: partly compensated, see [Accuracy](accuracy.md)) | `medium` may use bfloat16 products on some backends: not compensated |
| Execution modes | eager, `torch.compile` (inductor), `torch.func.vmap`, `torch.export` (with `on_breakdown="ignore"`), autograd incl. double backward, `inference_mode`, deterministic algorithms (CPU), threads, spawned processes | the breakdown check is skipped under `torch.compile`, `vmap` and tracing; `verify=True` raises there |
| CUDA execution modes | deterministic mode, CUDA graphs, reduce-overhead compile, second GPU (2x RTX 4090, torch 2.8: full test suite 124 passed) | tests |
| Not yet verified | Volta GPUs (V100), A100, Intel XPU, TPU (PyTorch/XLA: sanity only), FSDP/DTensor inputs | reports welcome |
| Not supported | complex dtypes (raise `TypeError`), TorchScript | |

## ROCm known issues

Measured on an AMD Instinct MI350X with torch 2.10 + ROCm 7.1.1 (`benchmarks/results/rocm_mi350x_2026-11/`):

- **Batched Cholesky with more than 256 columns fails.** `torch.linalg.cholesky_ex` raises
  `HIP error: unspecified launch failure` for batches of 2 or more matrices with r >= 257 (single matrices, and batches
  with r <= 256, work). This is an upstream PyTorch/ROCm issue, not a cholqr bug, and cholqr does not catch it: a HIP
  launch failure can leave the GPU context unusable. Until it is fixed upstream, call cholqr on one matrix at a time when
  r > 256 on ROCm.
- **No CUDA-graph capture.** `cholesky_ex` cannot be captured in a HIP graph ("operation not permitted when stream is
  capturing"), so raw `torch.cuda.graph` capture and `torch.compile(mode="reduce-overhead")` do not work with cholqr
  on ROCm (the corresponding tests are skipped there). Plain `torch.compile` works.
- `verify=True` copies Q to the CPU and checks it there (rocBLAS/hipBLASLt are not covered by the certificate's
  assumptions; see [verify](verify.md)). The 3xTF32 compensation is applied on NVIDIA GPUs only.

## `method="auto"`

CholeskyQR costs ~45 small kernel launches, so a single small matrix takes about 1 ms eager and 0.3 ms or more compiled.
`torch.linalg.qr` takes 0.2–0.9 ms there. `"auto"` picks the faster one: on an A100 it is within a few percent of the
better of `torch.linalg.qr` and `"scqr3"` for every shape we measured (`benchmarks/results/a100_v03_2026-11-07.txt`).
On the Householder path, R is normalized to a non-negative diagonal, as in the CholeskyQR methods.

[Back to README](../README.md)
