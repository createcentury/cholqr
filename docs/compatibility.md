# Compatibility

| | tested | notes |
|---|---|---|
| Python | 3.9-3.15 | CI |
| PyTorch | 2.1 to latest; nightly in CI (allowed to fail) | on torch < 2.4, compiling `method="rcqr"` fails or hangs in Inductor |
| OS / CPU | Linux x86-64 and ARM64, macOS (Apple silicon), Windows | CI |
| NumPy | 1.x and 2.x | only the PowerSGD DDP test uses it |
| NVIDIA GPUs | Turing, Ampere, Ada, Hopper, Blackwell (T4, A40, L4, RTX 4090/5090, RTX PRO 6000, H100) | [device table](devices.md) |
| Apple GPUs | MPS | no FP64: the fallback uses Householder directly |
| Input dtypes | float64, float32, bfloat16, float16, float8_e4m3fn | computation is FP32 (FP64 first pass with `robust=True`) |
| Matmul precision | `highest` (full accuracy); `high` (TF32: partly compensated, see [Accuracy](accuracy.md)) | `medium` may use bfloat16 products on some backends: not compensated |
| Execution modes | eager, `torch.compile` (inductor), `torch.func.vmap`, `torch.export` (with `on_breakdown="ignore"`), autograd incl. double backward, `inference_mode`, deterministic algorithms (CPU), threads, spawned processes | the breakdown check is skipped under `torch.compile`, `vmap` and tracing; `verify=True` raises there |
| CUDA execution modes | deterministic mode, CUDA graphs, reduce-overhead compile, second GPU (2x RTX 4090, torch 2.8: full test suite 124 passed) | tests |
| Not yet verified | Volta GPUs (V100), A100, AMD ROCm, Intel XPU, TPU (PyTorch/XLA: sanity only), FSDP/DTensor inputs | reports welcome |
| Not supported | complex dtypes (raise `TypeError`), TorchScript | |

## `method="auto"`

CholeskyQR costs ~45 small kernel launches, so a single small matrix takes about 1 ms eager and 0.3 ms or more compiled.
`torch.linalg.qr` takes 0.2–0.9 ms there. `"auto"` picks the faster one: on an A100 it is within a few percent of the
better of `torch.linalg.qr` and `"scqr3"` for every shape we measured (`benchmarks/results/a100_v03_2026-11-07.txt`).
On the Householder path, R is normalized to a non-negative diagonal, as in the CholeskyQR methods.

[Back to README](../README.md)
