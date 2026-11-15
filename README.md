# cholqr

Fast, compile-friendly **shifted CholeskyQR3** for batched tall-skinny matrices in PyTorch.
It is a drop-in for `torch.linalg.qr(Y, mode="reduced")[0]` when you only need an orthonormal basis of `range(Y)`,
for example in randomized SVD and sketching, low-rank optimizers such as randomized Muon, PowerSGD-style gradient
compression, or subspace iteration.

**The problem.** Orthonormalizing many tall-skinny matrices is a common inner step in those methods. Householder QR
(`torch.linalg.qr`) is accurate but slow on GPUs for large batches. CholeskyQR needs only a few matrix products and
small Cholesky factorizations, so it is much faster, but it can lose orthogonality or break down when the input is
ill-conditioned, especially in FP32. cholqr implements the shifted CholeskyQR3 variant, detects most breakdowns at run
time, and can certify its output with an FP64 check (`verify=True`).

**Use it** for batches or larger shapes on a GPU when you need `Q` (an orthonormal basis of `range(Y)`).
**Prefer `torch.linalg.qr`** for a single small matrix (`method="auto"` does this for you), and use `robust=True` or
`verify=True` if the inputs may be very ill-conditioned. How far each claim can be trusted is summarized in
[How much to trust this](#how-much-to-trust-this); the analysis is in the accompanying
[technical note](https://doi.org/10.5281/zenodo.24182849).

## Install

```bash
pip install cholqr        # or: pip install git+https://github.com/createcentury/cholqr
uv add cholqr             # in a uv project (or: uv pip install cholqr)
```

cholqr depends only on PyTorch (>= 2.1). If you need a specific CUDA or ROCm build of PyTorch, install that first.

## Quick start

```python
import torch
from cholqr import cholesky_qr

Y = torch.randn(40, 768, 210, device="cuda")       # batch of tall-skinny matrices
Q = cholesky_qr(Y)                                 # shifted CholeskyQR3 (default, safe)
Q = cholesky_qr(Y, method="auto")                  # when shapes vary: single small matrices go to Householder
Q = cholesky_qr(Y, method="cqr2")                  # CholeskyQR2: ~1/3 cheaper, only for cond(Y) <~ 1e3
Q = cholesky_qr(Y, robust=True)                    # FP64 first pass: cond(Y) up to >= 1e8
Q, R = cholesky_qr(Y, return_r=True)               # also R (upper triangular), Y = Q R
fast = torch.compile(cholesky_qr, fullgraph=True)  # single graph, no host synchronization (no breakdown check)
```

**Speed vs `torch.linalg.qr`** (default call, FP32, precision "highest", median of 50 calls, torch 2.8 / CUDA 12.8):

| GPU | batch 40 x 768 x 210 | batch 64 x 1024 x 32 | single 1 x 1024 x 64 |
|---|---|---|---|
| H100 80GB | **14.5x** faster | **12.4x** | 0.43x (slower) |
| RTX 4090 | **12.0x** | **5.7x** | 0.21x |
| RTX 5090 | **16.9x** | **19.6x** | 0.67x |
| RTX PRO 6000 Blackwell | **18.5x** | **20.1x** | 0.69x |

cholqr pays off for batches and larger shapes; for one small matrix `torch.linalg.qr` is faster, and `method="auto"`
dispatches it there automatically. The breakdown check adds 0–7% ([details](https://github.com/createcentury/cholqr/blob/main/docs/devices.md)).

> **TF32 warning.** Many training scripts set `torch.set_float32_matmul_precision("high")` (TF32). With plain TF32
> products, CholeskyQR does not fail, but orthogonality silently degrades to ~1e-3 and the breakdown check cannot see
> it. cholqr 0.5 compensates (3xTF32), which removes most but not all of the loss and is slower than full FP32. If you
> need orthogonality below 1e-4, wrap the call in `torch.set_float32_matmul_precision("highest")` or pass `verify=True`
> ([details](https://github.com/createcentury/cholqr/blob/main/docs/accuracy.md)).

**Breakdown detection and fallback (0.5).** CholeskyQR breaks down when the input is too ill-conditioned for the
chosen method. In eager mode, `cholqr` detects this and recomputes only the affected matrices: first with an FP64 first
pass, then with Householder QR. The check reuses the small r x r Cholesky factors the passes already compute (no extra
Gram product), but costs one host synchronization per call. Choose the behaviour with `on_breakdown`: `"fallback"`
(default), `"warn"`, `"raise"` or `"ignore"` (the 0.4 behaviour, no synchronization). Inside `torch.compile` the check
is skipped. This detector is **empirical** (calibrated and validated, not proved; under TF32 it misses most degraded
runs). For a guarantee use `verify=True` ([Certified orthogonality](https://github.com/createcentury/cholqr/blob/main/docs/verify.md)).
How the threshold was chosen: [Accuracy](https://github.com/createcentury/cholqr/blob/main/docs/accuracy.md).

| method | passes | reliable range in FP32 (tested) | use when |
|---|---|---|---|
| `"scqr3"` (default) | 3 | cond(Y) ≲ 1e6 | general use |
| `"scqr3"`, `robust=True` | 3 (first in FP64) | cond(Y) up to ≥ 1e8 | ill-conditioned inputs, data-center GPUs |
| `"cqr2"` | 2 | cond(Y) ≲ 1e3 recommended; first breakdown measured from 3e3 (clustered spectra) to 1e4–1e5 (geometric spectra) | known well-conditioned inputs, e.g. sketches without power iteration |
| `"auto"` | — | any | **recommended when shapes vary**: Householder (`torch.linalg.qr`) for a single matrix with r ≤ 64, otherwise `"scqr3"` |
| `"rcqr"` | sketch + 1 | cond(Y) up to ≥ 1e8 (orth ≈ 7e-6) | single large matrices; **slow when batched** (its preconditioner is a batched Householder QR) |

Beyond its range, a method breaks down abruptly. Since 0.5 this is detected and repaired in eager mode (see above),
but every repaired call pays for a second factorization. Pick the method for the conditioning you expect. Under
`torch.compile` there is no check, so use `robust=True` there if the conditioning is unknown, or check outside the graph:

```python
qr = torch.compile(cholesky_qr)                  # in the graph: no check, no fallback, no host sync
Q = qr(Y)
if step % 100 == 0:                              # occasionally, outside the compiled region
    if (cholqr.orthogonality_bound(Q) > 1e-4).any():
        Q = cholesky_qr(Y, verify=True)          # eager: certified, with fallback
```

Measured on an RTX 4090: checking every 100 steps adds about 2% per step (every 10 steps 16-19%, every step
2.6-2.8x, because FP64 is slow on consumer GPUs); no recompilations or graph breaks.

**Example (randomized low-rank Muon on nanoGPT):** the measured cond(Y) is at most 70 without power iteration (h=0), so `"cqr2"` is safe there. With one power iteration (h=1) it reaches 4.5×10⁴, which needs `"scqr3"`.

## Which settings to use

| goal | call | what you get / cost |
|---|---|---|
| default, most uses | `cholesky_qr(Y)` | ~1e-5 orthogonality for cond(Y) ≤ 1e6 (geometric spectra, measured); breakdowns detected and repaired in eager mode |
| speed, inputs known to be well conditioned (cond ≲ 1e3) | `method="cqr2"` | about 1/3 cheaper |
| one small matrix (r ≤ 64) | `method="auto"` | dispatches to `torch.linalg.qr`, which is faster there |
| ill-conditioned inputs | `robust=True` | FP64 first pass; not on Apple MPS |
| a stated orthogonality guarantee | `verify=True` (optionally `tol=`) | Q returned only if an FP64 bound proves ‖QᵀQ − I‖_F ≤ tol; cost +12–29% on an A100, up to +216% on an RTX 4090 (FP64 at 1/64 speed) |
| TF32 enabled globally | `torch.set_float32_matmul_precision("highest")` around the call, or `verify=True` | see the TF32 warning above |
| `torch.compile`, `torch.func`, `torch.jit.trace` | default call inside; `cholqr.orthogonality_bound(Q)` outside | the data-dependent check and fallback cannot run in a graph |
| float16 / bfloat16 / float8 output | `out_dtype=torch.float32`, or give `tol` | |
| no host synchronization at all | `on_breakdown="ignore"` | no check, no repair |
| multi-process training (DDP, FSDP, ZeRO) | not `on_breakdown="raise"`; make any fallback decision collective | see [Distributed training](https://github.com/createcentury/cholqr/blob/main/docs/powersgd.md#distributed-training-keep-decisions-identical-on-every-rank) |

## How much to trust this

| level | what | where |
|---|---|---|
| **Proved** (Lean 4) | the exact-arithmetic core of the rounding analysis: shifted first pass, per-pass orthogonality bound, contraction into the next pass, algebra of the `verify=True` certificate | [lean/](https://github.com/createcentury/cholqr/blob/main/lean/README.md) |
| **Proved on paper, used as hypotheses** | floating-point facts (rounding of the Gram product, the Cholesky factor, the normalisation) | [technical note](https://doi.org/10.5281/zenodo.24182849) |
| **Certified at run time** | `verify=True`: an FP64 bound on `‖QᵀQ − I‖_F` for the returned tensor. Rigorous if the note's assumptions (H0)–(H2) hold; (H2), on the FP64 product actually executed, is assumed, not checked | [verify.md](https://github.com/createcentury/cholqr/blob/main/docs/verify.md) |
| **Measured** (pre-registered) | where FP32 `scqr3` breaks, across CPUs and GPUs; speed-ups; TF32 behaviour | [benchmarks/results](https://github.com/createcentury/cholqr/tree/main/benchmarks/results), the note |
| **Empirical, not proved** | the run-time detector's threshold and the rule for when an FP32 Cholesky fails | [accuracy.md](https://github.com/createcentury/cholqr/blob/main/docs/accuracy.md) |

There is **no a-priori guarantee** that a given FP32 input will not break down at sizes such as 4096 x 276; the note
explains why one that holds on all hardware is out of reach. What you get instead is detection (empirical) and
certification after the fact (`verify=True`).

## Paper

> S. Fukuda. *Shifted CholeskyQR in FP32: measured breakdown, an a-posteriori bound, and an FP64 orthogonality check in
> cholqr.* Technical note, v1, 2026. doi:[10.5281/zenodo.24182849](https://doi.org/10.5281/zenodo.24182849)

## Documentation

| page | contents |
|---|---|
| [How it works](https://github.com/createcentury/cholqr/blob/main/docs/how-it-works.md) | why CholeskyQR, the shifted CholeskyQR3 algorithm, compile-friendly design |
| [Accuracy and range of validity](https://github.com/createcentury/cholqr/blob/main/docs/accuracy.md) | where each method breaks down, breakdown detection and fallback, TF32 |
| [Certified orthogonality](https://github.com/createcentury/cholqr/blob/main/docs/verify.md) | `verify=True`: an FP64 a-posteriori certificate, assumptions, cost |
| [Devices: proved vs measured](https://github.com/createcentury/cholqr/blob/main/docs/devices.md) | device sweep (Apple, NVIDIA), `python -m cholqr` to check your device |
| [Compatibility](https://github.com/createcentury/cholqr/blob/main/docs/compatibility.md) | Python / PyTorch versions, execution modes, `method="auto"` |
| [PowerSGD drop-in](https://github.com/createcentury/cholqr/blob/main/docs/powersgd.md) | `cholqr.contrib.powersgd` for DDP gradient compression |
| [Distributed training (DDP / FSDP / ZeRO)](https://github.com/createcentury/cholqr/blob/main/docs/powersgd.md#distributed-training-keep-decisions-identical-on-every-rank) | keep fallback decisions identical on every rank: no `on_breakdown="raise"`, all-reduce the check with `MAX` |

## Formally verified core

The exact-arithmetic core of the rounding analysis (the shifted first pass, the "indefinite part" lemma, the one-pass
orthogonality bound, the contraction into the next pass, and the algebra of the `verify=True` certificate) is proved in Lean 4 with Mathlib, with no `sorry`
and only the standard axioms. The floating-point facts are hypotheses, proved on paper. See
[lean/README.md](https://github.com/createcentury/cholqr/blob/main/lean/README.md) for exactly what is and is not formalized.

## Citation

If you use cholqr, please cite the software ([CITATION.cff](https://github.com/createcentury/cholqr/blob/main/CITATION.cff)):

> S. Fukuda. *cholqr: fast, compile-friendly shifted CholeskyQR3 for batched tall-skinny matrices in PyTorch.*
> Software, version 0.5.1, 2026. doi:[10.5281/zenodo.24157501](https://doi.org/10.5281/zenodo.24157501)
> (all versions: [10.5281/zenodo.24144958](https://doi.org/10.5281/zenodo.24144958)),

the technical note (see [Paper](#paper)), and the algorithm:

> T. Fukaya, R. Kannan, Y. Nakatsukasa, Y. Yamamoto, Y. Yanagisawa. *Shifted Cholesky QR for computing the QR
> factorization of ill-conditioned matrices.* SIAM J. Sci. Comput. 42(1):A477–A503, 2020.

> Y. Yamamoto, Y. Nakatsukasa, Y. Yanagisawa, T. Fukaya. *Roundoff error analysis of the CholeskyQR2 algorithm.*
> Electron. Trans. Numer. Anal. 44:306–326, 2015.

## License

MIT
