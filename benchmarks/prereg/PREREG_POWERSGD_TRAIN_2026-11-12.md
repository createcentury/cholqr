# Pre-registration: PowerSGD in real training, and a few device-sweep cells on a TPU (2026-11-12)

Written and committed before any run of `benchmarks/powersgd_train.py` and before any TPU run of
`benchmarks/xla_probe.py` / `benchmarks/device_sweep.py` in this round. Code, predictions and judge are committed
together. A smoke run prints timings only; any result seen before this commit would be listed in an addendum (none).

## A. PowerSGD in real training (`benchmarks/powersgd_train.py`)

**Question.** The note's PowerSGD experiment used synthetic gradients. Does the orthogonalization routine inside
PyTorch's PowerSGD DDP hook change the result of a real training run, and how large is the effect of a deliberately
degraded orthogonalization compared with seed-to-seed variation?

**Setup.** CIFAR-10 (binary version from the official site, read as raw bytes), a small ResNet (stem + 3 stages x 2
basic blocks, widths 32/64/128, BatchNorm, ~0.7M parameters), random crop (reflect pad 4) + horizontal flip.
`torch.distributed` with gloo on the CPU (Apple M4), 2 processes x 4 threads, 64 examples per process (global batch 128),
500 steps (~1.3 epochs). SGD, Nesterov momentum 0.9, weight decay 5e-4, lr 0.05 with 50 warm-up steps and linear decay
to 0. PowerSGD: `matrix_approximation_rank=4`, `start_powerSGD_iter=10`, `min_compression_rate=2`, error feedback,
warm start, `batch_tensors_with_same_shape=True` (so that cholqr's batched path is used). torch 2.15.0, cholqr 0.5.0,
precision "highest".

**Arms** (identical seeds, data order, augmentation, initialization and PowerSGD seed; only the orthogonalization of
P differs):
(a) `torch` - PyTorch's `_orthogonalize` (FP32 Householder QR, as rank > 2);
(b) `contrib` - `cholqr.contrib.powersgd.patch()`;
(c) `verify` - `cholesky_qr(P, verify=True)` (scqr3, FP64 certificate, tol 1e-4, default fallback) for every batch;
(d) `inj1e-3`, `inj1e-2` - $Q = U (I + F)^{1/2}$ with $U$ the FP64 Householder basis of P and random symmetric $F$,
$\|F\|_F = \delta$ (range unchanged; the "random F" construction of the note's Sec. 6.11).

**Seeds.** 0, 1, 2, 3 for every arm (20 runs), plus one rerun of (a) seed 0 (determinism check). Metrics: training
loss per step (mean over ranks), final training loss (mean of the last 50 steps), test loss and accuracy on the 10k
test set, median step time (steps 15-499), time in the orthogonalization per step, max orthogonality
$\|Q^TQ - I\|_F$ (FP64) over all calls, max parameter difference between ranks.

**Seed noise.** $S$ = sample standard deviation over the 4 seeds of arm (a). Paired difference $d_s$ = metric(arm, s)
- metric((a), s).

**Predictions (judge: `python benchmarks/powersgd_train.py --judge results.jsonl`).**
- **P0** The rerun of (a) seed 0 reproduces the per-step loss curve bit for bit (CPU gloo is deterministic).
- **P1** (b) is within seed noise of (a): $|\mathrm{mean}_s\, d_s| \le S$ and $\max_s |d_s| \le 3S$, for test accuracy and
  for final training loss.
- **P2** Same for (c).
- **P3** Same for (d) at $\delta = 10^{-3}$.
- **P4** Same for (d) at $\delta = 10^{-2}$. Reasoning: in Sec. 6.11 an orthogonality error $\delta$ moved the
  error-feedback trajectory like a gradient perturbation of relative size 0.1-6 $\delta$; minibatch gradient noise is
  of relative size O(1), so even $\delta = 10^{-2}$ should be invisible in the final metrics. Expected effect size:
  $|\mathrm{mean}_s\, d_s| \lesssim 0.5 S$.
- **P5** For each of (b), (c), (d): the paired loss-curve deviation (mean over the last 100 steps and over seeds of
  $|\mathrm{loss}(\text{arm}, s) - \mathrm{loss}((a), s)|$) is smaller than the across-seed deviation of (a) (mean over seed
  pairs). (Small differences grow in non-convex training, so the paired curves are expected to separate, but not to
  the across-seed level.)
- **P6a** Max orthogonality of (a), (b), (c) over all calls $\le 10^{-4}$. **P6b** Injected arms: max orthogonality within
  1% of $\delta$.
- **P7** Parameters are identical on both ranks at the end of every run.
- **P8** Median step time of (b) and (c) within 15% of (a) for every seed (on the CPU the orthogonalization of
  4 x n matrices is a small part of the step).

**Decision rule.** "On this setup, the choice among (a), (b), (c) does not change the training result" iff P1, P2
pass; "an orthogonality error up to 1e-2 is harmless here" iff P3, P4 pass. Scope: one small model, one dataset,
CPU, 500 steps; no claim beyond it.

## B. TPU (PyTorch/XLA): fix and a few cells

The earlier TPU v5e run (note Sec. 6.6) crashed in `benchmarks/device_sweep.py` (conversion of a non-scalar XLA tensor
to a Python scalar in the mirror's per-pass log). Plan: one spot TPU VM (v5e-1, else v6e-1), torch/torch_xla 2.8.
First a capability probe (`benchmarks/xla_probe.py`, precision highest and medium), then a debug print of the shapes
in the failing line (no results), a minimal fix in the benchmark (committed separately, explained), then the quick
grid (`--quick --draws 1`: $1024 \times 64$, 6 spectra, cond 1e2-1e10 in 2 decades, scqr3 + cqr2 = 60 runs) at
precision highest, high and medium (XLA: highest = FP32-accurate multi-pass bf16, high = 3-pass bf16, medium = XLA
default, 1-pass bf16). Library code unchanged.

- **T1** Mirror vs library bit-identity (`--sanity`, highest): 0 mismatches.
- **T2** Highest: `benchmarks/device_sweep_judge.py` on the quick file - every registered check that has data passes.
- **T3** Highest: $\varepsilon_{\mathrm{gram}}/u$ median (pass 2, scqr3) $\le 50$ (FP32 class).
- **T4** Medium (1-pass bf16): cholqr does not know the TPU's matmul precision, so its default call returns
  orthogonality $> 10^{-4}$ in more than half of the 60 runs (not caught by the breakdown flags).
- High: descriptive only (no prediction).
- Probe: descriptive (which ops work, fail or run on the CPU via PyTorch/XLA fallback).

Budget: USD 2 for both parts (TPU spot v5e ~USD 0.36/chip-hour in us-central1/us-east1; part A runs locally).

## Addendum 1 (2026-11-12, before looking at any TPU sweep output)

- **Bug diagnosis (debug print, no results).** The crash was not caused by `info`: on torch_xla 2.8, `info` from
  `cholesky_ex` is a 0-d tensor. `Tensor.amin()` / `amax()` *without* `dim` return the input unreduced on XLA
  (shape (64,) for a 64-vector; `.min()` and `.amin(dim=-1)` reduce correctly). The mirror logged
  `float(L.diagonal().amin())`, which raised. Fix (commit after the pre-registration): `amin(dim=-1)`. cholqr itself
  only uses `amin(dim=-1)`.
- **Extension.** The quick grid finished in about a minute per precision, so the full round-1 grid
  (PREREG_DEVSWEEP protocol: $m \in \{1024, 4096\} \times l \in \{64, 276\}$, 6 spectra, cond 1e2-1e10 in half decades,
  2 draws, scqr3 + cqr2 = 1632 runs) is added at precision highest, judged with `benchmarks/device_sweep_judge.py`
  (prediction: all registered checks pass, as on the other devices). Decided before opening the quick-grid files.
- **Smoke (part A) timings seen:** median step 0.31 s (torch, contrib), 0.39 s (verify), 0.33-0.36 s (injected), orth
  time per step 0.6 / 1.8 / 7.3 ms. P8 is not changed.
