# Pre-registration: what an orthogonality error does downstream (2026-11-12)

Written and committed before any run of `benchmarks/downstream_orth.py` (code, predictions and judge are committed
together). A smoke run (`--smoke`) prints timings only and writes outside the results directory. CPU only (Apple M4),
cholqr 0.5.0, torch 2.15.0, precision "highest". TF32 is emulated on the CPU (as in `benchmarks/verify_eval.py`).

## Questions

1. What does an orthogonality error $\delta = \|Q^T Q - I\|_F$ of 1e-4, 1e-3 (plain TF32) or 1e-2 (cqr2 gradual loss)
   do to (i) a randomized range finder / randomized SVD, (ii) PowerSGD-style compression with error feedback,
   (iii) subspace iteration? Is the default `tol = 1e-4` of `verify=True` sensible?
2. Residual $\|Y - QR\|_F / \|Y\|_F$ and subspace accuracy $\sin\theta_{\max}(\mathrm{range}\,Q, \mathrm{range}\,Y)$ of cholqr's methods.

## Argument behind the predictions (to be tested, not assumed)

With $Q = U (I + F)^{1/2}$, $U$ orthonormal (range unchanged, $Q^T Q = I + F$): $Q Q^T = U (I + F) U^T$, so
$\|A - Q Q^T A\|_F^2 = \|(I - U U^T) A\|_F^2 + \|F U^T A\|_F^2$ (orthogonal components): the extra error enters **in
quadrature** with the truncation error, i.e. at second order in $\delta$ when the truncation error is not small.
Singular values of $Q^T A = (I + F)^{1/2} U^T A$ change by a relative factor in $[\sqrt{1 - \|F\|_2}, \sqrt{1 + \|F\|_2}]$:
**first order**, $\le \|F\|_2 / 2$. Range errors are not controlled by orthogonality at all; for a QR,
$\sin\theta_{\max}(\mathrm{range}\,Q, \mathrm{range}\,Y) \le \|Y - QR\|_2 / \sigma_{\min}(Y)$ (any $Q$, $R$ with $R$ nonsingular).

## Experiments

**resid.** $m \times l \in \{1024 \times 64, 4096 \times 276\}$; spectra geometric, two-cluster, staircase (construction of
`benchmarks/device_sweep.py`); nominal cond $10^{2}, 10^{2.5}, \dots, 10^{8}$; 2 seeds; input rounded to FP32.
Methods: `scqr3`, `cqr2`, `rcqr`, `robust=True` (default fallback), `scqr3`/`cqr2` with `on_breakdown="ignore"`
(`_raw`), TF32-emulated `scqr3` (default and raw) and `cqr2` (raw), FP32 Householder baseline. Metrics: orthogonality,
relative residual vs the FP32 input, $\sin\theta$ vs range(fl(Y)) (`sin_fl`) and vs the exact range (`sin_ex`), and
$\|Y - QR\|_2/\sigma_{\min}(\mathrm{fl}(Y))$. **Broken** := orthogonality $> 0.1$ or non-finite.

**rsvd.** $A \in \mathbb{R}^{2048 \times 1024}$ with known spectra $\sigma_j = j^{-1}$, $j^{-2}$, $10^{-(j-1)/12}$, $10^{-(j-1)/6}$;
2 seeds; Gaussian sketch, $l = 64$, $q \in \{0, 1, 2\}$ power steps (FP64, then rounded to FP32 for the cholqr variants);
top $k = 48$ singular values. Q variants: exact (FP64 Householder), cholqr realistic variants (`scqr3`, `cqr2_raw`,
`tf32e_scqr3`, `tf32e_scqr3_raw`, `tf32e_cqr2_raw`, FP32 Householder), and injected on the exact basis with
$\delta \in \{10^{-4}, 10^{-3}, 10^{-2}\}$: `rand` (random symmetric $F$), `align` ($F = \delta w w^T$, $w$ the top left singular
vector of $U^T A$: worst case), `gen` (range-preserving part plus a range change $\|Z\|_2 = \delta/2$). Metrics: max relative
error of the top-48 singular values vs exact-Q and vs true; $\|A - QQ^TA\|_F/\|A\|_F$ and its excess over exact-Q.

**subiter.** Same $A$ (spectra $j^{-2}$, $10^{-(j-1)/12}$), $l = 64$, 10 steps $Q_{j+1} = \mathrm{qr}(A A^T Q_j)$; variants exact,
`scqr3` (FP32), `tf32e_scqr3_raw`, `tf32e_cqr2_raw`, injected `rand`/`gen` at every step. Metrics per step: orthogonality,
capture error $\|(I - P_{Q_j}) U_{48}\|_2$ of the true top-48 left singular subspace, deviation from the exact iterate.

**powersgd.** $G_t = L + N_t$ ($1024 \times 512$; $L$ rank 64 with $\sigma_j \propto j^{-1}$, $\|L\|_F = 1$; Gaussian $N_t$, $\|N_t\|_F = 0.5$),
rank $r \in \{4, 16\}$, 20 steps, 2 seeds, error feedback, warm-started $Q$. Orthogonalization of $P$: exact, cholqr
`cqr2`, `scqr3` (FP32), `tf32e_cqr2_raw`, injected `rand`/`gen` (all other arithmetic FP64). Metrics: per-step relative
compression error, cumulative error $\|\sum G_t - \sum \hat G_t\|_F/\|\sum G_t\|_F$, and a one-shot excess at each exact
step.

## Predictions and pass/fail criteria (judged by `downstream_orth.py --judge`)

- **R1** FP32 cholqr methods (scqr3, cqr2, rcqr, robust; default and raw), non-broken runs: max residual $\le 10^{-4}$
  and median $\le 10^{-5}$.
- **R2** Residual does not grow with cond: per method, median at cond $\ge 10^6$ $\le$ 10x median at cond $\le 10^3$.
- **R3** TF32-emulated non-broken runs: residual in $[10^{-5}, 10^{-2}]$.
- **R4a** FP32 cholqr non-broken: `sin_fl` $\le 10^{-7} \cdot$ cond in $\ge 98\%$ of runs.
- **R4b** (theorem check) `sin_fl` $\le \|Y - QR\|_2 / \sigma_{\min}$ in every run where the right side is $< 1$.
- **R5** Runs with orthogonality in $[10^{-3}, 0.1]$ (any method): `sin_fl` $\le$ orthogonality in $\ge 90\%$.
- **D1** Injected rand/align: singular-value error vs exact-Q $\le$ the multiplicative bound (theorem check); align attains
  $\ge 0.9$ of it.
- **D2** Injected rand/align: the Pythagoras identity holds within relative $10^{-6}$.
- **V1** Align, $\delta = 10^{-4}$: excess of $\|A - QQ^TA\|_F$ $\le 1\%$ for every case with relative tail $\ge 10^{-3}$, and
  singular-value error $\le 5 \cdot 10^{-5}$.
- **V2** Align, $\delta = 10^{-2}$: excess $\ge 10\%$ for at least one case with tail $\le 10^{-3}$, and $\le 1\%$ for every case
  with tail $\ge 0.1$.
- **D3** Realistic non-broken variants: singular-value error vs exact-Q $\le \max(\text{orth}, 10^{-6})$ in $\ge 90\%$.
- **S1** Subspace iteration: orthogonality does not accumulate: $\mathrm{orth}(Q_{10}) \le 3 \cdot \mathrm{median}_{j \ge 1}\,\mathrm{orth}(Q_j) + 10^{-12}$
  for every variant.
- **S2** Capture error at step 10 $\le \max(2 \times \text{exact}, \lambda)$ with $\lambda = \delta$ (injected), $10^{-4}$ (`scqr3`),
  $10^{-2}$ (TF32-emulated).
- **W1** (theorem check) one-shot excess $\le \sqrt{c^2 + \delta^2 \|U^T M\|_F^2/\|M\|_F^2} - c$.
- **W2** Range-preserving injected: $\max_t |\Delta\,\mathrm{err}_t| \le \delta$, and $\le 10^{-3}$ at $\delta = 10^{-2}$ (second
  order); cumulative error within 1% (relative) of exact.
- **W3** cholqr `cqr2`/`scqr3` FP32: $\max_t |\Delta\,\mathrm{err}_t| \le 10^{-5}$; `tf32e_cqr2_raw`: $\le 10^{-3}$.
- **W4** Range-changing injected: $\max_t |\Delta\,\mathrm{err}_t| \le \delta$.
- **TOL** (decision rule) "1e-4 is a sensible default tolerance for the target uses" iff V1, D1 and W2 pass.

Failures are reported as failures; nothing is re-registered after seeing results (any later analysis is labelled post
hoc).
