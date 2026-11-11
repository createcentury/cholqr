# Pre-registration: forming error of explicit inverse + GEMM vs triangular solve (2026-11-12)

Written and committed before any run of `benchmarks/forming_error.py` (code, predictions and judge `--judge` are
committed together). A smoke run (`--smoke`) prints timings only and writes nothing. CPU only (Apple M4), cholqr 0.5.0,
torch 2.15.0, precision "highest", FP32, $u = 2^{-24}$.

## Question

cholqr forms $Q = X L^{-T}$ by computing the $l \times l$ inverse $L^{-1}$ (`solve_triangular(L, I)`) and one GEMM
(stated reason in `docs/how-it-works.md`: "This replaces a slow triangular solve over m rows"). How large is the forming
error $\eta = \|\tilde Q - X L^{-T}\|_2$ compared with a row-wise triangular solve
(`solve_triangular(L^T, X, upper=True, left=False)`), and does the choice change the final orthogonality?
Known (seen): in the note's full-accounting sweep, median $\eta \approx 300\,u$ over all successful passes (dominated by the
first pass); second- and third-pass $\eta$ is about $1\,u$ in the examples seen.

## Design

Shapes $1024 \times 64$, $4096 \times 276$; spectra geometric, two-cluster, staircase; nominal cond $10^2 \dots 10^8$ in half
decades; draws 1, 2 (prefix `FE-`); methods `scqr3`, `cqr2`; variants `inv` (cholqr; checked bit-identical to the library)
and `trsm`. Each variant runs its own passes, so pass 1 is exactly paired. Reference $X L^{-T}$ in FP64 from the FP32 $X$
and $L$; a second FP64 reference (FP64 triangular solve) bounds its error.

## Predictions (judged by `forming_error.py --judge`)

- **F1** Pass 1 (paired), ratio $\eta_{\mathrm{inv}}/\eta_{\mathrm{trsm}}$: median in $[1, 10]$, max $\le 100$ (explicit inverse
  not more accurate; same order, since both forward errors scale with the conditioning of $L$).
- **F2** Passes $\ge 2$: median $\eta/u \le 5$ for both variants, ratio of medians in $[0.5, 2]$.
- **F3** Final $\|Q^TQ - I\|_2$, inputs non-broken in both variants: median ratio inv/trsm in $[0.5, 2]$; the breakdown
  status (any failed factorization) is the same for both variants in $\ge 95\%$ of inputs.
- **F4** Reference: $\|\mathrm{ref} - \mathrm{ref}_2\|_2 \le 0.01\,\eta$ in $\ge 99\%$ of successful passes.
- **F5** Theorem D(c) (with measured $e$, $\eta$): 0 violations (relative tolerance $10^{-9}$), both variants.

Descriptive (not judged): $\eta/u$ distributions per pass, share of the $\eta$ terms in the D(c) bound in the last pass.
