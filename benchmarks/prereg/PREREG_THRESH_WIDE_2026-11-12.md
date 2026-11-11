# Pre-registration: the last-pass threshold 0.195 at wider matrices (2026-11-12)

Written and committed before any run of `benchmarks/thresh_wide.py` (code, predictions and judge
`benchmarks/thresh_wide_judge.py` are committed together). A smoke run (`--smoke`) prints timings only and writes
nothing. CPU only (Apple M4), cholqr 0.5.0 (`_BREAKDOWN_LMIN = sqrt(64 u / 1e-4) = 0.195`, $u = 2^{-24}$), torch 2.15.0,
precision "highest", FP32.

## Question

The rule "flag if the smallest diagonal entry of the last pass's Cholesky factor is below 0.195" was derived from
orthogonality $\approx c\,u / L_{\min}^2$ with $c$ calibrated at $l \in \{64, 276\}$ ($c = 18$–33, safety factor 2 gives 64)
and validated at $l \in \{64, 276\}$ (rounding note, Sec. 7: CPU 155/155 degraded runs caught, 4/1931 false alarms;
Apple GPU 56/56, 0/760). Does it still hold at $l = 512$ ($m = 4096$) and $l = 1024$ ($m = 8192$)?

## What is already known (seen, motivates the predictions)

From the note's full-accounting sweep data ($m \in \{1024, 4096\}$, $l \in \{64, 276\}$; seen), the constant
$c = \|Q^TQ - I\|_2 \, L_{\min,\mathrm{last}}^2 / u$ over non-broken runs with $L_{\min,\mathrm{last}} < 0.6$ already grows with $l$:
maximum 19–31 at $l = 64$, 73–133 at $l = 276$ (median for `cqr2` 8–9 at $l = 64$, 32–41 at $l = 276$). It already exceeds the
safety value 64 at $l = 276$, without a missed degraded run, because few runs have $L_{\min,\mathrm{last}}$ just above 0.195
(1–5 per cell of ~160 `cqr2` runs in $[0.195, 0.3)$). Mechanism (Theorem D(a)): orthogonality $\le (s + e)\|L^{-1}\|_2^2$,
with $\|L^{-1}\|_2^2 L_{\min}^2 \approx 2$ in those runs and $e$ (Gram error) growing with $m$ and $l$. A run is a false negative
iff it is degraded and $0.195 \le L_{\min,\mathrm{last}} < \sqrt{c\,u/10^{-4}}$, a band that widens as $c$ grows.

## Design

$(m, l) \in \{(4096, 512), (8192, 1024)\}$; spectra geometric, power-law, spike, two-cluster, linear, staircase (construction
of `benchmarks/device_sweep.py`); nominal cond $10^2, 10^{2.5}, \dots, 10^{10}$; draws 1–3 (seed key prefix `TW-`, new
draws); methods `scqr3`, `cqr2`; Gram orders `lib` (cholqr's `_gram`) and `pw8` (pairwise over 8-row blocks). 612 inputs,
2448 runs. Instrumented copy of cholqr's passes; in `lib` mode its $Q$ is compared bit for bit with `cholqr.core._cqr`.

## Protocol (as in the note's Sec. 7)

Denominators: runs in which no factorization failed (any pass with `info > 0` or a non-finite factor; these are flagged
through `info` in any case). Degraded: final $\|Q^T Q - I\|_2 > 10^{-4}$ (FP64). Good: $\le 10^{-5}$. Flag:
$L_{\min,\mathrm{last}} < 0.195$. FN: degraded and not flagged. FP: good and flagged. Judged per $l$, pooled over spectra,
methods and orders.

- **C1** (as in Sec. 7): $\ge 99\%$ of degraded runs flagged (not testable if fewer than 20 degraded runs).
- **C2** (as in Sec. 7): $\le 0.5\%$ of good runs flagged.

## Predictions

- **W1** $l = 512$: C1 PASS, FN in $[0, 2]$ (low confidence).
- **W2** $l = 1024$: C1 FAIL, FN in $[1, 15]$ (expected $c$ median ~100, so a band of roughly $[0.195, 0.25]$ for typical runs
  and up to ~0.4 for the tail).
- **W3** C2 PASS at both sizes, FP $\le 3$ at each.
- **W4** 90th percentile of $c$ (non-broken runs, $L_{\min,\mathrm{last}} < 0.6$, orthogonality $> 10^{-6}$) is $\ge 64$ at
  $l = 1024$ and larger at $l = 1024$ than at $l = 512$.
- **W5** Every FN has $L_{\min,\mathrm{last}} \in [0.195, 0.6)$.
- **S0** (sanity, not a prediction): the mirror is bit-identical to `cholqr.core._cqr` in every `lib` run and its flag
  agrees with the library's.

Consequence: if C1 fails at $l = 1024$, the note states the threshold as validated for $l \le 276$ (or $\le 512$) only and
reports the observed $c(l)$; a size-dependent threshold $\sqrt{c(l)\,u/\mathrm{tol}}$ would need its own registered test.
Nothing in cholqr changes on the basis of this run alone.
