# Pre-registration: a size-dependent last-pass threshold tau(l), tested at fresh sizes (2026-11-12)

Written and committed before any run of `benchmarks/thresh_size.py` (rule `benchmarks/thresh_size_rule.py`, fitting
script `benchmarks/thresh_size_fit.py`, runner and judge `benchmarks/thresh_size_judge.py` are committed together). A
smoke run (`--smoke`) prints timings only and writes nothing. CPU only (Apple M4), torch 2.15.0, cholqr 0.5.1 (working
tree; the 0.5.1 changes do not touch Q in FP32), precision "highest", FP32, $u = 2^{-24}$.

## Why

cholqr 0.5 flags a matrix when the smallest diagonal entry of the last pass's Cholesky factor is below
$0.195 = \sqrt{64u/10^{-4}}$. It was validated at $l \le 276$ (CPU 155/155 degraded runs caught, 4/1931 false alarms; Apple
GPU 56/56, 0/760). At $l = 512$ / $1024$ (`PREREG_THRESH_WIDE_2026-11-12.md`, results `benchmarks/results/thresh_wide_2026-11/`)
it missed 1/69 and 6/84 degraded runs (orthogonality $1.05$–$1.8\times10^{-4}$), 0 false alarms, because the constant
$c$ in orthogonality $\approx c\,u/L_{\min}^2$ grows with $l$.

## The function (frozen; `benchmarks/thresh_size_rule.py`)

$$
\tau(l) = \min\Bigl(0.75,\ \max\bigl(0.1953,\ \sqrt{C(l)\,u/10^{-4}}\bigr)\Bigr), \qquad C(l) = 1.01\, l^{0.868},
$$

flag if $L_{\min,\mathrm{last}} < \tau(l)$ ($l$ = number of columns) or any pass failed (`info > 0` or a non-finite factor).
Values: $\tau(64) = 0.195$ (floor), $\tau(276) = 0.281$, $\tau(384) = 0.325$, $\tau(512) = 0.368$, $\tau(768) = 0.439$,
$\tau(1024) = 0.497$, $\tau(1536) = 0.593$; the cap 0.75 binds from $l \approx 2650$ (beyond every tested size).

## Fitting rule (applied only to existing data; `benchmarks/thresh_size_fit.py`, deterministic)

Data: the CPU, spectral-norm data with Lmin of the last pass: QRROUND3 (the 0.195 rule's CPU validation data;
$m \in \{1024, 4096\}$, $l \in \{64, 276\}$, orders seq/lib/pw8, TF32-emulated runs excluded) and THRESH_WIDE
($l \in \{512, 1024\}$, $m = 8l$). The Apple GPU CERT data (Frobenius norm) enter only constraint (b).

1. Runs without a failed factorization; $c = \mathrm{orth}\cdot L_{\min,\mathrm{last}}^2/u$ over runs with $L_{\min,\mathrm{last}} < 0.6$
   and $10^{-6} < \mathrm{orth} \le 10^{-3}$ (first-order regime; saturated runs with orth near 1 give meaningless $c$, up to 3554).
2. $q_l$ = max of $c$ at each $l$: 31.2, 68.1, 99.6, 431.5 at $l$ = 64, 276, 512, 1024.
3. Least squares $\ln q_l = \ln A + p \ln l$: $A = 0.6725$, $p = 0.8678$.
4. $C(l) = M A l^p$, margin $M$ = largest of $\{2, 1.5, 1.25, 1\}$ such that in sample (a) no degraded run is missed at any
   fitting size and (b) the false alarms on the 0.195 rule's validation data are not above those of 0.195. $M = 2$ violates
   (b) ($\tau(276) = 0.324$ adds one false alarm on the CPU and one on the Apple GPU); $M = 1.5$ satisfies both;
   $M = 1$ violates (a) (2 misses at $l = 1024$). Hence $M A = 1.0088 \to 1.01$, $p \to 0.868$ (3 significant digits).
5. Floor 0.1953 (never flag less than the validated rule), cap 0.75 (non-binding here).

Seen in sample with the frozen rule: 0 misses at $l$ = 64, 276, 512, 1024; false alarms unchanged at $l \le 276$
(CPU 4/1931, Apple GPU 0/760) and 0 at 512 / 1024. These are fit diagnostics, not evidence.

## Test (fresh sizes, fresh inputs; `benchmarks/thresh_size.py`)

$(m, l) \in \{(3072, 384), (6144, 768), (12288, 1536)\}$ (aspect ratio $m = 8l$ as in THRESH_WIDE; 384 and 768 interpolate,
1536 extrapolates); spectra geometric, power-law, spike, two-cluster, linear, staircase (`benchmarks/device_sweep.py`
construction); nominal cond $10^2, 10^{2.5}, \dots, 10^{10}$; draws 1–3 with the new seed key prefix `TS-`; methods `scqr3`,
`cqr2`; Gram orders `lib` (cholqr's `_gram`) and `pw8` (pairwise over 8-row blocks: balanced trees over aligned subtrees of
64 blocks, then pairwise over the subtree sums with an odd element carried; equal to THRESH_WIDE's tree for power-of-two
block counts). 918 inputs, 3672 runs. In `lib` mode the mirror's $Q$ is compared bit for bit with `cholqr.core._cqr`.
Compute: Mac CPU only (the data the function was fitted on are Mac CPU), sharded, 2 threads per shard, total load
kept below 12.

## Protocol (as in the note's Sec. 7 and THRESH_WIDE)

Denominators: runs in which no factorization failed (flagged through `info` anyway; excluded). Degraded: final
$\|Q^TQ - I\|_2 > 10^{-4}$ (FP64). Good: $\le 10^{-5}$. FN: degraded, not flagged by $\tau(l)$. FA: good, flagged by $\tau(l)$.
Judged per size, pooled over spectra, methods and orders.

## Registered criteria (gate for a library change)

- **T1** (per fresh size): $\ge 99\%$ of degraded runs flagged, i.e. FN $\le 1\%$ (not testable with fewer than 20 degraded runs;
  "not testable" does not pass the gate).
- **T2** (per fresh size): FA $\le 2\%$ of good runs.
- **T3** (secondary, reported, not part of the gate): FA $\le 0.5\%$ (the protocol's C2 band).
- **N1** (non-regression on the 0.195 rule's original validation data, CPU QRROUND3 and Apple GPU CERT): every degraded
  run flagged by 0.195 is flagged by $\tau$, and the false alarms are not above those of 0.195. In sample (these data
  shaped $M$), so this is a consistency check, known to hold before the run.

Gate: T1 and T2 PASS at all three fresh sizes and N1 PASS. If the gate passes, a library change is prepared in a
separate, clearly labelled local commit (not merged into the 0.5.1 changes; CHANGELOG under "Unreleased (after 0.5.1)").
If it fails, nothing in cholqr changes and the result is reported.

## Predictions

- **P1** $l = 384$: T1 PASS with FN 0; FA $\le 2$.
- **P2** $l = 768$: T1 PASS with FN 0; FA $\le 3$.
- **P3** $l = 1536$: T1 and T2 PASS (low confidence: extrapolation; a miss among ~80 degraded runs fails T1).
- **P4** max $c$ (first-order regime, as in step 1) at each fresh size within $[0.5, 2] \times A l^p$ (in sample the ratio
  was 0.66–1.57).
- **P5** the old rule 0.195 misses at least one degraded run at $l = 768$ and at $l = 1536$.
- **S0** (sanity, not a prediction): mirror bit-identical to `cholqr.core._cqr` in every `lib` run; 0.195 flag agrees.

## Limitations stated in advance

One machine (Apple M4 CPU, Accelerate); the size dependence may differ with BLAS blocking and on GPUs. The function
depends on $l$ only; the test keeps $m = 8l$ (the fit data had $m/l$ from 3.7 to 64 at $l \le 276$ and 8 above).
