# Lean formalization: exact-arithmetic core of the CholeskyQR rounding analysis

Lean 4 (`leanprover/lean4:v4.35.0`) + Mathlib (`v4.35.0`, pinned in `lake-manifest.json`). No `sorry`; every theorem
depends only on the standard axioms `[propext, Classical.choice, Quot.sound]` (check: `lake env lean Check.lean`).

Build:
```bash
cd lean && lake exe cache get && lake build
```

## What is proved (real matrices, exact arithmetic)
Notation: `‖x‖² = x ⬝ᵥ x`; `L` invertible with `L Lᵀ` as stated; `Q = X L⁻ᵀ`.

| statement | Lean name | file |
|---|---|---|
| **T0** (shifted first pass): with `L Lᵀ = G + s I`, `G = YᵀY`, `Q₁ = Y L⁻ᵀ`: `Q₁ᵀQ₁ = L⁻¹ G L⁻ᵀ` | `gram_Q1` | `CholQR/ShiftedPass.lean` |
| T0: an eigenpair `(λ, v)` of `G` gives the eigenpair `(λ/(λ+s), Lᵀv)` of `Q₁ᵀQ₁`, `Lᵀv ≠ 0` | `T0_eigvec`, `T0_eigvec_ne_zero` | same |
| T0: `L⁻ᵀ (L⁻¹ G L⁻ᵀ) Lᵀ = (G + s I)⁻¹ G` and `(G + s I)⁻¹ G v = (λ/(λ+s)) v` | `T0_similar`, `T0_resolvent_eig` | same |
| **Lemma I** (indefinite part): `Ĝ = BᵀB + A`, `‖Bx‖² ≥ c‖x‖²`, `|xᵀAx| ≤ a‖x‖²` ⟹ `xᵀ(Ĝ + sI)x ≥ (c + s − a)‖x‖²`; if `A` is symmetric and `c + s > a`, `Ĝ + sI` is positive definite | `lemmaI_quadratic`, `lemmaI_posDef` | same |
| **Theorem D (a)**: `L Lᵀ = XᵀX + sI + E` ⟹ `QᵀQ − I = −L⁻¹(sI + E)L⁻ᵀ`; quadratic form; `|yᵀ(QᵀQ − I)y| ≤ (s + e)‖L⁻ᵀy‖²` if `|zᵀEz| ≤ e‖z‖²` | `D_a_identity`, `D_a_quad`, `D_a_bound` | `CholQR/TheoremD.lean` |
| **Theorem D (b)**: if `‖Xz‖² ≥ c‖z‖²` and `c + s − e > 0`: `zᵀ(LLᵀ)z ≥ (c + s − e)‖z‖²`, `‖L⁻ᵀy‖² ≤ ‖y‖²/(c + s − e)`, `|yᵀ(QᵀQ − I)y| ≤ (s + e)/(c + s − e)‖y‖²` | `D_b_LLt`, `D_b_inv`, `D_b_bound` | same |
| **Theorem D (c)** (forming error): if `|yᵀ(QᵀQ − I)y| ≤ d‖y‖²` and `‖Δz‖² ≤ η²‖z‖²`, then for `Q̃ = Q + Δ`: `|yᵀ(Q̃ᵀQ̃ − I)y| ≤ (d + 2η√(1 + d) + η²)‖y‖²` | `D_c_bound` (with `cauchy_schwarz_dot`) | same |
| **D1** (contraction into the next pass): if `|yᵀ(Q̃ᵀQ̃ − I)y| ≤ d̄‖y‖²` (`0 ≤ d̄ < 1`), computed scales `d̂ⱼ = ‖q̃ⱼ‖(1 + θⱼ)` with `|θⱼ| ≤ ε < 1`, and `abs(X'ᵢⱼ − Z'ᵢⱼ) ≤ u abs(Z'ᵢⱼ)` for `Z'ᵢⱼ = Q̃ᵢⱼ/d̂ⱼ`, then `1 − d̄ ≤ ‖q̃ⱼ‖² ≤ 1 + d̄`, `‖Z'y‖ ≥ √((1−d̄)/(1+d̄))/(1+ε) ‖y‖`, `‖(Z' − X')y‖ ≤ √l u/(1−ε) ‖y‖`, hence `‖X'y‖ ≥ (√((1−d̄)/(1+d̄))/(1+ε) − √l u/(1−ε)) ‖y‖`; for `ε ≤ 1/2` this gives the note's `‖X'y‖ ≥ ((1−ε)√((1−d̄)/(1+d̄)) − 2√l u) ‖y‖` (`ε = mu`) | `col_sq_bounds`, `scale_facts`, `D1_scaled`, `D1_delta`, `D1_sharp`, `D1_paper` | `CholQR/D1.lean` |
| **Certificate** (`verify=True`): if `abs(Ĝᵢⱼ − (QᵀQ)ᵢⱼ) ≤ γ Σₖ abs(Qₖᵢ) abs(Qₖⱼ)` (`0 ≤ γ < 1`) then `abs(Ĝᵢⱼ − (QᵀQ)ᵢⱼ) ≤ γ/(1−γ) √Ĝᵢᵢ √Ĝⱼⱼ` and `‖QᵀQ − I‖_F ≤ ‖Ĝ − I‖_F + γ/(1−γ) tr Ĝ`; with the norm/trace/final-step rounding as hypotheses, `‖QᵀQ − I‖_F ≤ v`, and `v ≤ tol ⟹ ‖QᵀQ − I‖_F ≤ tol`; `γ_m/(1−γ_m) = mu/(1−2mu)` | `gram_entry_bound`, `frob_bound`, `certificate`, `certified`, `gamma_ratio`, `constants_ok` (the implementation's α, β satisfy hα, hβ for κ = (1−u)²(1−γ_{l²}), τ = 1−γ_l, ρ = (1−u)², given 2l²u < 1, 2mu < 1, l ≥ 1) (with `l2_triangle`, `l2_mono`, `l2_outer`) | `CholQR/Verify.lean` |

## What is NOT formalized
- For D1: the FP32 rounding facts (relative error `ε = mu` of the computed column norm, one rounded division per entry,
  no underflow/overflow) are hypotheses, proved on paper.
- For the certificate: the FP64 rounding facts (Gram entry bound from the standard model, rounding of the norm, the
  trace and the final multiply-add, absence of underflow for FP32-valued inputs) are hypotheses, proved on paper.
- The floating-point layer: that a computed Cholesky factor exists, and bounds on `E` (Gram rounding, Cholesky backward
  error, shift rounding) and on the forming error `Δ`. The theorems hold for ANY real `E`, `Δ` satisfying their
  hypotheses; in the analysis these quantities are bounded from the literature or measured (see `benchmarks/device_sweep.py`).
- The passage from quadratic-form bounds to operator norms / singular values (spectral theorem; on paper only).
- The passage from the D1 bound `c‖y‖ ≤ ‖X'y‖` to `σ_min(X') ≥ c` is immediate (definition of `σ_min`), but is stated only on paper.
