import Mathlib
import CholQR.Verify

/-!
# Lemma D1 (contraction into the next pass): the algebraic layer

A pass ends with `Q̃` (here `Q`, an exact real `m × l` matrix) satisfying the conclusion of Theorem D (c),
`|yᵀ(Q̃ᵀQ̃ - I)y| ≤ d̄ ‖y‖²` with `d̄ < 1`. The next pass normalises the columns in floating point,
`X' = fl(Q̃ / fl(‖q̃_j‖))` (`Q / d` in `cholqr.core`). The rounding facts enter as explicit hypotheses:

* `hdh` : the computed column scale is `d̂_j = ‖q̃_j‖ (1 + θ_j)` with `|θ_j| ≤ ε` (`ε = m u` for the FP32 norm);
* `hX`  : one rounded division per entry, `|X'ᵢⱼ - Z'ᵢⱼ| ≤ u |Z'ᵢⱼ|`, where `Z' = Q̃ D̂⁻¹` is the exact quotient
  by the *computed* scales.

Norms are `‖x‖ = √(x ⬝ᵥ x)`; a lower bound `c ‖y‖ ≤ ‖X' y‖` for all `y` is `σ_min(X') ≥ c`.

* `col_sq_bounds` : `1 - d̄ ≤ ‖q̃_j‖² ≤ 1 + d̄`.
* `D1_scaled`     : `√((1-d̄)/(1+d̄)) / (1+ε) · ‖y‖ ≤ ‖Z' y‖`.
* `D1_delta`      : `‖(Z' - X') y‖ ≤ √l · u/(1-ε) · ‖y‖`  (`‖Δy‖ ≤ ‖Δ‖_F ‖y‖`, `‖Δ‖_F ≤ u ‖Z'‖_F`,
  `‖Z'‖_F² = ∑ⱼ 1/(1+θⱼ)²`).
* `D1_sharp`      : `(√((1-d̄)/(1+d̄)) / (1+ε) - √l · u/(1-ε)) ‖y‖ ≤ ‖X' y‖`.
* `D1_paper`      : the note's form, valid for `ε ≤ 1/2`: `((1-ε) √((1-d̄)/(1+d̄)) - 2 √l u) ‖y‖ ≤ ‖X' y‖`.
-/

open Matrix Finset

namespace CholQR.D1

variable {m l : ℕ}

theorem dot_eq_sum_sq {n : ℕ} (v : Fin n → ℝ) : v ⬝ᵥ v = ∑ i, v i ^ 2 := by
  simp [dotProduct, sq]

theorem quad_sub_one (Q : Matrix (Fin m) (Fin l) ℝ) (y : Fin l → ℝ) :
    y ⬝ᵥ ((Qᵀ * Q - 1) *ᵥ y) = (Q *ᵥ y) ⬝ᵥ (Q *ᵥ y) - y ⬝ᵥ y := by
  rw [sub_mulVec, dotProduct_sub, ← mulVec_mulVec, dotProduct_mulVec, vecMul_transpose, one_mulVec]

/-- `0 ≤ c`, `c² Y ≤ Z`, `0 ≤ Y` ⟹ `c √Y ≤ √Z`. -/
theorem mul_sqrt_le (c Y Z : ℝ) (hc : 0 ≤ c) (h : c ^ 2 * Y ≤ Z) : c * Real.sqrt Y ≤ Real.sqrt Z := by
  have e : c * Real.sqrt Y = Real.sqrt (c ^ 2 * Y) := by
    rw [Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq hc]
  rw [e]; exact Real.sqrt_le_sqrt h

/-- Column norms of `Q̃`: `1 - d̄ ≤ ‖q̃_j‖² ≤ 1 + d̄`. -/
theorem col_sq_bounds (Q : Matrix (Fin m) (Fin l) ℝ) (dbar : ℝ)
    (hQ : ∀ y : Fin l → ℝ, |y ⬝ᵥ ((Qᵀ * Q - 1) *ᵥ y)| ≤ dbar * (y ⬝ᵥ y)) (j : Fin l) :
    1 - dbar ≤ ∑ k, Q k j ^ 2 ∧ ∑ k, Q k j ^ 2 ≤ 1 + dbar := by
  have h := hQ (Pi.single j 1)
  rw [quad_sub_one, dot_eq_sum_sq, dot_eq_sum_sq] at h
  have e1 : ∑ i, (Q *ᵥ Pi.single j 1) i ^ 2 = ∑ k, Q k j ^ 2 := by
    simp
  have e2 : ∑ i, (Pi.single j (1 : ℝ) : Fin l → ℝ) i ^ 2 = 1 := by
    rw [Finset.sum_eq_single j]
    · simp
    · intro b _ hb; simp [hb]
    · simp
  rw [e1, e2, mul_one] at h
  have := abs_le.mp h
  constructor <;> linarith [this.1, this.2]

/-- Facts about one computed scale: `d̂ⱼ² = ‖q̃ⱼ‖² (1+θⱼ)²`, `0 < d̂ⱼ²`, `d̂ⱼ² ≤ (1+d̄)(1+ε)²`. -/
theorem scale_facts (Q : Matrix (Fin m) (Fin l) ℝ) (dh θ : Fin l → ℝ) (dbar ε : ℝ)
    (hd1 : dbar < 1) (hε1 : ε < 1)
    (hQ : ∀ y : Fin l → ℝ, |y ⬝ᵥ ((Qᵀ * Q - 1) *ᵥ y)| ≤ dbar * (y ⬝ᵥ y))
    (hdh : ∀ j, dh j = Real.sqrt (∑ k, Q k j ^ 2) * (1 + θ j)) (hθ : ∀ j, |θ j| ≤ ε) (j : Fin l) :
    dh j ^ 2 = (∑ k, Q k j ^ 2) * (1 + θ j) ^ 2 ∧ 0 < ∑ k, Q k j ^ 2 ∧ 0 < dh j ^ 2 ∧
      dh j ^ 2 ≤ (1 + dbar) * (1 + ε) ^ 2 ∧ (1 - ε) ^ 2 ≤ (1 + θ j) ^ 2 := by
  obtain ⟨hS1, hS2⟩ := col_sq_bounds Q dbar hQ j
  have hS0 : 0 < ∑ k, Q k j ^ 2 := by linarith
  have hθj := abs_le.mp (hθ j)
  have hsq : dh j ^ 2 = (∑ k, Q k j ^ 2) * (1 + θ j) ^ 2 := by
    rw [hdh, mul_pow, Real.sq_sqrt hS0.le]
  have hlo : (1 - ε) ^ 2 ≤ (1 + θ j) ^ 2 := by nlinarith
  have hhi : (1 + θ j) ^ 2 ≤ (1 + ε) ^ 2 := by nlinarith
  have hpos : 0 < (1 + θ j) ^ 2 := by nlinarith
  refine ⟨hsq, hS0, by rw [hsq]; positivity, ?_, hlo⟩
  rw [hsq]; exact mul_le_mul hS2 hhi hpos.le (by linarith)

/-- Exact scaling by the computed scales: `√((1-d̄)/(1+d̄)) / (1+ε) · ‖y‖ ≤ ‖Z' y‖`. -/
theorem D1_scaled (Q : Matrix (Fin m) (Fin l) ℝ) (dh θ : Fin l → ℝ) (dbar ε : ℝ)
    (hd0 : 0 ≤ dbar) (hd1 : dbar < 1) (hε0 : 0 ≤ ε) (hε1 : ε < 1)
    (hQ : ∀ y : Fin l → ℝ, |y ⬝ᵥ ((Qᵀ * Q - 1) *ᵥ y)| ≤ dbar * (y ⬝ᵥ y))
    (hdh : ∀ j, dh j = Real.sqrt (∑ k, Q k j ^ 2) * (1 + θ j)) (hθ : ∀ j, |θ j| ≤ ε)
    (y : Fin l → ℝ) :
    Real.sqrt ((1 - dbar) / (1 + dbar)) / (1 + ε) * Real.sqrt (y ⬝ᵥ y)
      ≤ Real.sqrt ((of (fun i j => Q i j / dh j) *ᵥ y) ⬝ᵥ (of (fun i j => Q i j / dh j) *ᵥ y)) := by
  set w : Fin l → ℝ := fun j => y j / dh j
  have hZ : of (fun i j => Q i j / dh j) *ᵥ y = Q *ᵥ w := by
    funext i; simp only [mulVec, dotProduct, of_apply, w]
    refine Finset.sum_congr rfl (fun j _ => ?_); ring
  rw [hZ]
  have hQw : (1 - dbar) * (w ⬝ᵥ w) ≤ (Q *ᵥ w) ⬝ᵥ (Q *ᵥ w) := by
    have := (abs_le.mp (hQ w)).1
    rw [quad_sub_one] at this; linarith
  set K := (1 + dbar) * (1 + ε) ^ 2 with hK
  have hKpos : 0 < K := by positivity
  have hw : (y ⬝ᵥ y) / K ≤ w ⬝ᵥ w := by
    rw [dot_eq_sum_sq, dot_eq_sum_sq, Finset.sum_div]
    refine Finset.sum_le_sum (fun j _ => ?_)
    obtain ⟨-, -, hpos, hle, -⟩ := scale_facts Q dh θ dbar ε hd1 hε1 hQ hdh hθ j
    simp only [w, div_pow]
    exact div_le_div_of_nonneg_left (sq_nonneg _) hpos hle
  have hc0 : 0 ≤ Real.sqrt ((1 - dbar) / (1 + dbar)) / (1 + ε) := by positivity
  apply mul_sqrt_le _ _ _ hc0
  have hc2 : (Real.sqrt ((1 - dbar) / (1 + dbar)) / (1 + ε)) ^ 2 = (1 - dbar) / K := by
    rw [div_pow, Real.sq_sqrt (div_nonneg (by linarith) (by linarith)), hK]
    field_simp
  rw [hc2]
  calc (1 - dbar) / K * (y ⬝ᵥ y) = (1 - dbar) * ((y ⬝ᵥ y) / K) := by ring
    _ ≤ (1 - dbar) * (w ⬝ᵥ w) := mul_le_mul_of_nonneg_left hw (by linarith)
    _ ≤ _ := hQw

/-- Division rounding: for `Δ = Z' - X'` with `|Δᵢⱼ| ≤ u |Z'ᵢⱼ|`, `‖Δ y‖ ≤ √l · u/(1-ε) · ‖y‖`. -/
theorem D1_delta (Q X : Matrix (Fin m) (Fin l) ℝ) (dh θ : Fin l → ℝ) (dbar ε u : ℝ)
    (hd1 : dbar < 1) (hε1 : ε < 1) (hu : 0 ≤ u)
    (hQ : ∀ y : Fin l → ℝ, |y ⬝ᵥ ((Qᵀ * Q - 1) *ᵥ y)| ≤ dbar * (y ⬝ᵥ y))
    (hdh : ∀ j, dh j = Real.sqrt (∑ k, Q k j ^ 2) * (1 + θ j)) (hθ : ∀ j, |θ j| ≤ ε)
    (hX : ∀ i j, |X i j - Q i j / dh j| ≤ u * |Q i j / dh j|) (y : Fin l → ℝ) :
    Real.sqrt (((of (fun i j => Q i j / dh j) - X) *ᵥ y) ⬝ᵥ ((of (fun i j => Q i j / dh j) - X) *ᵥ y))
      ≤ Real.sqrt l * u / (1 - ε) * Real.sqrt (y ⬝ᵥ y) := by
  set Δ := of (fun i j => Q i j / dh j) - X with hΔ
  have hΔij : ∀ i j, Δ i j ^ 2 ≤ u ^ 2 * (Q i j ^ 2 / dh j ^ 2) := by
    intro i j
    have h1 : |Δ i j| ≤ u * |Q i j / dh j| := by
      rw [hΔ, Matrix.sub_apply, of_apply, abs_sub_comm]; exact hX i j
    have h2 := pow_le_pow_left₀ (abs_nonneg _) h1 2
    rwa [sq_abs, mul_pow, sq_abs, div_pow] at h2
  -- ‖Δ‖_F² ≤ l u²/(1-ε)²
  have hF : ∑ i, ∑ j, Δ i j ^ 2 ≤ l * (u ^ 2 / (1 - ε) ^ 2) := by
    calc ∑ i, ∑ j, Δ i j ^ 2 ≤ ∑ i, ∑ j, u ^ 2 * (Q i j ^ 2 / dh j ^ 2) :=
          Finset.sum_le_sum (fun i _ => Finset.sum_le_sum (fun j _ => hΔij i j))
      _ = ∑ j, u ^ 2 * ((∑ i, Q i j ^ 2) / dh j ^ 2) := by
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl (fun j _ => ?_)
          rw [← Finset.mul_sum, Finset.sum_div]
      _ ≤ ∑ _j : Fin l, u ^ 2 / (1 - ε) ^ 2 := by
          refine Finset.sum_le_sum (fun j _ => ?_)
          obtain ⟨hsq, hS0, -, -, hlo⟩ := scale_facts Q dh θ dbar ε hd1 hε1 hQ hdh hθ j
          have e : u ^ 2 * ((∑ i, Q i j ^ 2) / dh j ^ 2) = u ^ 2 / (1 + θ j) ^ 2 := by
            rw [hsq]; field_simp
          rw [e]
          exact div_le_div_of_nonneg_left (sq_nonneg u) (by nlinarith) hlo
      _ = l * (u ^ 2 / (1 - ε) ^ 2) := by simp
  -- ‖Δ y‖² ≤ ‖Δ‖_F² ‖y‖²
  have hrow : (Δ *ᵥ y) ⬝ᵥ (Δ *ᵥ y) ≤ (∑ i, ∑ j, Δ i j ^ 2) * (y ⬝ᵥ y) := by
    rw [dot_eq_sum_sq, dot_eq_sum_sq, Finset.sum_mul]
    refine Finset.sum_le_sum (fun i _ => ?_)
    simp only [mulVec, dotProduct]
    exact Finset.sum_mul_sq_le_sq_mul_sq _ _ _
  have hc0 : 0 ≤ Real.sqrt l * u / (1 - ε) := div_nonneg (mul_nonneg (Real.sqrt_nonneg _) hu) (by linarith)
  have hc2 : (Real.sqrt l * u / (1 - ε)) ^ 2 = l * (u ^ 2 / (1 - ε) ^ 2) := by
    rw [div_pow, mul_pow, Real.sq_sqrt (Nat.cast_nonneg l)]; ring
  have hY := dot_self_nonneg' y
  rw [← Real.sqrt_sq hc0, ← Real.sqrt_mul (sq_nonneg _), hc2]
  apply Real.sqrt_le_sqrt
  exact le_trans hrow (mul_le_mul_of_nonneg_right hF hY)
where
  dot_self_nonneg' (v : Fin l → ℝ) : 0 ≤ v ⬝ᵥ v := by
    rw [dot_eq_sum_sq]; exact Finset.sum_nonneg (fun i _ => sq_nonneg (v i))

theorem norm_eq_l2 {n : ℕ} (v : Fin n → ℝ) : Real.sqrt (v ⬝ᵥ v) = CholQR.Verify.l2 v := by
  rw [dot_eq_sum_sq]; rfl

/-- **D1 (sharp form).** `σ_min(X') ≥ √((1-d̄)/(1+d̄)) / (1+ε) - √l · u/(1-ε)`. -/
theorem D1_sharp (Q X : Matrix (Fin m) (Fin l) ℝ) (dh θ : Fin l → ℝ) (dbar ε u : ℝ)
    (hd0 : 0 ≤ dbar) (hd1 : dbar < 1) (hε0 : 0 ≤ ε) (hε1 : ε < 1) (hu : 0 ≤ u)
    (hQ : ∀ y : Fin l → ℝ, |y ⬝ᵥ ((Qᵀ * Q - 1) *ᵥ y)| ≤ dbar * (y ⬝ᵥ y))
    (hdh : ∀ j, dh j = Real.sqrt (∑ k, Q k j ^ 2) * (1 + θ j)) (hθ : ∀ j, |θ j| ≤ ε)
    (hX : ∀ i j, |X i j - Q i j / dh j| ≤ u * |Q i j / dh j|) (y : Fin l → ℝ) :
    (Real.sqrt ((1 - dbar) / (1 + dbar)) / (1 + ε) - Real.sqrt l * u / (1 - ε)) * Real.sqrt (y ⬝ᵥ y)
      ≤ Real.sqrt ((X *ᵥ y) ⬝ᵥ (X *ᵥ y)) := by
  set Z := of (fun i j => Q i j / dh j)
  have h1 := D1_scaled Q dh θ dbar ε hd0 hd1 hε0 hε1 hQ hdh hθ y
  have h2 := D1_delta Q X dh θ dbar ε u hd1 hε1 hu hQ hdh hθ hX y
  have hsplit : Z *ᵥ y = X *ᵥ y + (Z - X) *ᵥ y := by rw [sub_mulVec]; abel
  have htri := CholQR.Verify.l2_triangle (X *ᵥ y) ((Z - X) *ᵥ y)
  rw [← hsplit, ← norm_eq_l2, ← norm_eq_l2, ← norm_eq_l2] at htri
  nlinarith

/-- **D1 (the note's form).** For `ε ≤ 1/2` (`m u ≤ 1/2`):
`σ_min(X') ≥ (1-ε) √((1-d̄)/(1+d̄)) - 2 √l u`. -/
theorem D1_paper (Q X : Matrix (Fin m) (Fin l) ℝ) (dh θ : Fin l → ℝ) (dbar ε u : ℝ)
    (hd0 : 0 ≤ dbar) (hd1 : dbar < 1) (hε0 : 0 ≤ ε) (hε : ε ≤ 1 / 2) (hu : 0 ≤ u)
    (hQ : ∀ y : Fin l → ℝ, |y ⬝ᵥ ((Qᵀ * Q - 1) *ᵥ y)| ≤ dbar * (y ⬝ᵥ y))
    (hdh : ∀ j, dh j = Real.sqrt (∑ k, Q k j ^ 2) * (1 + θ j)) (hθ : ∀ j, |θ j| ≤ ε)
    (hX : ∀ i j, |X i j - Q i j / dh j| ≤ u * |Q i j / dh j|) (y : Fin l → ℝ) :
    ((1 - ε) * Real.sqrt ((1 - dbar) / (1 + dbar)) - 2 * Real.sqrt l * u) * Real.sqrt (y ⬝ᵥ y)
      ≤ Real.sqrt ((X *ᵥ y) ⬝ᵥ (X *ᵥ y)) := by
  have hε1 : ε < 1 := by linarith
  refine le_trans ?_ (D1_sharp Q X dh θ dbar ε u hd0 hd1 hε0 hε1 hu hQ hdh hθ hX y)
  apply mul_le_mul_of_nonneg_right _ (Real.sqrt_nonneg _)
  set a := Real.sqrt ((1 - dbar) / (1 + dbar))
  set b := Real.sqrt l * u
  have ha : 0 ≤ a := Real.sqrt_nonneg _
  have hb : 0 ≤ b := mul_nonneg (Real.sqrt_nonneg _) hu
  -- (1-ε) ≤ 1/(1+ε)  and  1/(1-ε) ≤ 2
  have h1 : (1 - ε) * a ≤ a / (1 + ε) := by
    rw [le_div_iff₀ (by linarith)]; nlinarith [mul_nonneg (sq_nonneg ε) ha]
  have h2 : b / (1 - ε) ≤ 2 * b := by
    rw [div_le_iff₀ (by linarith)]; nlinarith
  linarith

end CholQR.D1
