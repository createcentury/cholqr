import Mathlib

/-!
# Theorem D of the CholeskyQR rounding theory (exact-arithmetic layer)

`E` is *defined* from the computed factor, `L Lᵀ = XᵀX + s I + E`, so it contains the Gram rounding,
the Cholesky backward error and the shift rounding. Norms are written as dot products: `‖x‖² = x ⬝ᵥ x`.

* `D_a_identity` : `L Lᵀ = XᵀX + s I + E`, `L` invertible, `Q = X L⁻ᵀ`  ⟹  `QᵀQ - I = -(L⁻¹ (s I + E) L⁻ᵀ)`.
* `D_a_quad` : `yᵀ(QᵀQ - I)y = -(wᵀ (s I + E) w)` with `w = L⁻ᵀ y`.
* `D_a_bound` : if `0 ≤ s` and `|zᵀ E z| ≤ e ‖z‖²` for all `z`, then `|yᵀ(QᵀQ - I)y| ≤ (s + e) ‖L⁻ᵀ y‖²`.
* `D_b_LLt` : if moreover `‖X z‖² ≥ c ‖z‖²` for all `z`, then `zᵀ (L Lᵀ) z ≥ (c + s - e) ‖z‖²`.
* `D_b_inv` : and if `c + s - e > 0`, `‖L⁻ᵀ y‖² ≤ ‖y‖² / (c + s - e)`.
* `D_b_bound` : `|yᵀ(QᵀQ - I)y| ≤ (s + e)/(c + s - e) ‖y‖²`.
* `cauchy_schwarz_dot` : `(a ⬝ᵥ b)² ≤ (a ⬝ᵥ a)(b ⬝ᵥ b)`.
* `D_c_bound` : if `|yᵀ(QᵀQ - I)y| ≤ d ‖y‖²` and `‖Δ z‖² ≤ η² ‖z‖²` (`0 ≤ η`) then for `Q̃ = Q + Δ`,
  `|yᵀ(Q̃ᵀQ̃ - I)y| ≤ (d + 2 η √(1 + d) + η²) ‖y‖²`.
Floating point is not modelled: the statements hold for any real `E`, `Δ` satisfying the hypotheses.
-/

open Matrix

namespace CholQR.TheoremD

variable {m n : ℕ}

theorem dot_self_nonneg' (v : Fin n → ℝ) : 0 ≤ v ⬝ᵥ v :=
  Finset.sum_nonneg (fun i _ => mul_self_nonneg (v i))

theorem quad_gram' {k : ℕ} (B : Matrix (Fin k) (Fin n) ℝ) (x : Fin n → ℝ) :
    x ⬝ᵥ ((Bᵀ * B) *ᵥ x) = (B *ᵥ x) ⬝ᵥ (B *ᵥ x) := by
  rw [← mulVec_mulVec, dotProduct_mulVec, vecMul_transpose]

/-! ## D(a) -/

theorem D_a_identity (X : Matrix (Fin m) (Fin n) ℝ) (L E : Matrix (Fin n) (Fin n) ℝ) (s : ℝ)
    (hL : IsUnit L.det) (hLL : L * Lᵀ = Xᵀ * X + s • (1 : Matrix (Fin n) (Fin n) ℝ) + E) :
    (X * (Lᵀ)⁻¹)ᵀ * (X * (Lᵀ)⁻¹) - 1 = -(L⁻¹ * (s • (1 : Matrix (Fin n) (Fin n) ℝ) + E) * (Lᵀ)⁻¹) := by
  have hLt : IsUnit Lᵀ.det := by rwa [det_transpose]
  have hXX : Xᵀ * X = L * Lᵀ - (s • (1 : Matrix (Fin n) (Fin n) ℝ) + E) := by
    rw [hLL]; abel
  have hg : (X * (Lᵀ)⁻¹)ᵀ * (X * (Lᵀ)⁻¹) = L⁻¹ * (Xᵀ * X) * (Lᵀ)⁻¹ := by
    rw [transpose_mul, transpose_nonsing_inv, transpose_transpose]
    simp [Matrix.mul_assoc]
  have hone : L⁻¹ * (L * Lᵀ) * (Lᵀ)⁻¹ = 1 := by
    rw [← Matrix.mul_assoc, nonsing_inv_mul _ hL, Matrix.one_mul, mul_nonsing_inv _ hLt]
  rw [hg, hXX, Matrix.mul_sub, Matrix.sub_mul, hone]
  abel

theorem D_a_quad (X : Matrix (Fin m) (Fin n) ℝ) (L E : Matrix (Fin n) (Fin n) ℝ) (s : ℝ)
    (hL : IsUnit L.det) (hLL : L * Lᵀ = Xᵀ * X + s • (1 : Matrix (Fin n) (Fin n) ℝ) + E) (y : Fin n → ℝ) :
    y ⬝ᵥ (((X * (Lᵀ)⁻¹)ᵀ * (X * (Lᵀ)⁻¹) - 1) *ᵥ y)
      = -(((Lᵀ)⁻¹ *ᵥ y) ⬝ᵥ ((s • (1 : Matrix (Fin n) (Fin n) ℝ) + E) *ᵥ ((Lᵀ)⁻¹ *ᵥ y))) := by
  rw [D_a_identity X L E s hL hLL, neg_mulVec, dotProduct_neg]
  congr 1
  have hv : y ᵥ* L⁻¹ = (Lᵀ)⁻¹ *ᵥ y := by
    rw [← transpose_nonsing_inv, ← vecMul_transpose, transpose_transpose]
  rw [← mulVec_mulVec, ← mulVec_mulVec, dotProduct_mulVec, hv]

theorem D_a_bound (X : Matrix (Fin m) (Fin n) ℝ) (L E : Matrix (Fin n) (Fin n) ℝ) (s e : ℝ)
    (hL : IsUnit L.det) (hLL : L * Lᵀ = Xᵀ * X + s • (1 : Matrix (Fin n) (Fin n) ℝ) + E) (hs : 0 ≤ s)
    (hE : ∀ z : Fin n → ℝ, |z ⬝ᵥ (E *ᵥ z)| ≤ e * (z ⬝ᵥ z)) (y : Fin n → ℝ) :
    |y ⬝ᵥ (((X * (Lᵀ)⁻¹)ᵀ * (X * (Lᵀ)⁻¹) - 1) *ᵥ y)| ≤ (s + e) * (((Lᵀ)⁻¹ *ᵥ y) ⬝ᵥ ((Lᵀ)⁻¹ *ᵥ y)) := by
  rw [D_a_quad X L E s hL hLL y, abs_neg]
  set w := (Lᵀ)⁻¹ *ᵥ y
  have e1 : w ⬝ᵥ ((s • (1 : Matrix (Fin n) (Fin n) ℝ) + E) *ᵥ w) = s * (w ⬝ᵥ w) + w ⬝ᵥ (E *ᵥ w) := by
    rw [add_mulVec, dotProduct_add, smul_mulVec, one_mulVec, dotProduct_smul, smul_eq_mul]
  rw [e1]
  have h1 := hE w
  have h2 := dot_self_nonneg' w
  calc |s * (w ⬝ᵥ w) + w ⬝ᵥ (E *ᵥ w)| ≤ |s * (w ⬝ᵥ w)| + |w ⬝ᵥ (E *ᵥ w)| := abs_add_le _ _
    _ ≤ s * (w ⬝ᵥ w) + e * (w ⬝ᵥ w) := by
        rw [abs_of_nonneg (mul_nonneg hs h2)]; linarith
    _ = (s + e) * (w ⬝ᵥ w) := by ring

/-! ## D(b) -/

theorem D_b_LLt (X : Matrix (Fin m) (Fin n) ℝ) (L E : Matrix (Fin n) (Fin n) ℝ) (s e c : ℝ)
    (hLL : L * Lᵀ = Xᵀ * X + s • (1 : Matrix (Fin n) (Fin n) ℝ) + E)
    (hE : ∀ z : Fin n → ℝ, |z ⬝ᵥ (E *ᵥ z)| ≤ e * (z ⬝ᵥ z))
    (hX : ∀ z : Fin n → ℝ, c * (z ⬝ᵥ z) ≤ (X *ᵥ z) ⬝ᵥ (X *ᵥ z)) (z : Fin n → ℝ) :
    (c + s - e) * (z ⬝ᵥ z) ≤ z ⬝ᵥ ((L * Lᵀ) *ᵥ z) := by
  rw [hLL, add_mulVec, add_mulVec, dotProduct_add, dotProduct_add, quad_gram', smul_mulVec, one_mulVec,
    dotProduct_smul, smul_eq_mul]
  have h1 := hX z
  have h2 := (abs_le.mp (hE z)).1
  nlinarith

theorem D_b_inv (X : Matrix (Fin m) (Fin n) ℝ) (L E : Matrix (Fin n) (Fin n) ℝ) (s e c : ℝ)
    (hL : IsUnit L.det) (hLL : L * Lᵀ = Xᵀ * X + s • (1 : Matrix (Fin n) (Fin n) ℝ) + E)
    (hE : ∀ z : Fin n → ℝ, |z ⬝ᵥ (E *ᵥ z)| ≤ e * (z ⬝ᵥ z))
    (hX : ∀ z : Fin n → ℝ, c * (z ⬝ᵥ z) ≤ (X *ᵥ z) ⬝ᵥ (X *ᵥ z)) (hgap : 0 < c + s - e) (y : Fin n → ℝ) :
    ((Lᵀ)⁻¹ *ᵥ y) ⬝ᵥ ((Lᵀ)⁻¹ *ᵥ y) ≤ (y ⬝ᵥ y) / (c + s - e) := by
  have hLt : IsUnit Lᵀ.det := by rwa [det_transpose]
  set w := (Lᵀ)⁻¹ *ᵥ y
  have hy : Lᵀ *ᵥ w = y := by
    simp only [w]; rw [mulVec_mulVec, mul_nonsing_inv _ hLt, one_mulVec]
  have hyy : y ⬝ᵥ y = w ⬝ᵥ ((L * Lᵀ) *ᵥ w) := by
    rw [← hy]
    have := quad_gram' (Lᵀ) w
    rw [transpose_transpose] at this
    rw [this]
  have hq := D_b_LLt X L E s e c hLL hE hX w
  rw [le_div_iff₀ hgap, hyy]
  linarith [mul_comm (c + s - e) (w ⬝ᵥ w)]

theorem D_b_bound (X : Matrix (Fin m) (Fin n) ℝ) (L E : Matrix (Fin n) (Fin n) ℝ) (s e c : ℝ)
    (hL : IsUnit L.det) (hLL : L * Lᵀ = Xᵀ * X + s • (1 : Matrix (Fin n) (Fin n) ℝ) + E) (hs : 0 ≤ s) (he : 0 ≤ e)
    (hE : ∀ z : Fin n → ℝ, |z ⬝ᵥ (E *ᵥ z)| ≤ e * (z ⬝ᵥ z))
    (hX : ∀ z : Fin n → ℝ, c * (z ⬝ᵥ z) ≤ (X *ᵥ z) ⬝ᵥ (X *ᵥ z)) (hgap : 0 < c + s - e) (y : Fin n → ℝ) :
    |y ⬝ᵥ (((X * (Lᵀ)⁻¹)ᵀ * (X * (Lᵀ)⁻¹) - 1) *ᵥ y)| ≤ (s + e) / (c + s - e) * (y ⬝ᵥ y) := by
  have h1 := D_a_bound X L E s e hL hLL hs hE y
  have h2 := D_b_inv X L E s e c hL hLL hE hX hgap y
  calc _ ≤ (s + e) * (((Lᵀ)⁻¹ *ᵥ y) ⬝ᵥ ((Lᵀ)⁻¹ *ᵥ y)) := h1
    _ ≤ (s + e) * ((y ⬝ᵥ y) / (c + s - e)) := mul_le_mul_of_nonneg_left h2 (by linarith)
    _ = (s + e) / (c + s - e) * (y ⬝ᵥ y) := by ring

/-! ## D(c) -/

theorem cauchy_schwarz_dot (a b : Fin n → ℝ) : (a ⬝ᵥ b) ^ 2 ≤ (a ⬝ᵥ a) * (b ⬝ᵥ b) := by
  have hA := dot_self_nonneg' a
  rcases hA.lt_or_eq with hpos | hzero
  · have key := dot_self_nonneg' ((a ⬝ᵥ a) • b - (a ⬝ᵥ b) • a)
    simp only [sub_dotProduct, dotProduct_sub, smul_dotProduct, dotProduct_smul, smul_eq_mul] at key
    rw [dotProduct_comm b a] at key
    have : 0 ≤ (a ⬝ᵥ a) * ((a ⬝ᵥ a) * (b ⬝ᵥ b) - (a ⬝ᵥ b) ^ 2) := by nlinarith
    have := (mul_nonneg_iff_of_pos_left hpos).mp this
    nlinarith
  · have ha : a = 0 := dotProduct_self_eq_zero.mp hzero.symm
    subst ha; simp

theorem D_c_bound (Q Δ : Matrix (Fin m) (Fin n) ℝ) (d η : ℝ) (hd : 0 ≤ d) (hη : 0 ≤ η)
    (hQ : ∀ y : Fin n → ℝ, |y ⬝ᵥ ((Qᵀ * Q - 1) *ᵥ y)| ≤ d * (y ⬝ᵥ y))
    (hΔ : ∀ z : Fin n → ℝ, (Δ *ᵥ z) ⬝ᵥ (Δ *ᵥ z) ≤ η ^ 2 * (z ⬝ᵥ z)) (y : Fin n → ℝ) :
    |y ⬝ᵥ (((Q + Δ)ᵀ * (Q + Δ) - 1) *ᵥ y)| ≤ (d + 2 * η * Real.sqrt (1 + d) + η ^ 2) * (y ⬝ᵥ y) := by
  have quad : ∀ M : Matrix (Fin m) (Fin n) ℝ,
      y ⬝ᵥ ((Mᵀ * M - 1) *ᵥ y) = (M *ᵥ y) ⬝ᵥ (M *ᵥ y) - y ⬝ᵥ y := by
    intro M; rw [sub_mulVec, dotProduct_sub, quad_gram', one_mulVec]
  have hQy := hQ y
  rw [quad Q] at hQy
  rw [quad (Q + Δ), add_mulVec]
  set Y := y ⬝ᵥ y with hYdef
  set a := Q *ᵥ y
  set b := Δ *ᵥ y
  have hY : 0 ≤ Y := dot_self_nonneg' y
  have hexp : (a + b) ⬝ᵥ (a + b) = a ⬝ᵥ a + 2 * (a ⬝ᵥ b) + b ⬝ᵥ b := by
    rw [add_dotProduct, dotProduct_add, dotProduct_add, dotProduct_comm b a]; ring
  rw [hexp]
  have hA : a ⬝ᵥ a ≤ (1 + d) * Y := by linarith [(abs_le.mp hQy).2]
  have hA0 : 0 ≤ a ⬝ᵥ a := dot_self_nonneg' a
  have hB : b ⬝ᵥ b ≤ η ^ 2 * Y := hΔ y
  have hB0 : 0 ≤ b ⬝ᵥ b := dot_self_nonneg' b
  have hcs := cauchy_schwarz_dot a b
  set t := Real.sqrt (1 + d) with ht
  have ht0 : 0 ≤ t := Real.sqrt_nonneg _
  have ht2 : t ^ 2 = 1 + d := Real.sq_sqrt (by linarith)
  have hp : |a ⬝ᵥ b| ≤ t * η * Y := by
    apply abs_le_of_sq_le_sq _ (by positivity)
    calc (a ⬝ᵥ b) ^ 2 ≤ (a ⬝ᵥ a) * (b ⬝ᵥ b) := hcs
      _ ≤ ((1 + d) * Y) * (η ^ 2 * Y) := mul_le_mul hA hB hB0 (by nlinarith)
      _ = (t * η * Y) ^ 2 := by rw [mul_pow, mul_pow, ht2]; ring
  have h1 := abs_le.mp hQy
  have h2 := abs_le.mp hp
  rw [abs_le]
  constructor <;> nlinarith

end CholQR.TheoremD
