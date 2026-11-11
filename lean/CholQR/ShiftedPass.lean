import Mathlib

/-!
# Rounding theory of (shifted) CholeskyQR: the exact pass-1 spectrum (T0) and the indefinite-part lemma (Lemma I)


* T0 (exact arithmetic). Let `L Lᵀ = G + s I` with `L` invertible, `G = YᵀY`, and `Q₁ = Y L⁻ᵀ`.
  - `gram_Q1` : `Q₁ᵀ Q₁ = L⁻¹ G L⁻ᵀ`.
  - `T0_eigvec` : if `G v = λ v` and `λ + s ≠ 0`, then `(L⁻¹ G L⁻ᵀ)(Lᵀ v) = (λ / (λ + s)) (Lᵀ v)`;
    i.e. every eigenpair `(λ, v)` of `G` gives the eigenpair `(λ/(λ+s), Lᵀ v)` of `Q₁ᵀQ₁` (and `Lᵀ v ≠ 0` if `v ≠ 0`,
    `T0_eigvec_ne_zero`; `Lᵀ` is injective, so an eigenbasis of `G` is mapped to an eigenbasis of `Q₁ᵀQ₁`).
  - `T0_similar` : `Lᵀ⁻¹ (L⁻¹ G Lᵀ⁻¹) Lᵀ = (G + s I)⁻¹ G` (similarity).
  - `T0_resolvent_eig` : `(G + s I)⁻¹ G v = (λ/(λ+s)) v`.
* Lemma I (indefinite part). If `Ĝ = BᵀB + A` with `‖B x‖² ≥ c ‖x‖²` and `|xᵀ A x| ≤ a ‖x‖²` for all `x`, then
  `xᵀ (Ĝ + s I) x ≥ (c + s - a) ‖x‖²` (`lemmaI_quadratic`); if moreover `A` is symmetric and `c + s > a`, then
  `Ĝ + s I` is positive definite (`lemmaI_posDef`), i.e. its Cholesky factorization exists.
-/

open Matrix

namespace CholQR.ShiftedPass

variable {m n : ℕ}

/-! ## T0 -/

theorem gram_Q1 (Y : Matrix (Fin m) (Fin n) ℝ) (L : Matrix (Fin n) (Fin n) ℝ) :
    (Y * (Lᵀ)⁻¹)ᵀ * (Y * (Lᵀ)⁻¹) = L⁻¹ * (Yᵀ * Y) * (Lᵀ)⁻¹ := by
  rw [transpose_mul, transpose_nonsing_inv, transpose_transpose]
  simp [Matrix.mul_assoc]

theorem T0_resolvent_eig (G : Matrix (Fin n) (Fin n) ℝ) (s lam : ℝ) (v : Fin n → ℝ)
    (hA : IsUnit (G + s • (1 : Matrix (Fin n) (Fin n) ℝ)).det)
    (hv : G *ᵥ v = lam • v) (hls : lam + s ≠ 0) :
    (G + s • (1 : Matrix (Fin n) (Fin n) ℝ))⁻¹ *ᵥ (G *ᵥ v) = (lam / (lam + s)) • v := by
  set A := G + s • (1 : Matrix (Fin n) (Fin n) ℝ)
  have hAv : A *ᵥ v = (lam + s) • v := by
    simp [A, add_mulVec, hv, smul_mulVec, add_smul]
  have hinv : A⁻¹ *ᵥ v = (lam + s)⁻¹ • v := by
    have h1 : A⁻¹ *ᵥ (A *ᵥ v) = v := by
      rw [mulVec_mulVec, nonsing_inv_mul _ hA, one_mulVec]
    rw [hAv, mulVec_smul] at h1
    conv_rhs => rw [← h1]
    rw [smul_smul, inv_mul_cancel₀ hls, one_smul]
  rw [hv, mulVec_smul, hinv, smul_smul, div_eq_mul_inv]

theorem T0_eigvec (G L : Matrix (Fin n) (Fin n) ℝ) (s lam : ℝ) (v : Fin n → ℝ)
    (hL : IsUnit L.det) (hLL : L * Lᵀ = G + s • (1 : Matrix (Fin n) (Fin n) ℝ))
    (hv : G *ᵥ v = lam • v) (hls : lam + s ≠ 0) :
    (L⁻¹ * G * (Lᵀ)⁻¹) *ᵥ (Lᵀ *ᵥ v) = (lam / (lam + s)) • (Lᵀ *ᵥ v) := by
  have hLt : IsUnit Lᵀ.det := by rwa [det_transpose]
  have hcancel : (Lᵀ)⁻¹ *ᵥ (Lᵀ *ᵥ v) = v := by
    rw [mulVec_mulVec, nonsing_inv_mul _ hLt, one_mulVec]
  -- L (Lᵀ v) = (λ + s) v, hence L⁻¹ v = (λ+s)⁻¹ Lᵀ v
  have hLLv : L *ᵥ (Lᵀ *ᵥ v) = (lam + s) • v := by
    rw [mulVec_mulVec, hLL]; simp [add_mulVec, hv, smul_mulVec, add_smul]
  have hLinv : L⁻¹ *ᵥ v = (lam + s)⁻¹ • (Lᵀ *ᵥ v) := by
    have h1 : L⁻¹ *ᵥ (L *ᵥ (Lᵀ *ᵥ v)) = Lᵀ *ᵥ v := by
      rw [mulVec_mulVec, nonsing_inv_mul _ hL, one_mulVec]
    rw [hLLv, mulVec_smul] at h1
    rw [← h1, smul_smul, inv_mul_cancel₀ hls, one_smul]
  rw [← mulVec_mulVec, ← mulVec_mulVec, hcancel, hv, mulVec_smul, hLinv, smul_smul, div_eq_mul_inv]

theorem T0_eigvec_ne_zero (L : Matrix (Fin n) (Fin n) ℝ) (hL : IsUnit L.det) (v : Fin n → ℝ) (hv : v ≠ 0) :
    Lᵀ *ᵥ v ≠ 0 := by
  intro h
  have hLt : IsUnit Lᵀ.det := by rwa [det_transpose]
  apply hv
  have : (Lᵀ)⁻¹ *ᵥ (Lᵀ *ᵥ v) = v := by rw [mulVec_mulVec, nonsing_inv_mul _ hLt, one_mulVec]
  rw [← this, h, mulVec_zero]

theorem T0_similar (G L : Matrix (Fin n) (Fin n) ℝ) (s : ℝ)
    (hL : IsUnit L.det) (hLL : L * Lᵀ = G + s • (1 : Matrix (Fin n) (Fin n) ℝ)) :
    (Lᵀ)⁻¹ * (L⁻¹ * G * (Lᵀ)⁻¹) * Lᵀ = (G + s • (1 : Matrix (Fin n) (Fin n) ℝ))⁻¹ * G := by
  have hLt : IsUnit Lᵀ.det := by rwa [det_transpose]
  rw [← hLL, Matrix.mul_inv_rev]
  calc (Lᵀ)⁻¹ * (L⁻¹ * G * (Lᵀ)⁻¹) * Lᵀ = (Lᵀ)⁻¹ * L⁻¹ * G * ((Lᵀ)⁻¹ * Lᵀ) := by
        simp only [Matrix.mul_assoc]
    _ = (Lᵀ)⁻¹ * L⁻¹ * G := by rw [nonsing_inv_mul _ hLt, Matrix.mul_one]

/-! ## Lemma I -/

theorem quad_gram (B : Matrix (Fin m) (Fin n) ℝ) (x : Fin n → ℝ) :
    x ⬝ᵥ ((Bᵀ * B) *ᵥ x) = (B *ᵥ x) ⬝ᵥ (B *ᵥ x) := by
  rw [← mulVec_mulVec, dotProduct_mulVec, vecMul_transpose]

theorem lemmaI_quadratic (B : Matrix (Fin m) (Fin n) ℝ) (A : Matrix (Fin n) (Fin n) ℝ) (c a s : ℝ)
    (hB : ∀ x : Fin n → ℝ, c * (x ⬝ᵥ x) ≤ (B *ᵥ x) ⬝ᵥ (B *ᵥ x))
    (hA : ∀ x : Fin n → ℝ, |x ⬝ᵥ (A *ᵥ x)| ≤ a * (x ⬝ᵥ x)) (x : Fin n → ℝ) :
    (c + s - a) * (x ⬝ᵥ x) ≤ x ⬝ᵥ ((Bᵀ * B + A + s • (1 : Matrix (Fin n) (Fin n) ℝ)) *ᵥ x) := by
  have e : x ⬝ᵥ ((Bᵀ * B + A + s • (1 : Matrix (Fin n) (Fin n) ℝ)) *ᵥ x)
      = (B *ᵥ x) ⬝ᵥ (B *ᵥ x) + x ⬝ᵥ (A *ᵥ x) + s * (x ⬝ᵥ x) := by
    rw [add_mulVec, add_mulVec, dotProduct_add, dotProduct_add, quad_gram, smul_mulVec, one_mulVec,
      dotProduct_smul, smul_eq_mul]
  rw [e]
  have h1 := hB x
  have h2 := (abs_le.mp (hA x)).1
  nlinarith

theorem lemmaI_posDef (B : Matrix (Fin m) (Fin n) ℝ) (A : Matrix (Fin n) (Fin n) ℝ) (c a s : ℝ)
    (hAsym : Aᵀ = A)
    (hB : ∀ x : Fin n → ℝ, c * (x ⬝ᵥ x) ≤ (B *ᵥ x) ⬝ᵥ (B *ᵥ x))
    (hA : ∀ x : Fin n → ℝ, |x ⬝ᵥ (A *ᵥ x)| ≤ a * (x ⬝ᵥ x)) (hgap : a < c + s) :
    (Bᵀ * B + A + s • (1 : Matrix (Fin n) (Fin n) ℝ)).PosDef := by
  refine Matrix.PosDef.of_dotProduct_mulVec_pos ?_ ?_
  · unfold IsHermitian
    rw [conjTranspose_eq_transpose_of_trivial]
    simp [transpose_add, transpose_mul, hAsym, transpose_smul, transpose_one]
  · intro x hx
    have hq := lemmaI_quadratic B A c a s hB hA x
    have hxx : 0 < x ⬝ᵥ x := by
      obtain ⟨i, hi⟩ : ∃ i, x i ≠ 0 := by
        by_contra h; push Not at h; exact hx (funext h)
      have : 0 < x i * x i := mul_self_pos.mpr hi
      exact lt_of_lt_of_le this (by
        unfold dotProduct
        exact Finset.single_le_sum (f := fun j => x j * x j) (fun j _ => mul_self_nonneg (x j))
          (Finset.mem_univ i))
    have : 0 < (c + s - a) * (x ⬝ᵥ x) := mul_pos (by linarith) hxx
    rw [star_trivial]
    exact lt_of_lt_of_le this hq

end CholQR.ShiftedPass
