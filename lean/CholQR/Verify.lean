import Mathlib

/-!
# A-posteriori orthogonality certificate (`verify=True`): the algebraic layer

`Q` is the exact real matrix (the returned FP32/FP16/... values, exact in FP64), `A = QᵀQ`, `Ĝ` the computed FP64
Gram matrix. The floating-point facts enter as explicit hypotheses (they are proved on paper from the IEEE standard
model, see `paper/theory_notes/VERIFY_CERTIFICATE_1112.md`):

* `hG`  : `|Ĝ i j - A i j| ≤ γ ∑ₖ |Q k i| |Q k j|`        (Gram rounding, any summation order; `γ = γ_m`)
* `he`  : `κ ‖Ĝ - I‖_F ≤ ê`                               (computed norm of `fl(Ĝ - I)`)
* `ht`  : `τ tr Ĝ ≤ t̂`                                    (computed trace)
* `hv`  : `ρ (α ê + β t̂) ≤ v`                             (computed final multiply-add)
* `hα`, `hβ` : the constants used are large enough.

Results:
* `l2_triangle`, `l2_mono`, `l2_outer` : Frobenius-norm facts (Frobenius norm = `l2` over index pairs).
* `gram_entry_bound` : `|Ĝ i j - A i j| ≤ γ/(1-γ) √Ĝᵢᵢ √Ĝⱼⱼ`  (Cauchy–Schwarz + diagonal self-bound).
* `frob_bound`       : `‖A - I‖_F ≤ ‖Ĝ - I‖_F + γ/(1-γ) tr Ĝ`.
* `certificate`      : `‖A - I‖_F ≤ v`, and `certified` : `v ≤ tol → ‖A - I‖_F ≤ tol`.
* `gamma_ratio`      : `γ_m/(1-γ_m) = m u/(1 - 2 m u)` for `γ_m = m u/(1 - m u)`.
-/

open Matrix Finset

namespace CholQR.Verify

/-! ## Euclidean norm of a finite family -/

section L2
variable {ι : Type*} [Fintype ι]

/-- `l2 x = √(∑ xᵢ²)`. -/
noncomputable def l2 (x : ι → ℝ) : ℝ := Real.sqrt (∑ i, x i ^ 2)

theorem l2_nonneg (x : ι → ℝ) : 0 ≤ l2 x := Real.sqrt_nonneg _

theorem sum_sq_nonneg (x : ι → ℝ) : 0 ≤ ∑ i, x i ^ 2 :=
  Finset.sum_nonneg (fun i _ => sq_nonneg (x i))

theorem sum_mul_le_sqrt (a b : ι → ℝ) :
    ∑ i, a i * b i ≤ Real.sqrt (∑ i, a i ^ 2) * Real.sqrt (∑ i, b i ^ 2) := by
  have cs := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ a b
  rw [← Real.sqrt_mul (sum_sq_nonneg a)]
  exact le_trans (le_abs_self _) (Real.abs_le_sqrt cs)

theorem l2_triangle (a b : ι → ℝ) : l2 (a + b) ≤ l2 a + l2 b := by
  unfold l2
  have hA := sum_sq_nonneg a
  have hB := sum_sq_nonneg b
  have hcs := sum_mul_le_sqrt a b
  have hexp : ∑ i, (a + b) i ^ 2 = ∑ i, a i ^ 2 + ∑ i, b i ^ 2 + 2 * ∑ i, a i * b i := by
    simp only [Pi.add_apply, add_sq, Finset.sum_add_distrib, Finset.mul_sum]
    ring_nf
  have hsum0 : 0 ≤ Real.sqrt (∑ i, a i ^ 2) + Real.sqrt (∑ i, b i ^ 2) :=
    add_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  rw [← Real.sqrt_sq hsum0]
  apply Real.sqrt_le_sqrt
  rw [hexp]
  have h1 := Real.sq_sqrt hA
  have h2 := Real.sq_sqrt hB
  nlinarith [Real.sqrt_nonneg (∑ i, a i ^ 2), Real.sqrt_nonneg (∑ i, b i ^ 2)]

theorem l2_mono (a b : ι → ℝ) (h : ∀ i, |a i| ≤ b i) : l2 a ≤ l2 b := by
  unfold l2
  apply Real.sqrt_le_sqrt
  apply Finset.sum_le_sum
  intro i _
  have h0 : 0 ≤ |a i| := abs_nonneg _
  have := h i
  nlinarith [sq_abs (a i)]

/-- Frobenius norm of the rank-one family `c √gᵢ √gⱼ`. -/
theorem l2_outer (g : ι → ℝ) (hg : ∀ i, 0 ≤ g i) (c : ℝ) (hc : 0 ≤ c) :
    l2 (fun p : ι × ι => c * Real.sqrt (g p.1) * Real.sqrt (g p.2)) = c * ∑ i, g i := by
  unfold l2
  have hs : ∀ i, Real.sqrt (g i) ^ 2 = g i := fun i => Real.sq_sqrt (hg i)
  have key : ∑ p : ι × ι, (c * Real.sqrt (g p.1) * Real.sqrt (g p.2)) ^ 2 = (c * ∑ i, g i) ^ 2 := by
    rw [Fintype.sum_prod_type]
    simp only [mul_pow, hs]
    rw [sq (∑ i, g i), Finset.sum_mul_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl (fun i _ => ?_)
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    ring
  rw [key, Real.sqrt_sq]
  exact mul_nonneg hc (Finset.sum_nonneg (fun i _ => hg i))

end L2

/-! ## Gram-matrix layer -/

variable {m l : ℕ}

/-- Frobenius norm of an `l × l` matrix. -/
noncomputable def frob (M : Matrix (Fin l) (Fin l) ℝ) : ℝ := l2 (fun p : Fin l × Fin l => M p.1 p.2)

theorem gram_diag (Q : Matrix (Fin m) (Fin l) ℝ) (i : Fin l) :
    (Qᵀ * Q) i i = ∑ k, Q k i ^ 2 := by
  simp [Matrix.mul_apply, sq]

theorem gram_diag_nonneg (Q : Matrix (Fin m) (Fin l) ℝ) (i : Fin l) : 0 ≤ (Qᵀ * Q) i i := by
  rw [gram_diag]; exact sum_sq_nonneg _

theorem abs_cs (Q : Matrix (Fin m) (Fin l) ℝ) (i j : Fin l) :
    ∑ k, |Q k i| * |Q k j| ≤ Real.sqrt ((Qᵀ * Q) i i) * Real.sqrt ((Qᵀ * Q) j j) := by
  rw [gram_diag, gram_diag]
  have := sum_mul_le_sqrt (fun k => |Q k i|) (fun k => |Q k j|)
  simpa only [sq_abs] using this

/-- The computed diagonal bounds the true one: `(1-γ) Aᵢᵢ ≤ Ĝᵢᵢ`. -/
theorem diag_lower (Q : Matrix (Fin m) (Fin l) ℝ) (G : Matrix (Fin l) (Fin l) ℝ) (γ : ℝ)
    (hG : ∀ i j, |G i j - (Qᵀ * Q) i j| ≤ γ * ∑ k, |Q k i| * |Q k j|) (i : Fin l) :
    (1 - γ) * (Qᵀ * Q) i i ≤ G i i := by
  have h := hG i i
  have hsum : ∑ k, |Q k i| * |Q k i| = (Qᵀ * Q) i i := by
    rw [gram_diag]; apply Finset.sum_congr rfl; intro k _
    rw [← sq, sq_abs]
  rw [hsum] at h
  have := (abs_le.mp h).1
  linarith

theorem G_diag_nonneg (Q : Matrix (Fin m) (Fin l) ℝ) (G : Matrix (Fin l) (Fin l) ℝ) (γ : ℝ) (hγ1 : γ < 1)
    (hG : ∀ i j, |G i j - (Qᵀ * Q) i j| ≤ γ * ∑ k, |Q k i| * |Q k j|) (i : Fin l) : 0 ≤ G i i := by
  have h1 := diag_lower Q G γ hG i
  have h2 := gram_diag_nonneg Q i
  have : 0 ≤ (1 - γ) * (Qᵀ * Q) i i := mul_nonneg (by linarith) h2
  linarith

theorem sqrt_diag_le (Q : Matrix (Fin m) (Fin l) ℝ) (G : Matrix (Fin l) (Fin l) ℝ) (γ : ℝ) (hγ1 : γ < 1)
    (hG : ∀ i j, |G i j - (Qᵀ * Q) i j| ≤ γ * ∑ k, |Q k i| * |Q k j|) (i : Fin l) :
    Real.sqrt (1 - γ) * Real.sqrt ((Qᵀ * Q) i i) ≤ Real.sqrt (G i i) := by
  rw [← Real.sqrt_mul (by linarith)]
  exact Real.sqrt_le_sqrt (diag_lower Q G γ hG i)

/-- Entrywise: `|Ĝᵢⱼ - Aᵢⱼ| ≤ γ/(1-γ) √Ĝᵢᵢ √Ĝⱼⱼ`. -/
theorem gram_entry_bound (Q : Matrix (Fin m) (Fin l) ℝ) (G : Matrix (Fin l) (Fin l) ℝ) (γ : ℝ)
    (hγ0 : 0 ≤ γ) (hγ1 : γ < 1)
    (hG : ∀ i j, |G i j - (Qᵀ * Q) i j| ≤ γ * ∑ k, |Q k i| * |Q k j|) (i j : Fin l) :
    |G i j - (Qᵀ * Q) i j| ≤ γ / (1 - γ) * Real.sqrt (G i i) * Real.sqrt (G j j) := by
  have hpos : 0 < 1 - γ := by linarith
  have hs : 0 < Real.sqrt (1 - γ) := Real.sqrt_pos.mpr hpos
  have hss : Real.sqrt (1 - γ) * Real.sqrt (1 - γ) = 1 - γ := Real.mul_self_sqrt hpos.le
  have hi := sqrt_diag_le Q G γ hγ1 hG i
  have hj := sqrt_diag_le Q G γ hγ1 hG j
  have hai := Real.sqrt_nonneg ((Qᵀ * Q) i i)
  have haj := Real.sqrt_nonneg ((Qᵀ * Q) j j)
  -- √Aᵢᵢ √Aⱼⱼ (1-γ) ≤ √Ĝᵢᵢ √Ĝⱼⱼ
  have hprod : (1 - γ) * (Real.sqrt ((Qᵀ * Q) i i) * Real.sqrt ((Qᵀ * Q) j j))
      ≤ Real.sqrt (G i i) * Real.sqrt (G j j) := by
    have := mul_le_mul hi hj (mul_nonneg hs.le haj) (Real.sqrt_nonneg _)
    have e : (1 - γ) * (Real.sqrt ((Qᵀ * Q) i i) * Real.sqrt ((Qᵀ * Q) j j))
        = Real.sqrt (1 - γ) * Real.sqrt ((Qᵀ * Q) i i) * (Real.sqrt (1 - γ) * Real.sqrt ((Qᵀ * Q) j j)) := by
      linear_combination (-(Real.sqrt ((Qᵀ * Q) i i) * Real.sqrt ((Qᵀ * Q) j j))) * hss
    rw [e]; exact this
  calc |G i j - (Qᵀ * Q) i j|
      ≤ γ * ∑ k, |Q k i| * |Q k j| := hG i j
    _ ≤ γ * (Real.sqrt ((Qᵀ * Q) i i) * Real.sqrt ((Qᵀ * Q) j j)) :=
        mul_le_mul_of_nonneg_left (abs_cs Q i j) hγ0
    _ = γ / (1 - γ) * ((1 - γ) * (Real.sqrt ((Qᵀ * Q) i i) * Real.sqrt ((Qᵀ * Q) j j))) := by
        field_simp
    _ ≤ γ / (1 - γ) * (Real.sqrt (G i i) * Real.sqrt (G j j)) :=
        mul_le_mul_of_nonneg_left hprod (div_nonneg hγ0 hpos.le)
    _ = γ / (1 - γ) * Real.sqrt (G i i) * Real.sqrt (G j j) := by ring

/-- `‖QᵀQ - I‖_F ≤ ‖Ĝ - I‖_F + γ/(1-γ) tr Ĝ`. -/
theorem frob_bound (Q : Matrix (Fin m) (Fin l) ℝ) (G : Matrix (Fin l) (Fin l) ℝ) (γ : ℝ)
    (hγ0 : 0 ≤ γ) (hγ1 : γ < 1)
    (hG : ∀ i j, |G i j - (Qᵀ * Q) i j| ≤ γ * ∑ k, |Q k i| * |Q k j|) :
    frob (Qᵀ * Q - 1) ≤ frob (G - 1) + γ / (1 - γ) * ∑ i, G i i := by
  unfold frob
  have hsplit : (fun p : Fin l × Fin l => (Qᵀ * Q - 1 : Matrix (Fin l) (Fin l) ℝ) p.1 p.2)
      = (fun p : Fin l × Fin l => (G - 1 : Matrix (Fin l) (Fin l) ℝ) p.1 p.2)
        + (fun p : Fin l × Fin l => (Qᵀ * Q) p.1 p.2 - G p.1 p.2) := by
    funext p; simp [Matrix.sub_apply]
  rw [hsplit]
  refine le_trans (l2_triangle _ _) (add_le_add le_rfl ?_)
  have hc : 0 ≤ γ / (1 - γ) := div_nonneg hγ0 (by linarith)
  rw [← l2_outer (fun i => G i i) (G_diag_nonneg Q G γ hγ1 hG) _ hc]
  apply l2_mono
  intro p
  rw [abs_sub_comm]
  exact gram_entry_bound Q G γ hγ0 hγ1 hG p.1 p.2

/-- The certificate: with the rounding of the norm, the trace and the final multiply-add as hypotheses,
`‖QᵀQ - I‖_F ≤ v`. -/
theorem certificate (Q : Matrix (Fin m) (Fin l) ℝ) (G : Matrix (Fin l) (Fin l) ℝ)
    (γ κ τ ρ α β e t v : ℝ) (hγ0 : 0 ≤ γ) (hγ1 : γ < 1)
    (hG : ∀ i j, |G i j - (Qᵀ * Q) i j| ≤ γ * ∑ k, |Q k i| * |Q k j|)
    (hκ : 0 < κ) (he : κ * frob (G - 1) ≤ e)
    (hτ : 0 < τ) (ht : τ * ∑ i, G i i ≤ t)
    (hv : ρ * (α * e + β * t) ≤ v)
    (hα : 1 ≤ ρ * κ * α) (hβ : γ / (1 - γ) ≤ ρ * τ * β) :
    frob (Qᵀ * Q - 1) ≤ v := by
  have hF := frob_bound Q G γ hγ0 hγ1 hG
  have htr : 0 ≤ ∑ i, G i i := Finset.sum_nonneg (fun i _ => G_diag_nonneg Q G γ hγ1 hG i)
  have hfrob0 : 0 ≤ frob (G - 1) := l2_nonneg _
  have hc : 0 ≤ γ / (1 - γ) := div_nonneg hγ0 (by linarith)
  -- ‖Ĝ - I‖_F ≤ ρ α e
  have hα0 : 0 ≤ ρ * α := by
    by_contra hneg
    have hneg := lt_of_not_ge hneg
    have : ρ * κ * α = κ * (ρ * α) := by ring
    nlinarith
  have h1 : frob (G - 1) ≤ ρ * α * e := by
    have : frob (G - 1) ≤ ρ * κ * α * frob (G - 1) := by nlinarith
    have h' : ρ * κ * α * frob (G - 1) = ρ * α * (κ * frob (G - 1)) := by ring
    nlinarith [mul_le_mul_of_nonneg_left he hα0]
  -- γ/(1-γ) tr Ĝ ≤ ρ β t
  have h2 : γ / (1 - γ) * ∑ i, G i i ≤ ρ * β * t := by
    have hβ0 : 0 ≤ ρ * β := by
      have h0 : 0 ≤ ρ * τ * β := le_trans hc hβ
      by_contra hneg
      have hneg := lt_of_not_ge hneg
      have : τ * (ρ * β) < 0 := mul_neg_of_pos_of_neg hτ hneg
      have h' : ρ * τ * β = τ * (ρ * β) := by ring
      linarith
    calc γ / (1 - γ) * ∑ i, G i i ≤ ρ * τ * β * ∑ i, G i i := mul_le_mul_of_nonneg_right hβ htr
      _ = ρ * β * (τ * ∑ i, G i i) := by ring
      _ ≤ ρ * β * t := mul_le_mul_of_nonneg_left ht hβ0
  calc frob (Qᵀ * Q - 1) ≤ frob (G - 1) + γ / (1 - γ) * ∑ i, G i i := hF
    _ ≤ ρ * α * e + ρ * β * t := add_le_add h1 h2
    _ = ρ * (α * e + β * t) := by ring
    _ ≤ v := hv

/-- Certification: if the computed `v` passes `v ≤ tol`, the true orthogonality error is at most `tol`. -/
theorem certified (Q : Matrix (Fin m) (Fin l) ℝ) (G : Matrix (Fin l) (Fin l) ℝ)
    (γ κ τ ρ α β e t v tol : ℝ) (hγ0 : 0 ≤ γ) (hγ1 : γ < 1)
    (hG : ∀ i j, |G i j - (Qᵀ * Q) i j| ≤ γ * ∑ k, |Q k i| * |Q k j|)
    (hκ : 0 < κ) (he : κ * frob (G - 1) ≤ e)
    (hτ : 0 < τ) (ht : τ * ∑ i, G i i ≤ t)
    (hv : ρ * (α * e + β * t) ≤ v)
    (hα : 1 ≤ ρ * κ * α) (hβ : γ / (1 - γ) ≤ ρ * τ * β) (hcert : v ≤ tol) :
    frob (Qᵀ * Q - 1) ≤ tol :=
  le_trans (certificate Q G γ κ τ ρ α β e t v hγ0 hγ1 hG hκ he hτ ht hv hα hβ) hcert

/-- `γ_m / (1 - γ_m) = m u / (1 - 2 m u)` for `γ_m = m u / (1 - m u)`, `2 m u < 1`. -/
theorem gamma_ratio (x : ℝ) (hx : 2 * x < 1) :
    (x / (1 - x)) / (1 - x / (1 - x)) = x / (1 - 2 * x) := by
  have h1 : (1 - x) ≠ 0 := by linarith
  have h2 : (1 - 2 * x) ≠ 0 := by linarith
  have h3 : 1 - x / (1 - x) = (1 - 2 * x) / (1 - x) := by field_simp; ring
  rw [h3]; field_simp

/-! ## The implementation's constants satisfy `hα`, `hβ`

`γ k = k u / (1 - k u)`. With `κ = (1-u)²(1-γ_{l²})`, `τ = 1-γ_l`, `ρ = (1-u)²` and the constants of
`cholqr.core._cert_constants` (`α ≥ 1/((1-u)⁴(1-γ_{l²}))`, `β ≥ γ_m/(1-γ_m)/((1-γ_l)(1-u)²)`), the hypotheses
`hα : 1 ≤ ρ κ α`, `hβ : γ_m/(1-γ_m) ≤ ρ τ β` of `certificate` hold, provided `2 l² u < 1`, `2 m u < 1`, `1 ≤ l`. -/

/-- `γ k < 1` when `2 k u < 1`. -/
theorem gamma_lt_one (x : ℝ) (hx : 2 * x < 1) : x / (1 - x) < 1 := by
  have h : 0 < 1 - x := by linarith
  rw [div_lt_one h]; linarith

theorem constants_ok (u L M α β : ℝ) (hu : 0 < u) (hu1 : u < 1) (hL : 1 ≤ L) (hM : 0 ≤ M)
    (hLsize : 2 * (L ^ 2 * u) < 1) (hMsize : 2 * (M * u) < 1)
    (hα : 1 / ((1 - u) ^ 4 * (1 - (L ^ 2 * u) / (1 - L ^ 2 * u))) ≤ α)
    (hβ : ((M * u) / (1 - M * u)) / (1 - (M * u) / (1 - M * u))
          / ((1 - (L * u) / (1 - L * u)) * (1 - u) ^ 2) ≤ β) :
    1 ≤ (1 - u) ^ 2 * ((1 - u) ^ 2 * (1 - (L ^ 2 * u) / (1 - L ^ 2 * u))) * α ∧
    ((M * u) / (1 - M * u)) / (1 - (M * u) / (1 - M * u))
      ≤ (1 - u) ^ 2 * (1 - (L * u) / (1 - L * u)) * β ∧
    0 ≤ (M * u) / (1 - M * u) ∧ (M * u) / (1 - M * u) < 1 := by
  have hL2 : 0 ≤ L ^ 2 * u := by positivity
  have hLu0 : 0 ≤ L * u := by nlinarith
  have hLu : 2 * (L * u) < 1 := by nlinarith
  have hMu0 : 0 ≤ M * u := by positivity
  have g2 := gamma_lt_one _ hLsize
  have gl := gamma_lt_one _ hLu
  have hpu : 0 < (1 - u) ^ 2 := by have : 0 < 1 - u := by linarith
                                   positivity
  refine ⟨?_, ?_, div_nonneg hMu0 (by linarith), gamma_lt_one _ hMsize⟩
  · have hK : 0 < (1 - u) ^ 4 * (1 - (L ^ 2 * u) / (1 - L ^ 2 * u)) :=
      mul_pos (by have : 0 < 1 - u := by linarith
                  positivity) (by linarith)
    have := (div_le_iff₀ hK).mp hα
    calc (1 : ℝ) ≤ α * ((1 - u) ^ 4 * (1 - (L ^ 2 * u) / (1 - L ^ 2 * u))) := this
      _ = (1 - u) ^ 2 * ((1 - u) ^ 2 * (1 - (L ^ 2 * u) / (1 - L ^ 2 * u))) * α := by ring
  · have hT : 0 < (1 - (L * u) / (1 - L * u)) * (1 - u) ^ 2 := mul_pos (by linarith) hpu
    have := (div_le_iff₀ hT).mp hβ
    calc ((M * u) / (1 - M * u)) / (1 - (M * u) / (1 - M * u))
        ≤ β * ((1 - (L * u) / (1 - L * u)) * (1 - u) ^ 2) := this
      _ = (1 - u) ^ 2 * (1 - (L * u) / (1 - L * u)) * β := by ring

end CholQR.Verify
