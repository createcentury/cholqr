import CholQR
-- Axiom check (not part of `lake build`): `lake env lean Check.lean`
-- Expected for every theorem: [propext, Classical.choice, Quot.sound]
open CholQR.ShiftedPass CholQR.TheoremD
#print axioms gram_Q1
#print axioms T0_resolvent_eig
#print axioms T0_eigvec
#print axioms T0_eigvec_ne_zero
#print axioms T0_similar
#print axioms lemmaI_quadratic
#print axioms lemmaI_posDef
#print axioms D_a_identity
#print axioms D_a_quad
#print axioms D_a_bound
#print axioms D_b_LLt
#print axioms D_b_inv
#print axioms D_b_bound
#print axioms cauchy_schwarz_dot
#print axioms D_c_bound
open CholQR.Verify
#print axioms l2_triangle
#print axioms l2_mono
#print axioms l2_outer
#print axioms gram_entry_bound
#print axioms frob_bound
#print axioms certificate
#print axioms certified
#print axioms gamma_ratio
#print axioms gamma_lt_one
#print axioms constants_ok
open CholQR.D1
#print axioms col_sq_bounds
#print axioms scale_facts
#print axioms D1_scaled
#print axioms D1_delta
#print axioms D1_sharp
#print axioms D1_paper
