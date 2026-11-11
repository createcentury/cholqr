"""Size-dependent last-pass breakdown threshold, frozen before any run of benchmarks/thresh_size.py
(benchmarks/prereg/PREREG_THRESH_SIZE_2026-11-12.md). Constants from benchmarks/thresh_size_fit.py on existing data
(QRROUND3, l = 64 / 276; THRESH_WIDE, l = 512 / 1024), rounded to 3 significant digits.

    tau(l) = min(TAU_MAX, max(TAU_MIN, sqrt(C(l) u / TOL))),   C(l) = C_A * l ** C_P

flag a matrix if the smallest diagonal entry of the last pass's Cholesky factor is below tau(l) (l = number of columns),
or if any pass failed (info > 0 or a non-finite factor).
"""
import math

U = 2.0 ** -24
TOL = 1e-4
C_A = 1.01          # margin 1.5 x least-squares power-law fit of the per-size maximum of c (A = 0.6725)
C_P = 0.868         # exponent of that fit
TAU_MIN = math.sqrt(64 * U / TOL)   # 0.1953: the validated cholqr 0.5 threshold (never flag less than it)
TAU_MAX = 0.75


def tau(l):
    return min(TAU_MAX, max(TAU_MIN, math.sqrt(C_A * l ** C_P * U / TOL)))


if __name__ == "__main__":
    for l in (64, 128, 276, 384, 512, 768, 1024, 1536, 2048, 4096):
        print(f"l = {l:5d}: C = {C_A * l ** C_P:7.1f}, tau = {tau(l):.4f}")
