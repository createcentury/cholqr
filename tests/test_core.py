import torch




def test_chunked_gram_matches_matmul():
    import torch
    from cholqr.core import _gram
    for m in (100, 512, 513, 4096, 6144, 6150):
        Y = torch.randn(3, m, 17, dtype=torch.float64)
        assert torch.allclose(_gram(Y), Y.mT @ Y, rtol=1e-12, atol=1e-10)


def _tf32_round(X):
    b = X.contiguous().view(torch.int32)
    return ((b + 0x1000) & ~0x1FFF).view(torch.float32)


def test_tf32_split_exact():
    from cholqr.core import _tf32_split
    A = torch.randn(300, 70, generator=torch.Generator().manual_seed(0))
    hi, lo = _tf32_split(A)
    assert torch.equal(hi + lo, A)                                  # exact split
    assert torch.equal(_tf32_round(hi), hi)                         # hi representable in TF32


def test_3xtf32_restores_fp32_accuracy_emulated():
    # emulate a TF32 GEMM on CPU: inputs rounded to TF32, products/accumulation in high precision
    from cholqr.core import _tf32_split
    def tf32mm(X, Y):
        return (_tf32_round(X).double() @ _tf32_round(Y).double()).float()
    g = torch.Generator().manual_seed(1)
    A = torch.randn(64, 2048, generator=g)
    B = torch.randn(2048, 64, generator=g)
    exact = A.double() @ B.double()
    naive = (tf32mm(A, B).double() - exact).norm() / exact.norm()
    Ah, Al = _tf32_split(A)
    Bh, Bl = _tf32_split(B)
    three = tf32mm(Ah, Bh) + (tf32mm(Ah, Bl) + tf32mm(Al, Bh))
    err3 = (three.double() - exact).norm() / exact.norm()
    assert naive > 1e-4 and err3 < 1e-6 and err3 < naive / 100


def test_tf32_split_edge_cases():
    from cholqr.core import _tf32_split
    A = torch.tensor([3.40199e38, -3.4028e38, float("inf"), -float("inf"), 1e-45, -2e-40, 0.0, 1.0, -1.0000001])
    hi, lo = _tf32_split(A)
    fin = torch.isfinite(A)
    assert torch.equal((hi + lo)[fin], A[fin])                       # exact for every finite input, incl. near overflow
    assert torch.isfinite(hi[fin]).all() and torch.isfinite(lo[fin]).all()
    assert torch.equal(hi[~fin], A[~fin])                            # infinities pass through


def test_3xtf32_gram_truncating_emulation():
    # tensor cores may truncate (not round) TF32 inputs: hi stays exact, lo loses up to 2t; bound 5 t^2 = 20u
    from cholqr.core import _tf32_split
    def trunc(X):
        return (X.contiguous().view(torch.int32) & ~0x1FFF).view(torch.float32)
    g = torch.Generator().manual_seed(2)
    Q = torch.randn(4096, 64, generator=g)
    Q = Q / Q.norm(dim=0)
    exact = Q.double().mT @ Q.double()
    Ah, Al = _tf32_split(Q)
    M = trunc(Ah).double().mT @ trunc(Al).double()
    G3 = trunc(Ah).double().mT @ trunc(Ah).double() + (M + M.mT)
    err = ((G3 - exact).abs() / (Q.double().abs().mT @ Q.double().abs())).max()
    assert err <= 5 * 2.0 ** -22 * 1.01


def test_r_products_use_compensated_mm_under_tf32(monkeypatch):
    """Regression (0.5.1): with TF32 active, the R accumulation must go through _mm (3xTF32), like Q.
    Before the fix, F @ R was a plain @, so R had TF32-level accuracy under precision "high" (Q was unaffected)."""
    import torch
    from cholqr import core
    calls = []
    real_mm = core._mm

    def spy(A, B, tf32):
        calls.append((tuple(A.shape), tuple(B.shape), tf32))
        return real_mm(A, B, False)          # exact FP32 on CPU; only the routing is tested

    monkeypatch.setattr(core, "_mm", spy)
    monkeypatch.setattr(core, "_tf32_active", lambda Q: True)
    Y = torch.randn(3, 256, 16, generator=torch.Generator().manual_seed(0))
    Q, R = core.cholesky_qr(Y, return_r=True)
    rr = [c for c in calls if c[0][-2:] == (16, 16) and c[1][-2:] == (16, 16)]
    assert len(rr) >= 2 and all(c[2] for c in rr)       # r x r products of the passes after the first
    assert torch.allclose(Q @ R, Y, rtol=1e-4, atol=1e-4)
