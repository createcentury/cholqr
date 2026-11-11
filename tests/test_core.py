import torch




def test_chunked_gram_matches_matmul():
    import torch
    import cholqr.core as core
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
    A = torch.randn(64, 2048, generator=g); B = torch.randn(2048, 64, generator=g)
    exact = A.double() @ B.double()
    naive = (tf32mm(A, B).double() - exact).norm() / exact.norm()
    Ah, Al = _tf32_split(A); Bh, Bl = _tf32_split(B)
    three = tf32mm(Ah, Bh) + (tf32mm(Ah, Bl) + tf32mm(Al, Bh))
    err3 = (three.double() - exact).norm() / exact.norm()
    assert naive > 1e-4 and err3 < 1e-6 and err3 < naive / 100
