

def test_chunked_gram_matches_matmul():
    import torch
    from cholqr.core import _gram
    for m in (100, 512, 513, 6144, 6150):
        Y = torch.randn(3, m, 17, dtype=torch.float64)
        assert torch.allclose(_gram(Y), Y.mT @ Y, rtol=1e-12, atol=1e-10)
