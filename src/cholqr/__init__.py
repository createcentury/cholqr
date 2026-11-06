"""cholqr: fast, compile-friendly shifted CholeskyQR3 for batched tall-skinny matrices in PyTorch.

Replaces ``torch.linalg.qr(Y, mode="reduced")[0]`` when only an orthonormal basis of ``range(Y)`` is needed
(randomized SVD / sketching, low-rank optimizers such as randomized Muon, subspace iteration).

Algorithm: shifted CholeskyQR3 (Fukaya, Kannan, Nakatsukasa, Yamamoto, Yanagisawa, SIAM J. Sci. Comput. 42(1), 2020)
built on CholeskyQR2 (Yamamoto, Nakatsukasa, Yanagisawa, Fukaya, ETNA 44, 2015).
"""
from .core import cholesky_qr, orthonormal_basis

__all__ = ["cholesky_qr", "orthonormal_basis"]
__version__ = "0.2.0"
