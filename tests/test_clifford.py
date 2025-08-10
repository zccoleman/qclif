import pytest
import numpy as np

from qclif import CliffordBase


CD2 = CliffordBase.set_d(2)
CD3 = CliffordBase.set_d(3)
CD5 = CliffordBase.set_d(5)
CD13 = CliffordBase.set_d(13)

def test_clifford_decomposition():
    for _ in range(20):
        assert CD3.random_symplectic(1).decompose()
        assert CD5.random_symplectic(5).decompose()
        assert CD2.random_symplectic(2).decompose()
        assert CD2.random_symplectic(9).decompose()