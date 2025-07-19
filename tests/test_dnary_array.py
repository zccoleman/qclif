import numpy as np
from qclif._qclifBase.dnary_array import DnaryArrayBase


class D5(DnaryArrayBase):d=5
class D3(DnaryArrayBase):d=3

def test_invertible():
    assert D3.eye(2).is_invertible()
    assert not D3.zeros((2,2)).is_invertible()

def test_random_inverse():
    for _ in range(10):
        t=D3.random_array((12,12))
        s=D5.random_array((16,16))
        t_inv = t.mod_matrix_inv()
        s_inv = s.inv()
        if t_inv is not None:
            assert np.array_equal(t @ t_inv, D3.eye(12))
        if s_inv is not None:
            assert np.array_equal(s @ s_inv, D5.eye(16))


def test_composite_dnary():
    class D4(DnaryArrayBase):d=4
    
