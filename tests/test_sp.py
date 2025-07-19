import pytest
import numpy as np

from qclif import CliffordBase, Transvection

class CD2(CliffordBase):
    d=2

class CD3(CliffordBase):
    d=3

class CD5(CliffordBase):
    d=5

class CD13(CliffordBase):
    d=13

def test_attributes_persist():
    u = CD5([1,0,0,0])
    v = CD5([1,0,0,0])
    result = v ^ u
    assert result.d==5
    assert result.n==2
    assert result.nn==4

    assert u.is_vector
    assert not u.is_matrix
    assert v.is_vector
    assert not v.is_matrix
    assert v.Lambda.is_matrix
    assert not v.Lambda.is_vector

def test_standard_basis():
    u = CD3([1,0,0,0])
    v = CD3.basis_vector(0,4)
    assert np.array_equal(u,v)

def test_odd_dimension():
    size3 = CD3([1,2,3])

    with pytest.raises(ValueError):
        size3.Lambda

    with pytest.raises(ValueError):
        size3.inner_product_with(size3)

def test_even_dimension():
    size2 = CD3([1,2])
    assert np.array_equal(size2.Lambda, [[0,2],[1,0]])
    
def test_dnary_logic():
    one = CD3([1])
    two = CD3([2])
    five = CD3([5])

    assert (one + two).item()==0
    assert five.item()==two.item()
    assert (two+five).item()==one.item()

def test_inner_product():
    u = CD3([1,0,0,0])
    result = u | [0,0,1,0]

    neg1 = CD3([-1])

    assert result == neg1.item()

    with pytest.raises(ValueError):
        u | [1,2,3,4,5,6]

    assert CD5([2, 1, 0, 3, 4, 1]).is_symplectic_pair_with(CD5([3, 0, 1, 3, 0, 2]))

def test_symplectic_pairs():
    u = CD3([1,0,0,0])
    v = CD3([0,0,1,0])
    assert CD3.is_symplectic_pair(u, v)

def test_symplectic_matrix():
    assert CD5.LambdaN(5).is_symplectic_matrix()
    assert CD5([[4, 2, 0, 0, 0, 3],
                [2, 4, 1, 0, 4, 1],
                [2, 0, 0, 2, 1, 2],
                [1, 4, 0, 3, 2, 2],
                [2, 2, 3, 1, 0, 0],
                [3, 0, 3, 4, 4, 4]],
                ).is_symplectic_matrix()

def test_embedding():
    assert CD5([[4, 2, 0, 0, 0, 3],
                [2, 4, 1, 0, 4, 1],
                [2, 0, 0, 2, 1, 2],
                [1, 4, 0, 3, 2, 2],
                [2, 2, 3, 1, 0, 0],
                [3, 0, 3, 4, 4, 4]],
                ).embed_symplectic().is_symplectic_matrix()
    
    t = CD13.random_array((6,6))
    embed = t.embed_symplectic()

    assert embed.n==4
    assert embed.d==13
    assert embed.nn==8

def test_transvection():
    u = CD3([1,0,0,0])
    v = CD3([1,2,3,4])

    T = Transvection(v, c=5)
    result = T(u)
    
    assert isinstance(result, CD3)

def test_transvection_finding():
    u = CD5([1,2,3,4])
    v = CD5([2,3,4,5])

    t1, t2 = Transvection.find_transvection(u, v)
    assert np.array_equal(t2(t1(u)), v)




def test_random_symplectic():
    for i in range(50):
        for n in [1,2,3,4]:
            assert CD5.random_symplectic(n).is_symplectic_matrix()

def test_index_symplectic():
    assert CD3.from_index(100000, n=3).is_symplectic_matrix()

    assert np.array_equal(
        CD5.from_index(100000, n=3),
        CD5([[2, 0, 3, 4, 3, 0],
            [1, 1, 4, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 3, 2, 0],
            [0, 0, 0, 0, 1, 0],
            [2, 0, 3, 0, 3, 1]]),
    )
    
def test_clifford_decomposition():
    for _ in range(20):
        assert CD3.random_symplectic(1).decompose()
        assert CD5.random_symplectic(5).decompose()
        assert CD2.random_symplectic(2).decompose()
        assert CD2.random_symplectic(9).decompose()