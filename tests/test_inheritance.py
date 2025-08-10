import pytest

from qclif import DnaryArray


def test_needs_d():
    with pytest.raises(TypeError):
        t = DnaryArray([0])

def test_can_create_child():
    class Child(DnaryArray):
        ...

def test_child_needs_d():
    class Child(DnaryArray):
        ...
    with pytest.raises(TypeError):
        Child([0])

def test_second_child_needs_d():
    class Child(DnaryArray):
        ...
    class Child2(Child):
        ...
    with pytest.raises(TypeError):
        Child2([0])

def test_cannot_reassign_classvar():
    class Child(DnaryArray):
        d=5
    with pytest.raises(AttributeError):
        Child.d=10

def test_cannot_reassign_instance_var():
    class Child(DnaryArray):
        d=5
    t=Child([0])
    with pytest.raises(AttributeError):
        t.d=10

def test_requires_prime():
    with pytest.raises(ValueError):
        class D4(DnaryArray):
            _validate_prime=True
            d=4
    