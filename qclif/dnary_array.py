from __future__ import annotations
import numpy as np
from typing import Self, Any 
import warnings

from abc import ABCMeta, abstractmethod

from qclif.dnary_arithmetic import rint, dnary_inverse, int_to_dnary, dnary_to_int
from qclif.validation import validate_primes

def d_get(cls):
    return cls._d
def d_set(cls, value):
    raise AttributeError('You cannot change d')
def d_del(cls):
    raise AttributeError('You cannot delete d')
dprop = property(d_get, d_set, d_del, 'The modulus of the set of matrices.')

class DNaryMeta(ABCMeta):
    """A metaclass for d-nary arrays that handles property creation for the class modulus d as an immutable class property.
    """
    def __new__(cls, clsname, bases, attrs):
        if 'd' in attrs:
            d = attrs.pop('d')
            
            if (validate_prime:=attrs.get('_validate_prime')) is not None:
                pass
            else:
                for base in bases:
                    if (validate_prime:=getattr(base, '_validate_prime', None)) is not None:
                        break
                else:
                    validate_prime=False

            if validate_prime:
                validate_primes(d)
            attrs['_d'] = d
            attrs['d'] = dprop
        return super().__new__(cls, clsname, bases, attrs)
    d=dprop
    
    def check_or_coerce_type(cls, u:Any) -> Self:
        """Coerces the input into the type of the current class.

        Args:
            u (Any): A symplectic array-like object to be coerced into the current class.

        Raises:
            TypeError: The object could not be typecast as the new type.

        Returns:
            Self: The input object cast into the new type.
        """
        if not isinstance(u, cls):
            print(f'Coercing type on {u} {type(u)} to {cls}')
            try:
                u = cls(u)
            except Exception as e:
                raise TypeError("Input must be symplectic array-like", e, u)
        return u
    

class DnaryArray(np.ndarray, metaclass=DNaryMeta):
    r"""A $d$-nary integer `numpy` array. Use as a normal `numpy`,
    but all outputs will be computed using arithmetic in $\mathbb{Z}_d$.
    
    Do not directly create instances of this class! Instead, use the classmethod 
    [`DnaryArray.set_d(...)`][qclif.DnaryArray.set_d] to create
    a subclass with a specific value for `d`, then create instances of that subclass.
    
    Instances can be created from any array-like data acceptable by `np.array(...)`.

    Examples:
    ```
    >>> D3 = DnaryArray.set_d(3)
    >>> D3([1, 2, 3, 4])
    DnaryArray(d=3)([1, 2, 0, 1])

    >>> D3.eye(4)
    D3([[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    >>> D3.eye(4)-2
    D3([[2, 1, 1, 1],
        [1, 2, 1, 1],
        [1, 1, 2, 1],
        [1, 1, 1, 2]])

    >>> (D3.eye(4)-2).inverse()
    D3([[2, 1, 1, 1],
        [1, 2, 1, 1],
        [1, 1, 2, 1],
        [1, 1, 1, 2]])

    >>> import numpy as np
    >>> np.linalg.matrix_power(D3.eye(4)-2, 2)
    D3([[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    
    >>> MyD6Array = DnaryArray.set_d(6)
    >>> (2 * MyD6Array.eye(2)) @ [2,3]
    MyD6Array([4, 0])
    ```
    """
    
    _validate_prime=False
    @property
    @abstractmethod
    def d(self):
        """The modulus of the array class.
        Should not be changed after class creation.
        """
        raise NotImplementedError
    
    def __new__(cls, data: Any, *args, **kwargs) -> Self:
        if not isinstance(cls.d, int):
            raise TypeError('Cannot instantiate base class without a definition for d.')
        d = cls.d
        array = np.array(data, dtype=int) % d

        obj = array.view(cls)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        if not issubclass(self.dtype.type, np.integer):
            raise ValueError('Would result in array with non-integer values', self)
        
        if self.ndim==1:
            # nn = len(self)
            pass
        elif self.ndim==2:
            l, w = self.shape
            if not l==w:
                raise ValueError('Would result in array or vector of incorrect size', self)
            # nn = l
        else: raise ValueError('Would result in array or vector of incorrect size', self)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # print(f"In __array_ufunc__ with self {self} and inputs {inputs}")
        ds = set([input.d for input in inputs if isinstance(input, type(self))])
        inputs_ = []
        ds = set()
        for input_ in inputs:
            if isinstance(input_, type(self)):
                inputs_.append(input_.view(np.ndarray))
                ds.add(input_.d)
            else:
                inputs_.append(input_)

        if len(ds)>1:
            raise TypeError('Cannot compute with arrays with different moduli:', ds)
        d = ds.pop()
        result = super().__array_ufunc__(ufunc, method, *inputs_, **kwargs)
        if result is NotImplemented:
            return NotImplemented 
        
        if result.ndim==0:
            return result.item() % d
         
        result_obj = np.asarray(result % d).view(type(self))
        
        # result_obj.d = d
        return result_obj


    @property
    def is_matrix(self) -> bool:
        """
        Returns:
            Whether this is a 2D array.
        """
        return self.ndim==2
    
    @property
    def is_vector(self) -> bool:
        """
        Returns:
            Whether this is a 1D array.
        """
        return self.ndim==1
    
    @classmethod
    def dnary_inverse(cls, n: int) -> int:
        r"""
        Alias for [`dnary_inverse(n, cls.d)`][qclif.dnary_arithmetic.dnary_inverse].
        Calculates the multiplicative inverse of `n` in $\mathbb{Z}_d$ if it exists.

        If $d$ is prime and $n>0$, the integer is guaranteed to exist.
        For general $d$, $n^{-1}$ exists if $\text{gcd}(n,d)=1$.

        Args:
            n (int): The integer to invert.

        Returns:
            The integer $n^{-1}$ such that $n n^{-1} \text{ mod } d = 1$, 
                or `None` if such an integer does not exist.
                
        """
        return dnary_inverse(n, cls.d)
    
    @classmethod
    def int_to_dnary(cls, n:int, result_list_size:int|None=None) -> list[int]:
        r"""Alias for [`int_to_dnary(n, cls.d)`][qclif.dnary_arithmetic.int_to_dnary].
        
        Decomposes a base 10 integer into a list of its $d$-nary digits.

        See also [qclif.DnaryArray.dnary_to_int].

        Args:
            n (int): The base 10 integer to be factored into its $d$-nary digits.
            result_list_size (int | None, optional): The length of the
                list to return; if `None`, will use the minimum number
                of digits to result the input integer in $d$-nary.
                Defaults to `None`.

        Raises:
            ValueError: Invalid input parameters.

        Returns:
            A list of the $d$-nary digits of `n`
                The list has $L$ elements $l_i$ such that $$n=\sum_{i=0}^L l_i d^i.$$
        """
        return int_to_dnary(n, cls.d, result_list_size)
    
    @classmethod
    def dnary_to_int(cls, digits) -> int:
        """Alias for [`int_to_dnary(n, cls.d)`][qclif.dnary_arithmetic.dnary_to_int].
        Given a list of the $d$-nary digits of a number, return that
        number in base 10. 

        See also [qclif.DnaryArray.int_to_dnary].

        Args:
            digits (list[int]): A list of the $d$-nary digits of a
                number.

        Returns:
            The number in base 10.
        """
        return dnary_to_int(digits, cls.d)
    
    def is_nonzero(self) -> bool:
        """
        Returns:
            Whether the array has any nonzero values in it.
        """
        return bool(self.any())
    
    def is_zero(self) -> bool:
        """
        Returns:
            Whether the array is all zero.
        """
        return not self.is_nonzero()


    @classmethod
    def random_array(cls, shape:tuple[int], allow_zero=True) -> Self:
        """Returns an array of the given shape with uniform
        random integers mod $d$, including the all-zero array
        unless specified otherwise.

        Args:
            shape (tuple[int]): The shape of the array.
            allow_zero (bool, optional): Whether to allow the zero 
                matrix. Defaults to `True`.

        Returns:
            A uniform random $d$-nary array.
        """
        if allow_zero:
            return cls(np.random.randint(0, cls.d, shape))
        for _ in range(1000):
            a = cls.random_array(shape, allow_zero=True)
            if a.is_nonzero():
                return a
        raise RuntimeError('Could not create a non-zero random matrix.')
    
    @classmethod
    def eye(cls, n: int) -> Self:
        """
        Returns:
            The `(n, n)` identity matrix with integer type.
        """
        return cls(np.identity(n, dtype='int32'))
    
    @classmethod
    def zeros(cls, shape:int|tuple[int]) -> Self:
        """Returns an all-zero array of the given shape.

        Args:
            shape (int|tuple[int]): The shape of the desired array.

        Returns:
            The zero array of the given shape.
        """
        return cls(np.zeros(shape, dtype='int32'))
    
    @classmethod
    def basis_vector(cls, i: int, n: int) -> Self:
        """Generates the $i$th standard basis vector of
        length $n$. This is the vector of all zeros except
        the $i$th component equal to $1$.

        Args:
            i (int): The component of the basis vector.
            n (int): The length of the vector

        Returns:
            The $i$th standard basis vector.

        Examples:
        ```
        >>> class D3(DnaryArray): d=3
        >>> D3.basis_vector(1, 5)
        D3([0, 1, 0, 0, 0])
        """
        e_i = cls.zeros(n)
        e_i[i]=1
        return e_i
    
    def determinant(self) -> int:
        """
        Returns:
            The determinant of the array modulo d.
        """
        return rint(np.linalg.det(self)) % self.d
    
    def det(self) -> int:
        """Alias for [`.determinant`][qclif.DnaryArray.determinant].

        Returns:
            The determinant of the array modulo $d$.
        """
        return self.determinant()
    
    def is_invertible(self) -> bool:
        r"""Determines if an array is invertible by checking if
        its determinant is invertible in $\mathbb{Z}_d$.

        Returns:
            Whether the array is invertible.
        """
        return self.dnary_inverse(self.det()) is not None
    
    def mod_matrix_inv(self, validate=True, suppress_warnings=True) -> Self|None:
        r"""Calculates the matrix inverse in $\mathbb{Z}_d$-arithmetic
        using
        $$A^{-1}=\frac{1}{|A|}\text{Adj}(A),$$
        where $\text{Adj}(A)$ is the
        [matrix adjugate](https://en.wikipedia.org/wiki/Adjugate_matrix)
        of $A$ and $|A|$ is the determinant.

        Args:
            validate (bool, optional): Whether to validate the
                result through multiplication. Defaults to True.
            suppress_warnings (bool, optional): If False,
                warnings will be raise if the array is not
                invertible. Defaults to True.

        Raises:
            RuntimeError: If the inversion fails for an invertible
                array. Theoretically this should never happen.

        Returns:
            The array's inverse, or `None` if the array
                is not invertible.
        """
        
        det = self.determinant()

        if not(det_inv:=self.dnary_inverse(det)):
            if not suppress_warnings: 
                warnings.warn("Tried to invert non-invertible matrix", RuntimeWarning, stacklevel=2, source=self)
            return None
        
        inverse = det_inv * self._mod_matrix_adjugate()

        if validate:
            if not np.array_equal(inverse @ self, self.eye(len(self))):
                raise RuntimeError("Matrix inversion failed with non-zero invertible determinant.", self, inverse, det)
        
        return inverse

    def inv(self, **kwargs) -> Self|None:
        """Alias for [`mod_matrix_inv`][qclif.DnaryArray.mod_matrix_inv].

        Returns:
            The array's inverse, or `None` if
                the array is not invertible.
        """
        return self.mod_matrix_inv(**kwargs)
    
    def inverse(self, **kwargs) -> Self|None:
        """Alias for [`mod_matrix_inv`][qclif.DnaryArray.mod_matrix_inv].

        Returns:
            The array's inverse, or `None` if the
                array is not invertible.
        """
        return self.mod_matrix_inv(**kwargs)
    
    def _mod_matrix_adjugate(self) -> np.ndarray:
        l, w = self.shape
        Adj = self.zeros(self.shape)
        for i in range(l):
            for j in range(w):
                Adj[i,j] = self._matrix_cofactor(j,i) ## note the transpose here. Adj_{i,j} = C_{j,i}

        return Adj

    def _matrix_cofactor(self, row: int, col: int) -> int:
        """The elements of the cofactor matrix are
        $C_{i,j} = (-1)^{i+j} M_{i,j}$, where M is
        the matrix of minors of A.
        """
        return (-1)**(row+col) * self._matrix_minor(row,col)

    def _matrix_minor(self, row: int, col: int) -> int:
        """
        The elements of the minor matrix $M_{i,j}$
        are the determinants of $A$ with rows $i,j$ deleted.
        """
        
        l, w = self.shape

        if not (l==w and l>1):
            raise ValueError("Invalid matrix", self)
        
        B = self._matrix_delete_rowcol(row,col)

        return rint(np.linalg.det(B))
                
    def _matrix_delete_rowcol(self, row: int, col: int) -> np.ndarray:
        """Delete the given row and column and return the reduced array.
        """
        B = self.view(np.ndarray)
        B=np.delete(B,row,0)
        B=np.delete(B,col,1)
        return B

    @classmethod
    def set_d(cls, d) -> type[DnaryArray]:
        """Classmethod for creating a child class with a specific d.

        Args:
            d (_type_): _description_

        Returns:
            _type_: _description_
        """
        return type(f'{cls.__name__}(d={d})', (cls, ), {'d':d})

    
