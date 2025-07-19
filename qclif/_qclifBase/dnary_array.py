import numpy as np
from typing import Self, Any
import warnings

from abc import ABCMeta, abstractmethod

from .dnary_arithmetic import rint, dnary_inverse, int_to_dnary, dnary_to_int
from .validation import validate_primes

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
    

class DnaryArrayBase(np.ndarray, metaclass=DNaryMeta):
    """Base class for d-nary integer numpy arrays (with d prime) that are either 1d or 2d square arrays. 
    
    Do not create instances of this class; instead, subclass this class, setting a class property for d, then create instances of that subclass like so:
    ```
    class DnaryArrayD3(PrimeDnaryArrayBase):
        d=3
    instance = DnaryArrayD3([1,2,0,1])
    ```
    Accepts any data acceptable by np.array(...). 
    """
    
    _validate_prime=False

    @property
    @abstractmethod
    def d(self):
        """The modulus of the array class.
        """
        raise NotImplementedError
    
    def __new__(cls, data: Any, *args, **kwargs) -> Self:
        """Create a new prime integer d-nary array. All input data will be coerced to an integer array in numpy, then modded by d.

        Args:
            data (Any): Data in any acceptable numpy format for a 1d array or a 2d square array.

        Raises:
            TypeError: You did not set a value for d in the class.
            ValueError: The data does not result in a 1d array or a 2d square array.

        Returns:
            Self: A d-nary array.
        """
        if not isinstance(cls.d, int):
            raise TypeError('Cannot instantiate base class without a definition for d.')
        d = cls.d
        # validate_primes(d)
        array = np.array(data, dtype=int) % d
        
        if array.ndim==1:
            nn = len(array)
        elif array.ndim==2:
            l, w = array.shape
            if not (l==w or 1 in [l, w]):
                raise ValueError(f'Input data must be a 1d vector or a 2d square array', data)
            # nn = max(l,w)
        else: raise ValueError(f'Input data must be 1- or 2-dimensional', data)
        # if not nn%2==0: raise ValueError(f'Input data must have an even dimension:', data)
        obj = array.view(cls)
        # obj.flags.writeable = False
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
        """Checks the dimension of the array. 2d arrays will return True.

        Returns:
            bool
        """
        return self.ndim==2
    
    @property
    def is_vector(self) -> bool:
        """Checks the dimension of the array. 1d arrays will return True.

        Returns:
            bool
        """
        return self.ndim==1
    
    @classmethod
    def dnary_inverse(cls, n: int) -> int:
        """Calculate the d-nary inverse of an integer using the class's value of `d`.

        Args:
            n (int): The integer to invert.

        Returns:
            int: The integer n_inv such that (n*n_inv)%d=1
        """
        return dnary_inverse(n, cls.d)
    
    @classmethod
    def int_to_dnary(cls, n:int, result_list_size:int|None=None) -> list[int]:
        """Alias to `int_to_dnary(n, cls.d)`.

        Args:
            n (int): The integer to be factored into its d-nary digits.
            result_list_size (int | None, optional): The length of the list to return; if None, will use the minimum number of digits to result the input integer in d-nary. Defaults to None.

        Raises:
            ValueError: Invalid input parameters.

        Returns:
            list[int]: A list of the d-nary digits of the input integer. The ith element is the coefficient of d**i in the d-nary expansion of the input.
        """
        return int_to_dnary(n, cls.d, result_list_size)
    
    @classmethod
    def dnary_to_int(cls, digits) -> int:
        """Given a list of the d-nary digits of a number, return that number in base 10.

        Args:
            digits (list[int]): A list of the d-nary digits of the number, where the ith element is the coefficient of d**i.

        Returns:
            int: The number in base 10.
        """
        return dnary_to_int(digits, cls.d)
    
    def is_nonzero(self) -> bool:
        """Whether the array has any nonzero values in it.

        Returns:
            bool
        """
        return bool(self.any())
    
    def is_zero(self) -> bool:
        """Returns True for arrays that are all 0.

        Returns:
            bool
        """
        return not self.is_nonzero()


    @classmethod
    def random_array(cls, shape:tuple[int], allow_zero=True) -> Self:
        """Returns an array of the given shape with uniform random integers mod d.

        Args:
            shape (tuple[int]): The shape of the array.
            allow_zero (bool, optional): Whether to allow the zero matrix. Defaults to True.

        Returns:
            Self: A uniform random d-nary array.
        """
        if allow_zero:
            return cls(np.random.randint(0, cls.d, shape))
        for i in range(1000):
            a = cls.random_array(shape, allow_zero=True)
            if a.is_nonzero():
                return a
        raise RuntimeError('Could not create a non-zero random matrix.')
    
    @classmethod
    def eye(cls, n: int) -> Self:
        """
        Returns:
            Self: The (n, n) identity matrix with integer type.
        """
        return cls(np.identity(n, dtype='int32'))
    
    @classmethod
    def zeros(cls, shape:int|tuple[int]) -> Self:
        """Returns an all-zero array of the given shape.

        Args:
            shape (int|tuple[int]): The shape of the desired array.

        Returns:
            Self: The zero matrix of the given shape with integer type.
        """
        return cls(np.zeros(shape, dtype='int32'))
    
    @classmethod
    def basis_vector(cls, i: int, n: int) -> Self:  ## returns the length-n standard basis vector e_i (python indexing)
        """Generates the ith standard basis vector of length n for the class.

        Args:
            i (int): The component of the basis vector.
            n (int): The length of the vector

        Returns:
            Self: A basis vector.
        """
        e_i = cls.zeros(n)
        e_i[i]=1
        return e_i
    
    def determinant(self) -> int:
        """
        Returns:
            int: The determinant of the array modulo d.
        """
        return rint(np.linalg.det(self)) % self.d
    
    def det(self) -> int:
        """Alias for the `determinant` method.

        Returns:
            int: The d-nary determinant of the matrix.
        """
        return self.determinant()
    
    def is_invertible(self) -> bool:
        """Determines if a matrix is invertible by checking if its determinant is invertible mod d.

        Returns:
            bool
        """
        return self.dnary_inverse(self.det()) is not None
    
    def mod_matrix_inv(self, validate=True, suppress_warnings=True) -> Self|None:  ## calculates the modular inverse of a matrix
        """Calculates the modular integer matrix inverse using $$A^{-1}=\frac{1}{|A|}Adj(A)$$, where $Adj(A)$ is the matrix adjugate of $A$.

        Args:
            validate (bool, optional): Whether to validate the result through multiplication. Defaults to True.
            suppress_warnings (bool, optional): If False, warnings will be raise if the array is not invertible. Defaults to True.

        Raises:
            RuntimeError: If the inversion fails for an invertible array. Theoretically this should never happen.

        Returns:
            Self|None: The array's inverse, or None if the array is not invertible.
        """
        assert self.is_matrix
        

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
        """Alias for `mod_matrix_inv`.

        Returns:
            Self|None: The array's inverse, or None if the array is not invertible.
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
        """The elements of the cofactor matrix are $C_{i,j} = (-1)^{i+j} M_{i,j}$, where M is the matrix of minors of A."""
        return (-1)**(row+col) * self._matrix_minor(row,col)

    def _matrix_minor(self, row: int, col: int) -> int:
        """The elements of the minor matrix $M_{i,j}$ are the determinants of $A$ with rows $i,j$ deleted."""
        
        l, w = self.shape

        if not (l==w and l>1):
            raise ValueError("Invalid matrix", self)
        
        B = self._matrix_delete_rowcol(row,col)

        return rint(np.linalg.det(B))
                
    def _matrix_delete_rowcol(self, row: int, col: int) -> np.ndarray:
        B = self.view(np.ndarray)
        B=np.delete(B,row,0)
        B=np.delete(B,col,1)
        return B



    
