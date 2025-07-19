import numpy as np
from typing import Self

from .dnary_array import DnaryArrayBase

class SymplecticArrayBase(DnaryArrayBase):
    """Class implementing the logic of symplectic vectors and matrices such as inner products and embeddings.
    """

    @property
    def n(self) -> int:
        """
        Raises:
            ValueError: Not accessible for odd-sized arrays.

        Returns:
            int: Half of the array size.
        """
        if self.nn%2==0:
            return self.nn//2
        raise ValueError('Cannot perform symplectic operations with odd-sized array:', self)

    @property
    def nn(self) -> int:
        """
        Returns:
            int: The full array size, or 2*n. 
        """
        return len(self)
    
    @classmethod
    def LambdaN(cls, n: int) -> Self:
        r"""Returns the defining symplectic array for any given n, defined here as 
        $\Lambda_n=\begin{bmatrix}0 & -\one_n\\\one_n & 0\end{bmatrix}$.

        See https://en.wikipedia.org/wiki/Symplectic_matrix#The_matrix_%CE%A9.

        Args:
            n (int): Lambda is a (2n, 2n) array.

        Returns:
            Self: The d-nary (2n, 2n) array that defines the symplectic group and symplectic inner product.
        """
        U = np.zeros((2*n,2*n),dtype='int32')
        for i in range(n):
            U[i + n,i]=1
        return cls((U - U.T))
    
    @property
    def Lambda(self) -> Self:
        """
        Returns:
            Self: The symplectic array matching the current object's dimension (nn by nn).
        """
        return self.LambdaN(self.n)

    def is_symplectic_pair_with(self, other: Self) -> bool:
        """If the current object is a vector, determine whether it forms a symplectic pair with another vector.

        Args:
            other (Self): The other vector.

        Raises:
            TypeError: Both the calling object and the input must be 1d arrays.

        Returns:
            bool: Whether the two vectors form a symplectic pair (have a symplectic inner product of -1 (mod d)).
        """
        if not self.is_vector and other.is_vector:
            raise TypeError('Can only compute symplectic pairs between vectors', self, other)
        return self.__class__.is_symplectic_pair(self, other)
    
    def is_symplectic_matrix(self) -> bool:
        """
        Returns:
            bool: whether the calling array is a symplectic matrix.
        """
        result = np.array_equal(
            self.Lambda,
            (self.T @ self.Lambda @ self),
        )
        return result
    

    def inner_product_with(self, other: Self) -> int:
        """Compute the symplectic inner product of the calling vector (v1) with the other vector (v2).

        Args:
            other (Self): The other vector (v2).

        Returns:
            int: The symplectic inner product v1 @ Lambda @ v2 (mod d).
        """
        return self.__class__.inner_product(self, other)
    
    def __or__(self, other) -> int:
        """Overloads the pipe (|) operator as an alias for the symplectic inner product.
        """
        return self.inner_product_with(other)
    

    def embed_symplectic(self) -> Self:  ## embeds the 2n x 2n symplectic matrix q into a 2n+1 x 2n+1 symplectic matrix Q.
        """Embeds a (2n, 2n) array into a (2n+1, 2n+1) array, preserving symplecticity.

        Returns:
            Self: The embedded array.
        """
        if not self.is_matrix:
            raise ValueError("Can only perform symplectic embedding for square arrays.")
        n, nn = self.n, self.nn
        e0 = self.basis_vector(0, nn+2).view(np.ndarray)
        e1 = self.basis_vector(n+1, nn+2).view(np.ndarray)

        zs = np.zeros((1,nn),dtype='int32')
        Q = self.view(np.ndarray)
        Q = np.insert(Q, 0, zs, axis=0)
        Q = np.insert(Q, n+1, zs, axis=0)
        Q = np.insert(Q, 0, e0, axis=1)
        Q = np.insert(Q, n+1, e1, axis=1)
        return type(self)(Q)
    
    @classmethod
    def inner_product(cls, v1: Self, v2: Self) -> int:
        """Computes the symplecitc inner product of the given vectors coerced into the type of the calling class.
        Thus, the vectors may be defined in a class with a different modulus (d).

        Args:
            v1 (Self): The first vector. Will be coerced into the type of the calling class.
            v2 (Self): The second vector. Will be coerced into the type of the calling class.

        Raises:
            ValueError: Input arrays are not one-dimensional.
            ValueError: Input arrays are not the same size.

        Returns:
            int: The symplectic inner product (as defined by the calling class) of v1 and v2.
        """
        v1 = cls.check_or_coerce_type(v1)
        v2 = cls.check_or_coerce_type(v2) 

        if not (v1.ndim==1 and v2.ndim==1):
            raise ValueError(f"Inputs must be 1-d vectors", v1, v2)
        
        try:
            result = v1 @ v1.Lambda @ v2
            # assert isinstance(result, int)
            return result
        except ValueError as e:
            raise ValueError(f"Vectors must have the same dimensions", v1, v2, e)

    @classmethod
    def is_symplectic_pair(cls, v1: Self, v2: Self) -> bool:
        """Determine whether a pair of vectors coerced into the calling class form a symplectic pair (have an inner product of -1).

        Args:
            v1 (Self): The first vector.
            v2 (Self): The second vector.

        Returns:
            bool: Whether the vectors form a symplectic pair.
        """
        return cls.inner_product(v1, v2)==(-1 % cls.d)