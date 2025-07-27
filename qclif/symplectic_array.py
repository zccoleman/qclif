import numpy as np
import math
from typing import Self

from qclif.dnary_array import DnaryArrayBase

class SymplecticArrayBase(DnaryArrayBase):
    r"""Class implementing
    [symplectic](https://en.wikipedia.org/wiki/Symplectic_matrix)
    algebra on matrices and vectors over 
    $\mathbb{Z}_d$, the modulo-$d$ integers.

    For a given $n>0$, the symplectic group $\text{Sp}(2n, \mathbb{Z}_d)$
    is the group of $2n\times2n$ matrices $A$ such that
    $$ A^T \Lambda_n A=\mathbb{I}_{2n},$$
    where
    $$\Lambda_n=\begin{bmatrix}0 & -\mathbb{I}_n \\ \mathbb{I}_n & 0 \end{bmatrix} \text{ mod } d$$
    and $\mathbb{I}_n$ is the $n\times n$ identity matrix.

    Similarly, the symplectic inner product $\langle v, w\rangle$
    between length-$2n$ vectors $v,w$ is
    defined as
    $$ \langle v, w\rangle = v^T\Lambda_n w.$$

    If $\langle v, w\rangle=-1 \text{ mod } d$, we say 
    $(v,w)$ is a symplectic pair.

    An equivalent definition for a symplectic matrix
    is the requirement that, for each $0\leq i< n$, the 
    matrix's $i$th and $(i+n)$th columns form a symplectic pair
    (Python indexing).

    Note: 
        $2n\times2n$ symplectic matrices over $\mathbb{Z}_d$ form
        a representation of the Clifford group $\mathcal{C}_d^n$ on
        $n$ qudits of dimension $d$. In fact,
        $$
        \text{Sp}(2n, \mathbb{Z}_d)\simeq \mathcal{C}_d^n / \mathcal{P}_d^n,
        $$
        where $\mathcal{P}_d^n$ is the $n$ qudit Pauli group.
        See [arXiv:quant-ph/0408190](https://arxiv.org/abs/quant-ph/0408190).

    Note:
        The matrix $\Lambda_n$ may be defined differently in other contexts.
        The definition used here follows
        [arXiv:quant-ph/0408190](https://arxiv.org/abs/quant-ph/0408190).
        See [here](https://en.wikipedia.org/wiki/Symplectic_matrix#The_matrix_%CE%A9)
        for other definitions.
  
    """

    @property
    def n(self) -> int:
        r"""
        Half of the array size. Will raise an exception for odd-sized arrays.
        
        Defines the defining symplectic array $\Lambda_n$. See the 
        [`.Lambda`][qclif.SymplecticArrayBase.Lambda] property
        and the [`.LambdaN`][qclif.SymplecticArrayBase.LambdaN] method. 

        Raises:
            ValueError: Not accessible for odd-sized arrays.

        Returns:
            Half of the array size.
        """
        if self.nn%2==0:
            return self.nn//2
        raise ValueError('Cannot perform symplectic operations with odd-sized array:', self)

    @property
    def nn(self) -> int:
        """
        Returns:
            The symplectic array size $2n$.
        """
        return len(self)
    
    @classmethod
    def LambdaN(cls, n: int) -> Self:
        r"""Returns the defining symplectic array $\Lambda_n$ for any given $n$ 
        using the class modulus $d$.

        See [`SymplecticArrayBase`][qclif.SymplecticArrayBase].

        Args:
            n (int): $\Lambda_n$ is a $2n\times 2n$ array.

        Returns:
            The $d$-nary (2n, 2n) array that defines the symplectic group and symplectic inner product.
        """
        U = np.zeros((2*n,2*n),dtype='int32')
        for i in range(n):
            U[i + n,i]=1
        return cls((U - U.T))
    
    @property
    def Lambda(self) -> Self:
        r"""
        Returns:
            The defining symplectic array matching the invoking
                object's dimension ($2n\times 2n$).
                
                See [`.LambdaN`][qclif.SymplecticArrayBase.LambdaN].
        """
        return self.LambdaN(self.n)

    def is_symplectic_pair_with(self, other: Self) -> bool:
        r"""If the calling object is a vector (i.e. 1D array),
        determine whether it forms a symplectic pair with another
        vector.

        See the [`.is_symplectic_pair`][qclif.SymplecticArrayBase.is_symplectic_pair]
        classmethod.

        Raises an exception if the calling object is not a vector.

        Args:
            other (Self): The other vector.

        Raises:
            TypeError: Both the calling object and the input 
                must be 1d arrays.

        Returns:
            Whether the two vectors form a
                symplectic pair.
        """
        if not self.is_vector and other.is_vector:
            raise TypeError('Can only compute symplectic pairs between vectors', self, other)
        return self.__class__.is_symplectic_pair(self, other)
    
    def is_symplectic_matrix(self) -> bool:
        """
        Returns:
            Whether the calling array is a symplectic matrix.
        """
        result = np.array_equal(
            self.Lambda,
            (self.T @ self.Lambda @ self),
        )
        return result
    

    def inner_product_with(self, other: Self) -> int:
        r"""Compute the symplectic inner product
        of the calling vector with the other vector.

        See the [`.inner_product`][qclif.SymplecticArrayBase.inner_product]
        classmethod.

        Args:
            other (Self): The other vector.

        Returns:
            The symplectic inner product of the calling
                vector with the other vector.

        Tip:
            Python's pipe operator `|` is overloaded with this method, so you can
            call `v1.inner_product_with(v2)` as `v1 | v2`.

        Examples:
        ```
        >>> class S3(SymplecticArrayBase): d=3
        >>> v1 = S3([1, 0, 0, 0])
        >>> v2 = S3([0, 0, 1, 0])
        >>> v1 | v2
        2
        ```
        """
        return self.__class__.inner_product(self, other)
    
    def __or__(self, other) -> int:
        return self.inner_product_with(other)
    

    def embed_symplectic(self) -> Self:  
        r"""Embeds a $2n\times 2n$ array into a
        $(2n+1)\times(2n+1)$ array via the symplecticity-preserving
        block embedding
        $$
        \begin{bmatrix}
            M_{11} & M_{12}\\
            M_{21} & M_{22}
        \end{bmatrix}
        \mapsto
        \begin{bmatrix}
            1 & 0 & 0 & 0\\
            0& M_{11} & 0 & M_{12}\\
            0 & 0 & 1 & 0\\
            0 & M_{21} & 0 & M_{22}
        \end{bmatrix},
        $$
        which inserts identity rows/columns at indices
        $0$ and $n$ (Python indexing) and fills the gaps with zeros.

        Returns:
            The embedded array.
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
    def inner_product(cls, v1: Self|np.ndarray, v2: Self|np.ndarray) -> int:
        r"""Computes the symplectic inner product of the
        given vectors.

        The symplectic inner product $\langle v, w\rangle$
        between length-$2n$ vectors $v,w$ is
        defined as
        $$ \langle v, w\rangle = v^T\Lambda_n w.$$ 

        Args:
            v1 (Self|np.ndarray): The first vector. Will be coerced into the type of the calling class.
            v2 (Self|np.ndarray): The second vector. Will be coerced into the type of the calling class.

        Raises:
            ValueError: Input arrays are not one-dimensional.
            ValueError: Input arrays are not the same size.

        Returns:
            The symplectic inner product of `v1` and `v2`.

        Tip:
            If you've already created a vector `v`, you can call
            `v.inner_product_with(w)` method to compute
            $\langle v,w\rangle$.

        Note:
            The modulus $d$ is defined by the calling class.
            Thus, `v1` and `v2` can be any array-like data, but they
            will be coerced into the type of the calling class
            (by coercing components to integers and modding by $d$).
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
        r"""Determine whether a pair of vectors 
        form a symplectic pair.

        Two vectors $v, w$ form a symplectic pair if
        $$\langle v, w\rangle = -1 \text{ mod } d.$$

        See also [`.is_symplectic_pair_with`][qclif.SymplecticArrayBase.is_symplectic_pair_with].

        Args:
            v1 (Self): The first vector.
            v2 (Self): The second vector.

        Returns:
            Whether the vectors form a symplectic pair.
        """
        return cls.inner_product(v1, v2)==(-1 % cls.d)
        
    @classmethod
    def symplectic_group_size(cls, n: int) -> int:
        r"""Returns the size of the symplectic group for a given n.

        Args:
            n (int): Half the size of the symplectic matrix,
                or the number of qubits in the equivalent Clifford group.

        Returns:
            The number of different $2n\times 2n$ symplectic
                matrices over $\mathbb{Z}_d$. 
        """
        d = cls.d
        return math.prod((d**(2*j-1)) * (d**(2*j)-1) for j in range(1, n+1))
