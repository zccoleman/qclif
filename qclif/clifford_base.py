from __future__ import annotations
import numpy as np
import math
from typing import Self, Any, Literal
from numbers import Integral

from ._qclifBase.symplectic_array_base import SymplecticArrayBase
from ._qclifBase.validation import validate_primes

class Transvection:
    r"""A class representing transvections, a class of linear transformations on vectors.
    Transvections are defined by a vector $\vec u$ and a constant $c$, and act as
    $v\mapsto T_{u, c}(v)=v + c\langle v, u\rangle u$. All arithmetic is mod d.
    Transvections applied to matrices apply the above transformation column-wise.
    """
    def __init__(self, u: CliffordBase, c:int):
        """Create a transvection within the class defined by the input vector.

        Args:
            u (CliffordBase): The vector to transvect by.
            c (int): The constant of the transvection.
        """
        self.u = u
        self.c = c

    def __call__(self, vector_or_matrix: CliffordBase) -> CliffordBase:
        """Apply the transvection to a vector or matrix of the same type as the transvection's vector.

        Args:
            vector_or_matrix (CliffordBase): A vector or matrix of the same type as the Transvection's vector.

        Returns:
            CliffordBase: The transvection applied to the vector or matrix.
        """
        if vector_or_matrix.is_matrix:
            return self.transvect_matrix_columns(vector_or_matrix)
        
        assert vector_or_matrix.is_vector

        return vector_or_matrix + self.c*(vector_or_matrix | self.u)*self.u
    
    def __repr__(self):
        return f'({self.u.__repr__()}, {self.c.__repr__()})'
    
    def transvect_vector(self, v: CliffordBase) -> CliffordBase:
        """Apply the transvection to a vector of the same type as the transvection's vector.

        Args:
            v (CliffordBase): A vector of the same type as the Transvection's vector.

        Returns:
            CliffordBase: The transvection applied to the vector.
        """
        return v + self.c*(v | self.u)*self.u
    
    def transvect_matrix_columns(self, A: CliffordBase) -> CliffordBase:
        """Apply the transvection to a matrix of the same type as the transvection's vector.

        Args:
            A (CliffordBase): A matrix of the same type as the Transvection's vector.

        Returns:
            CliffordBase: The transvection applied to the matrix.
        """
        rows,cols = A.shape
        # ans = np.zeros((rows, cols), dtype='int')
        ans = np.zeros_like(A)
        for i in range(cols):
            ans[:,i] = self(A[:,i])
        return ans
    
    @classmethod
    def _easy_transvection(cls, u: CliffordBase, v: CliffordBase) -> Self:
        r"""Find a transvection from u to v in the simple case where $\langle u, v\rangle \neq 0$.

        Args:
            u (CliffordBase): Vector you are transvecting *from*.
            v (CliffordBase): Vector you are transvecting *to*.

        Raises:
            RuntimeError: A transvection from u to v does not exist.

        Returns:
            Self: A transvection T such that T(u)==v.
        """
        ## returns w, c such that v = Z_{w,c}(u)
        k = u | v
        if k==0:
            raise RuntimeError('No single transvection between vectors exists', u, v)
        w = v-u
        c = u.dnary_inverse(k)
        return cls(w, c)

    @classmethod
    def find_transvection(cls, u: CliffordBase, v: CliffordBase) -> tuple[Self, Self]:
        """Returns a tuple of transvections T1, T2 such that T2(T1(u))=v

        Args:
            u (CliffordBase): Vector you are transvecting *from*.
            v (CliffordBase): Vector you are transvecting *to*.

        Returns:
            (Transvection, Transvection): A tuple (T1, T2) such that T2(T1(u))==v
        """
        nn=u.nn
        assert nn==v.nn
        n = u.n
        assert type(u) is type(v)
        SP_D = type(u)

        h1, h2 = SP_D(np.zeros(nn, dtype='int32')), SP_D(np.zeros(nn, dtype='int32'))
        c1, c2 = 0,0
        k = u | v
        
        if np.array_equal(u, v):
            return Transvection(h1, c1), Transvection(h2, c2)
        
        if k != 0:
            T1 = cls._easy_transvection(u, v)
            return T1, Transvection(h2, c2)
        
        
        z = np.zeros(nn, dtype='int32')
        u_found = 0
        v_found = 0
        for i in range(n):
            pair1 = (u[i], u[n+i])
            pair2 = (v[i], v[n+i])
            zero = (0,0)
            
            if (pair1 != zero) and (pair2 != zero):
                if pair1[0] and pair2[0]:
                    z = z + SP_D.basis_vector(i+n, nn)
                elif pair1[1] and pair2[1]:
                    z = z + SP_D.basis_vector(n, nn)
                else:
                    z = z + SP_D.basis_vector(i, nn) + SP_D.basis_vector(n+i, nn)
                assert u | z != 0, (u, v, z)
                assert z | v != 0, (u, v, z)
                T2 = cls._easy_transvection(u, z)
                # assert np.array_equal(transvect(x,h2,c2,d),z)
                assert np.array_equal(T2(u), z)
                T1 = cls._easy_transvection(z, v)
                # assert np.array_equal(transvect(z,h1,c1,d),y)
                assert np.array_equal(T1(z), v)
                return T1, T2
            
            if u_found == 0 and pair1 != zero:
                u_found = i
            if v_found == 0 and pair2 != zero:
                v_found = i
                
        if u[u_found] != 0:
            z = z + SP_D.basis_vector(u_found+n, nn)
        else:
            assert u[u_found+n] != 0
            z = z + SP_D.basis_vector(u_found, nn)
        if v[v_found] != 0:
            z = z + SP_D.basis_vector(v_found+n, nn)
        else:
            assert v[v_found+n] != 0
            z = z + SP_D.basis_vector(v_found,nn)
            
        assert u|z != 0, (u, z)
        assert z|v != 0, (z, v)
        T2 = cls._easy_transvection(u, z)
        assert np.array_equal(T2(u),z)
        T1 = cls._easy_transvection(z,v)
        assert np.array_equal(T1(z),v)
        return T1, T2

class _RandomSymplectic(SymplecticArrayBase):

    @classmethod
    def symplectic_group_size(cls, n: int) -> int:
        """Returns the size of the symplectic group for a given n.

        Args:
            n (int): Half the size of the symplectic matrix, or the number of qubits in the equivalent Clifford group.

        Returns:
            int: The size of the symplectic group. 
        """
        d = cls.d
        return math.prod((d**(2*j-1)) * (d**(2*j)-1) for j in range(1, n+1))
    
    @classmethod
    def random_symplectic(cls, n: int) -> Self:
        """Generate a uniformly random element of the symplectic group.

        Args:
            n (int): Half the size of the symplectic matrix to be generated, or the number of qubits in the equivalent Clifford group.

        Returns:
            Self: A uniform random symplectic matrix mod d.
        """

        nn = 2*n
        e1 = cls.basis_vector(0, nn)
        en = cls.basis_vector(n, nn)

        v = cls.random_matrix(nn, allow_zero=False)

        t1, t2 = Transvection.find_transvection(e1, v)

        assert np.array_equal(t1(t2(e1)), v), (v, t1, t2)

        b = cls.random_matrix(nn, allow_zero=True)
        b[n]=0
        u0 = t1(t2(b))

        assert u0|v == 0


        T_prime = Transvection(u0, c=1)

        # w = T_prime(t1(t2(en)))

        if n==1:
            Q=cls.eye(2)
        else:
            Q=cls.random_symplectic(n-1).embed_symplectic()
        
        return T_prime(t1(t2(Q)))

    @classmethod
    def from_index(cls, index:int, n:int) -> Self:
        """Generate an element of the symplectic group from that element's index. 

        Args:
            index (int): An integer from 0 (inclusive) to the size of the symplectic group (exclusive). See `.symplectic_group_size`.
            n (int): Half the size of the symplectic matrix to be generated, or the number of qubits in the equivalent Clifford group.

        Returns:
            Self: _description_
        """
        assert index in range(cls.symplectic_group_size(n))

        d = cls.d
        nn = 2*n
        e1 = cls.basis_vector(0, nn)
        en = cls.basis_vector(n, nn)
        
        v1_count = d**(2*n) - 1 ## the number of choices for v1
        v2_count = d**(2*n-1)   ## the number of choices for v2
        v1_v2_count = v1_count * v2_count ## the number of choices for (v1, v2)
        new_index, remainder = divmod(index, v1_v2_count)   ## div out the number of choices for (v1, v2)
                                                            ## remainder determines (v1, v2)
                                                            ## new_index will determine the next pair of vectors.

        v1_index = remainder % v1_count + 1
        v2_index = remainder//v1_count

        v1 = cls(cls.int_to_dnary((v1_index), result_list_size=nn))
        b = cls.int_to_dnary(v2_index, result_list_size=nn-1)
        b.insert(n, 0)
        b = cls(b)

        t1, t2 = Transvection.find_transvection(e1, v1)

        assert np.array_equal(t1(t2(e1)), v1), (v1, t1, t2)

        u0 = t1(t2(b))
        assert u0|v1 == 0, (n, v1, b, u0)

        T_prime = Transvection(u0, c=1)

        # v2 = T_prime(t1(t2(en)))

        if n==1:
            Q=cls.eye(2)
        else:
            Q=cls.from_index(new_index, n-1).embed_symplectic()
    
        return T_prime(t1(t2(Q)))

class _CliffordValidation(object):
    @staticmethod
    def _validate_qudit_indices(num_qudits:int, *qudit_indices):
        """Validates whether a list of given qudit indices are valid given the total number of qudits available, and raises exceptions if not.
        Args:
            num_qudits (int): The number of qudits in the system.
        """
        if not isinstance(num_qudits, Integral) or num_qudits<=0:
            raise ValueError('The number of qudits must be a positive integer:', num_qudits)
        
        for qudit_index in qudit_indices:
            if not isinstance(qudit_index, Integral):
                raise ValueError('Qudit index must be an integer:', qudit_index)
            
            if not 0<=qudit_index<num_qudits:
                raise ValueError('Qudit index must be in [0, n):', qudit_index)

    @classmethod
    def _validate_multiplier(cls, *mults):
        """Validates whether a given list of multiplier gates are valid gates.
        To be valid, each multiplier must be less than the class modulus (d) and must be integer type.
        """
        for r in mults:
            if not isinstance(r, Integral):
                raise ValueError('Input must be an integer:', r)
            if not r<cls.d:
                raise ValueError(f'Input must be less than the class modulus {cls.d}:', r)

class _SpecialClifford(SymplecticArrayBase):
    @classmethod
    def identity(cls, n: int) -> Self:
        """Representation for the identity gate.

        Args:
            n (int): The number of qudits.

        Returns:
            Self: The symplectic matrix corresponding to the identity gate.
        """
        Q = cls.eye(2*n)
        Q.name='Id'
        return Q
    
    @classmethod
    def clif_swap(cls, i: int, j: int, n: int) -> Self:
        """Representation for Clifford gate that swaps qudits i and j in an n-qudit system.
        
        Args:
            i (int): The first qudit to swap.
            j (int): The second qudit to swap.
            n (int): The number of qudits in the system.

        Returns:
            Self: A symplectic matrix representing the Clifford gate.
        """
        cls._validate_qudit_indices(n, i, j)
        Q = cls.eye(2*n)
        Q[i,i] = Q[j,j] = Q[i+n, i+n] = Q[j+n, j+n] = 0
        Q[i,j] = Q[j,i] = Q[i+n,j+n] = Q[j+n,i+n] = 1
        Q.name=f'SWAP({i}, {j})'
        return Q

    @classmethod
    def clif_mult(cls, i:int, r:int, n:int) -> Self:
        """Representation for the Clifford gate that multiplies the configuration space of qudit i by r in an n-qudit system.
        For example, a mult(2) gate on d=3 qudits will take 0 -> 0, 1 -> 2, and 2 -> 1 (4 mod 3 = 1).

        Args:
            i (int): The qudit to apply the gate to.
            r (int): The number to multiply the qudit's configuration space by.
            n (int): The number of qudits in the system.

        Returns:
            Self: A symplectic matrix representing the Clifford gate.
        """
        cls._validate_qudit_indices(n, i)
        cls._validate_multiplier(r)
        r = r%cls.d
        r_inv = cls.dnary_inverse(r)

        Q = cls.eye(2*n)
        Q[i,i], Q[i+n, i+n] = r, r_inv
        Q.name=f'MULT({i}, {r})'
        return Q
    
    @classmethod
    def clif_csum(cls, i:int, j:int, r:int, n:int) -> Self:
        """Representation for the CSUM(i,j)^r Clifford gate, i.e. adding qudit i to qudit j, r times.

        Args:
            i (int): The control of the CSUM.
            j (int): The target of the CSUM.
            r (int): The power of the CSUM
            n (int): The number of qudits in the system.

        Returns:
            Self: A symplectic matrix representing the Clifford gate.
        """
        cls._validate_qudit_indices(n, i, j)
        cls._validate_multiplier(r)
        r = r%cls.d
        Q = cls.eye(2*n)
        Q[j,i]=r
        Q[i+n, j+n] = (r*-1)%cls.d
        Q.name = f'CSUM({i}, {j})^{r}'
        return Q
    
    @classmethod
    def clif_phase(cls, i:int, r:int, n:int) -> Self:
        """Representation of the phase gate on qudit i, r times.

        Args:
            i (int): The qudit to apply the gate to.
            r (int): The number of phase gates to apply.
            n (int): The number of qudits in the system.

        Returns:
            Self: A symplectic matrix representing the Clifford gate.
        """
        cls._validate_qudit_indices(n, i)
        cls._validate_multiplier(r)
        r = r%cls.d
        Q = cls.eye(2*n)
        Q[i+n, i] = r
        Q.name = f'P({i})^{r}'
        return Q
    
    @classmethod
    def clif_phase_inverse(cls, i:int, r:int, n:int) -> Self:
        """Representation of the inverse phase gate on qudit i, r times.

        Args:
            i (int): The qudit to apply the gate to.
            r (int): The number of phase gates to apply.
            n (int): The number of qudits in the system.

        Returns:
            Self: A symplectic matrix representing the Clifford gate.
        """
        cls._validate_qudit_indices(n, i)
        cls._validate_multiplier(r)
        r = r%cls.d
        Q = cls.eye(2*n)
        Q[i, i+n] = (-1*r)%cls.d
        Q.name = f'Pdg({i})^{r}'
        return Q
    
    @classmethod
    def clif_fourier(cls, i:int, n:int) -> Self:
        """Representation of the Fourier transform gate, or the qudit-generalized H gate.

        Args:
            i (int): The qudit to apply the gate to.
            n (int): The number of qudits in the system.

        Returns:
            Self: A symplectic matrix representing the Clifford gate.
        """
        
        cls._validate_qudit_indices(n, i)
        Q = cls.eye(2*n)
        Q[i,i]=Q[i+n, i+n]=0
        Q[i, i+n] = -1%cls.d
        Q[i+n, i] = 1
        Q.name = f'H({i})'
        return Q
    
    @classmethod
    def clif_fourier_inv(cls, i:int, n:int) -> Self:
        """Representation of the inverse Fourier transform gate.

        Args:
            i (int): The qudit to apply the gate to.
            n (int): The number of qudits in the system.

        Returns:
            Self: A symplectic matrix representing the Clifford gate.
        """
        cls._validate_qudit_indices(n, i)
        Q = cls.eye(2*n)
        Q[i,i]=Q[i+n, i+n]=0
        Q[i, i+n] = 1
        Q[i+n, i] = -1%cls.d
        Q.name = f'Hdg({i})'
        return Q

class _DecomposeClifford(_SpecialClifford):
    
    def decompose(self, output_form:Literal['string', 'matrix']='string') -> list[str|Self]:
        """Decomposes a symplectic matrix into a product of elementary symplectic matrices, each of which correspond to a known qudit Clifford gate.

        Args:
            output_form (Literal[&#39;string&#39;, &#39;matrix&#39;]): The type of output to be generated. Default is a list of strings of gate names.

        Returns:
            list[str|Self]: A list of either string gate names of symplectic matrices. Elements in the list compose to the original Clifford gate.
        """
        for i in range(self.n):
            self = self._get_column_i_to_identity(i)._get_column_i_n_to_identity(i)
        
        assert np.array_equal(self, self.eye(2*self.n))

        if not hasattr(self, 'operators'):
            self._apply_operator(self.identity(self.n))
        if output_form=='matrix':
            return self.operators
        return [op.name for op in self.operators]
        
    def _apply_operator(self, op:Self) -> Self:
        """Apply a custom named gate and append it to the current gate's list of gates.

        Args:
            op (Self): The operator to apply.

        Returns:
            Self: The new gate.
        """
        if hasattr(self, 'operators'):
            assert isinstance(self.operators, list)
            self.operators.append(op)
        else:
            self.operators = [op]
        operators = self.operators
        self = op @ self
        self.operators = operators
        return self

    def _get_ii_nonzero(self, i:int) -> Self:
        """Apply custom operators such that the resulting operator's [i,i] component is non-zero.
        If self[i,i] is already non-zero, nothing happens.

        Args:
            i (int): The index of the column to target.

        Returns:
            Self: The gate with the operators applied and appended to the operators list.
        """
        if self[i,i]==0:
            row=None
            for k in range(self.nn):
                if self[k, i] != 0:
                    row = k
                    break
            assert row is not None

            if row<self.n:
                self = self._apply_operator(self.clif_swap(row, i, self.n))
            else:
                self =self._apply_operator(self.clif_fourier(row-self.n, self.n))
                if row != i+self.n:
                    self =self._apply_operator(self.clif_swap(row-self.n, i, self.n))
        return self
    
    def _get_nonzero_ii_to_one(self, i:int) -> Self:
        """Apply an operator to self to ensure self[i,i]=1.

        Args:
            i (int): The index of the column to target.

        Returns:
            Self: The gate with the operators applied and appended to the operators list.
        """
        assert self[i,i]%self.d
        if self[i,i] != 1:
            self = self._apply_operator(self.clif_mult(i, self.dnary_inverse(self[i,i].item()), n=self.n))
        return self

    def _get_first_n_rows_except_i_to_zero(self, i:int) -> Self:
        """Get the first n rows of the ith column to be all zero except for [i,i].

        Args:
            i (int): The index of the column to target.

        Returns:
            Self: The gate with the operators applied and appended to the operators list.
        """
        assert self[i,i]==1, (self, i)
        n = self.n
        d = self.d
        ## get the first n rows except for row i to have zero
        for j in range(n):
            if (j!=i) and (self[j,i] != 0):
                r = (-1 * self[j,i])%d
                self = self._apply_operator(self.clif_csum(i,j,r, n))
        return self
    
    def _get_i_n_to_zero(self, i:int) -> Self:
        """Get self[i+n, i] to zero.

        Args:
            i (int): The index of the column to target.

        Returns:
            Self: The gate with the operators applied and appended to the operators list.
        """
        n = self.n
        d = self.d
        ## get [i+n, i]=0
        if self[i+n, i]!=0:
            r = (-1*self[i+n, i])%d
            self = self._apply_operator(self.clif_phase(i, r, n))
        return self
    
    def _get_lower_block_to_zero(self, i:int) -> Self:
        """Assuming self[i+n, i]=0, set the rest of the lower half of column i to zero.

        Args:
            i (int): The index of the column to target.

        Returns:
            Self: The gate with the operators applied and appended to the operators list.
        """
        n = self.n
        d = self.d
        ## if the lower block is not already zero:
        if not np.array_equal(self[n:2*n, i], np.zeros(n)):
            ## swap row i to row n+i. [i+n, i] will then be 1 and the whole top block will be 0. CSUMs will leave top block as 0.
            self = self._apply_operator(self.clif_fourier(i, n))
            ## apply CSUMs on the bottom using the 1 in [i+n, i]
            for j in range(n):
                if (j!=i) and (self[j+n, i]!=0):
                    r = (self[j+n, i])%d
                    self = self._apply_operator(self.clif_csum(j, i, r, n))
            ## then swap n+i back to i
            self = self._apply_operator(self.clif_fourier_inv(i, n))
        return self
    
    def _get_column_i_to_identity(self, i:int) -> Self:
        """Apply row reduction operators to set column i to the identity column.
        The resulting column should be all zeros except for the ith row a one.

        Args:
            i (int): The index of the column to target.

        Returns:
            Self: The gate with the operators applied and appended to the operators list.
        """
        return (
            self._get_ii_nonzero(i)
            ._get_nonzero_ii_to_one(i)
            ._get_first_n_rows_except_i_to_zero(i)
            ._get_i_n_to_zero(i)
            ._get_lower_block_to_zero(i)
        )
    
    def _get_column_i_n_to_identity(self, i:int) -> Self:
        """Apply row reduction operators to set column i+n to the identity column.
        The resulting column should be all zeros except for the ith row a one.

        Args:
            i (int): The qudit index to target; row i+n will be targeted.

        Returns:
            Self: The gate with the operators applied and appended to the operators list.
        """
        n = self.n
        d = self.d

        assert self[i+n, i+n]==1 ## guaranteed if column i is identity
        ## get lower block to all zeros
        for j in range(n):
            if (j!=i) and (self[j+n, i+n]!=0):
                r = (self[j+n, i+n])%d
                self = self._apply_operator(self.clif_csum(j, i, r, n))
        ## get [i, i+n] to zero
        if self[i, i+n]!=0:
            r = (self[i, i+n])%d
            self = self._apply_operator(self.clif_phase_inverse(i, r, n))
        ## get upper block to all zeros if not already
        if not np.array_equal(self[0:n, i+n], np.zeros(n)):
            self = self._apply_operator(self.clif_fourier_inv(i, n))
            for j in range(n):
                if (j!=i) and (self[j, i+n]!=0):
                    r = (-1*self[j, i+n])
                    self = self._apply_operator(self.clif_csum(i, j, r, n))
            self = self._apply_operator(self.clif_fourier(i, n))
        
        return self

class CliffordBase(_RandomSymplectic, _DecomposeClifford, _CliffordValidation):

    
    @classmethod
    def clifford_group_size(cls, n:int) -> int:
        """Compute the size of the Clifford group for any number of qudits for the calling class's modulus d.

        Args:
            n (int): The number of qudits. Must be non-negative.

        Returns:
            int: The size of the Clifford group on n qudits of dimension d.
        """
        d = cls.d
        if not isinstance(n, Integral):
            raise ValueError("The number of qudits must be an integer")
        if not n>=0:
            raise ValueError("n must be a nonnegative integer", n)
        return d**(n*(n+2)) * math.prod(d**(2*j)-1 for j in range(1, n+1))
    
    
