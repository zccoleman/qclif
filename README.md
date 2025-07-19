# Qudit Cliffords
Python library for generating random qudit Clifford gates and decomposing them.



### Class Hierarchy:

DNaryMeta - metaclass for all the following classes:
- PrimeDnaryArrayBase - prime d-nary arrays.
- SymplecticArrayBase - Includes logic for symplectic algebra, such as nn and n, Lambda, inner products, symplectic checking, and embedding into larger matrices.
- Transvection - Uses the SymplecticArrayBase API and is used by the CliffordBase API
- ClifordBase - Implements the random symplectic algorithm (using Transvection) and also clifford decomposition