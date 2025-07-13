import numpy as np
import sympy as sp
import math
# from math import gcd, log, ceil
from numbers import Real

from .validation import validate_integers, validate_primes


def extra_ceil(num: Real) -> int:
    """Returns the smallest integer N such that N>num.

    Args:
        num (Real): The number to ceil

    Returns:
        int: The smallest integer N such that N>num.
    """
    result = math.ceil(num)
    if result==num:
        return int(result+1)
    return int(result)

def int_to_dnary(input_integer: int, d: int, result_list_size: int|None=None) -> list[int]:
    """Determines the d-nary digits of an input integer such that input_integer = sum(result[i] * d**i for i in range(result_list_size))

    Args:
        input_integer (int): The integer to be factored into its d-nary digits.
        d (int): The modulus.
        result_list_size (int | None, optional): The length of the list to return; if None, will use the minimum number of digits to result the input integer in d-nary. Defaults to None.

    Raises:
        ValueError: Invalid input parameters.

    Returns:
        list[int]: A list of the d-nary digits of the input integer. The ith element is the coefficient of d**i in the d-nary expansion of the input.
    """
    validate_integers(input_integer, d)
    if input_integer<0:
        raise ValueError('Input integer must be non-negative', input_integer)

    if input_integer==0:
        required_size = 1
    else:
        required_size = extra_ceil(math.log(input_integer, d))
    
    if result_list_size is None:
        result_list_size = required_size
    else:
        if not result_list_size >= required_size:
            raise ValueError(f'Input integer {input_integer} requires more than {result_list_size} digits.')

    validate_integers(result_list_size)
    
    digits = [0 for _ in range(result_list_size)]

    for j in range(result_list_size):
        current_digit = rint(input_integer%d)
        digits[j] = current_digit
        input_integer -= current_digit
        input_integer = input_integer/d

    return digits

def dnary_to_int(digits: list[int], d: int) -> int:
    """Given a list of the d-nary digits of a number, return that number in base 10

    Args:
        digits (list[int]): A list of the d-nary digits of the number, where the ith element is the coefficient of d**i.
        d (int): The modulus of the number.

    Returns:
        int: The number in base 10.
    """
    validate_integers(d, *digits)
    return sum(digits[i] * d**i for i in range(len(digits)))

def dnary_inverse(input_integer: int, d: int) -> int:
    """Calculates the inverse of the input integer in base d.

    Args:
        input_integer (int): The integer to invert.
        d (int): The modulus to invert by.

    Returns:
        int: The integer n such that n * input_integer % d = 1.
    """
    validate_integers(input_integer, d)
    if sp.isprime(d) and input_integer!=0:
        return input_integer**(d-2)%d

    return euclid_algorithm(input_integer, d)

def euclid_algorithm(input_integer:int, d:int) -> int:
    validate_integers(input_integer, d)
    if input_integer>d:
        input_integer = input_integer%d
    if not math.gcd(input_integer, d)==1:
        return None
    
    t, newt = 0, 1
    r, newr = d, input_integer

    while newr != 0:
        quotient = r//newr
        (t, newt) = (newt, t-quotient*newt) 
        (r, newr) = (newr, r-quotient*newr)

    if r > 1:
        return None
    if t < 0:
        t = t + d

    return t

def rint(i:Real)->int:
    """Rounds i to the nearest integer and returns an integer type.

    Args:
        i (Real)

    Returns:
        int
    """
    return int(np.rint(i))
