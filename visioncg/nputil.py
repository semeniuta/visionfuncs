import numpy as np


def indices_bool_arr(x, bool_arr):
    """
    Returns indices of an array x for which 
    the corresponding value in bool_arr is true.
    """

    all_indices = np.arange(len(x), dtype=np.uint64)
    return all_indices[bool_arr]


def indices_cond(x, cond_func):
    """
    Returns indices of an array for which the supplied
    condition is true.
    """

    cond_res = cond_func(x)
    return indices_bool_arr(x, cond_res)


def close_to_zero(x, tol=1e-6):
    """
    Condition applied to an array that 
    marks which elements of the array 
    have values close to zero.
    """

    return np.bitwise_and(x > -tol, x < tol)


def not_inf(x):
    """
    Condition applied to an array that 
    marks which elements of the array 
    have values not in set {-inf, inf}.
    """

    return np.abs(x) != np.inf


def bool_arr_diff(a, b):
    """
    Returns a boolean array representing 
    a set difference (a \ b).
    """

    return np.bitwise_and(a, np.bitwise_not(b))


def apply_multiple_conditions(x, *conds):
    """
    Applies multiple conditions to an array
    and returns bitwise_and of all the outputs.
    """

    cond_bool_arrays = [cnd(x) for cnd in conds]
    return np.bitwise_and(*cond_bool_arrays)


def arrays_are_identical(x1, x2):
    """
    Returns True of two arrays have identical values.
    Should be applied to arrays of booleans.
    """

    return np.all(x1 == x2)
    
    


