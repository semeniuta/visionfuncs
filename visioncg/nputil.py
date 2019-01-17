import numpy as np


def indices_cond(x, cond_func):
    """
    Get indices of an array for which the supplied
    condition is true.
    """

    cond_res = cond_func(x)

    n = len(x)
    all_indices = np.arange(n, dtype=np.uint64)

    return all_indices[cond_res]


def close_to_zero(x, tol=1e-6):
    """
    Condition applied to an array that 
    marks which elements of the array 
    have values close to zero.
    """
    return np.bitwise_and(x > -tol, x < tol)


def arrays_are_identical(x1, x2):
    """
    Returns True of two arrays have identical values.
    Should be applied to arrays of booleans.
    """
    return np.all(x1 == x2)
    
    


