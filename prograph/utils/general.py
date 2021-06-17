"""
Some general python utility functions.
"""
import numpy as np

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
    Uses np.allclose to determine whether or not an array is symmetric by comparing
    it to its transpose.
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def flatten(lst):
    """
    Flattens a list of lists using a list comprehension.
    """
    return [item for sublist in lst for item in sublist]
