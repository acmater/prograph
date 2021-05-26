"""
Minkowski distance calculator
"""
import torch
import numpy as np
from .utils import clean_input

def minkowski(X, Y, p=2):
    """
    Minkowski distance calculator for two blocks of row vectors X and Y. Y will be broadcast
    across X and as such X is usually the full dataset, while Y is a subset. The two tensors
    must share their second dimension, D.

    Parameters
    ----------
    X : torch.array, shape=(NxD)
        The first tensor of row vectors to have all pairwise distances calculated.
        Usually this is the entirety of the data.

    Y : torch.array, shape=(MxD)
        The second tensor of row vectors that will be stretched across the first
        to calculate pairwise distances.

    Returns
    -------
        torch.array, shape=(NxM)
    """
    X,Y = clean_input(X,Y)
    if isinstance(X,torch.Tensor):
        return torch.pow(torch.sum(torch.pow(X - Y[:,None,:],exponent=p),axis=2),exponent=1/p) 
    else:
        return np.pow(np.sum(np.pow(X - Y[:,None,:],exponent=p),axis=2),exponent=1/p)
