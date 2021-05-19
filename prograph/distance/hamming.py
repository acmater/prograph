"""
Hamming distance calculator
"""
import torch

def hamming(X, Y):
    """
    Hamming distance calculator for two blocks of row vectors X and Y. Y will be broadcast
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
    return torch.sum(X != Y[:,None,:],axis=2)
