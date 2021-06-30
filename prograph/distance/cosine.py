"""
Cosine similarity calculator
"""
import torch
import numpy as np
from .utils import clean_input

def cosine(X, Y, similarity=False):
    """
    Cosine distance calculator for two blocks of row vectors X and Y. Y will be broadcast
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

    similarity : bool, default=False
        Whether or not to return the distance values as similarity values.

    Returns
    -------
        torch.array, shape=(NxM)
    """
    X,Y = clean_input(X,Y)
    if isinstance(X,torch.Tensor):
        print("Add method.")
        #distances = torch.pow(torch.sum(torch.pow(X - Y[:,None,:],exponent=p),axis=2),exponent=1/p)
    else:
        print("Add method.")
        #distances = np.power(np.sum(np.power(X - Y[:,None,:],exponent=p),axis=2),exponent=1/p)
    if similarity:
        distances = 1/(1+distances)
    return distances
