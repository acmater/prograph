"""
Utilities for distance functions
"""
import torch
import torch.nn.functional as F

def clean_input(X,Y):
    """
    Function that cleans input arrays so that they are the correct type and padded in such
    a way that their dimensions match.

    Parameters:
    -----------
    X : torch.array, shape=(NxD)
        The first tensor of row vectors to have all pairwise distances calculated.
        Usually this is the entirety of the data.

    Y : torch.array, shape=(MxD)
        The second tensor of row vectors that will be stretched across the first
        to calculate pairwise distances.

    Returns
    -------
    X, Y
    """
    X, Y = torch.as_tensor(X), torch.as_tensor(Y)
    if X.shape[1] != Y.shape[1]:
        print("X and Y have different sequence lengths (dimension 1)")
        if Y.shape[1] > X.shape[1]:
            X = F.pad(X, (0,Y.shape[1] - X.shape[1]))
        else:
            Y = F.pad(Y, (0,X.shape[1] - Y.shape[1]))
    return X, Y