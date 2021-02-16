def collapse_concat(arrays,dim=0):
    """
    Takes an iterable of arrays and recursively concatenates them. Functions similarly
    to the reduce operation from python's functools library.

    Parameters
    ----------
    arrays : iterable(np.array)

        Arrays contains an iterable of np.arrays

    dim : int, default=0

        The dimension on which to concatenate the arrays.

    returns : np.array

        Returns a single np array representing the concatenation of all arrays
        provided.
    """
    if len(arrays) == 1:
        return arrays[0]
    else:
        return np.concatenate((arrays[0],collapse_concat(arrays[1:])))
