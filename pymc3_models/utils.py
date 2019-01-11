import numpy as np


def normalize(array):
    """
    Normalize values in the array to get probabilities.

    Parameters
    ----------
    array : numpy array of shape [1,]

    Returns
    -------
    A normalized array
    """
    return array/np.sum(array)
