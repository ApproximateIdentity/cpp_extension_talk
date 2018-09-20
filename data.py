from random import random, shuffle
from time import time
import numpy as np


def generate_data(size, mean1, mean2):
    """
    Generate 2-dimensional data around mean1 and mean2. There is an equal
    number of points in each cluster and the data is distributed around each
    mean in a disc of size 1. One cluster is given the label 1 and the other
    -1.

    Parameters
    ----------
    size : int
    mean1 : tuple of floats of length 2
    mean2 : tuple of floats of length 2
    """
    X, Y = generate_data_np(size, mean1, mean2)
    py_X = [tuple(point) for point in X.tolist()]
    py_Y = Y.tolist()
    return py_X, py_Y


def generate_data_np(size, mean1, mean2):
    """
    Generate 2-dimensional data around mean1 and mean2. There is an equal
    number of points in each cluster and the data is distributed around each
    mean in a disc of size 1. One cluster is given the label 1 and the other
    -1.

    Parameters
    ----------
    size : int
    mean1 : tuple of floats of length 2
    mean2 : tuple of floats of length 2
    """
    X = np.concatenate([(2 * np.random.rand(size // 2, 2) - 1) + mean1,
                        (2 * np.random.rand(size // 2, 2) - 1) + mean2])
    Y = np.concatenate([np.full(size // 2, -1, dtype=int),
                        np.full(size // 2, 1, dtype=int)])
    p = np.random.permutation(len(X))
    return X[p], Y[p]


class Timer:
    def __init__(self):
        self._time = time()
    def split(self):
        new_time = time()
        split_time = new_time - self._time
        self._time = new_time
        return split_time
