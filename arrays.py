from typing import Union

import numpy as np


def add_column(array: np.ndarray, axis: int, values: Union[tuple, int]):
    return np.insert(array, array.shape[axis], values=values, axis=axis)


def shuffle_arrays(*arrays: np.ndarray):
    if len(arrays) == 0:
        return []

    mask = np.random.permutation(len(arrays[0]))
    return tuple([array[mask] for array in arrays])


def to_batches(array: np.ndarray, size):
    shape = list(array.shape[1:])
    shape.insert(0, size)
    shape.insert(0, int(array.shape[0] / size))

    return np.resize(array, shape)
