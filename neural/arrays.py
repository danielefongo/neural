from typing import Union

import numpy as np


def add_column(array: np.ndarray, axis: int, values: Union[tuple, int]):
    return np.insert(array, array.shape[axis], values=values, axis=axis)


def bias_shape(shape):
    shape = list(shape)
    shape[0] = 1
    return tuple(shape)


def sum_to_shape(array: np.ndarray, output_shape: tuple):
    helper_shape = list(output_shape)
    helper_shape.insert(0, -1)

    array = np.sum(np.reshape(array, helper_shape), axis=0)

    return array


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
