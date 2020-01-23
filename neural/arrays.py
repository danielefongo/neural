import numpy as np


def shuffle_arrays(*arrays: np.ndarray):
    if len(arrays) == 0:
        return []

    mask = np.random.permutation(len(arrays[0]))
    return tuple([array[mask] for array in arrays])


def to_batches(array: np.ndarray, batch_size):
    shape = [int(array.shape[0] / batch_size), batch_size] + list(array.shape[1:])
    return np.resize(array, shape)
