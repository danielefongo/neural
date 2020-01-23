import numpy as np


def sum_to_shape(array: np.ndarray, output_shape: tuple):
    return np.sum(np.reshape(array, [-1] + list(output_shape)), axis=0)


def shuffle_arrays(*arrays: np.ndarray):
    if len(arrays) == 0:
        return []

    mask = np.random.permutation(len(arrays[0]))
    return tuple([array[mask] for array in arrays])


def to_batches(array: np.ndarray, batch_size):
    shape = [int(array.shape[0] / batch_size), batch_size] + list(array.shape[1:])
    return np.resize(array, shape)
