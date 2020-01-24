from typing import List

import numpy as np


# basic operations
def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    return a / b


def square(a):
    return np.square(a)


def log(a):
    return np.log(a, out=np.zeros_like(a), where=(a != 0))


def exp(a):
    return np.exp(a)


def power(a, b):
    return np.power(a, b)


def sqrt(a):
    return np.sqrt(a)


def tanh(a):
    return np.tanh(a)


# matrix operations
def dot(a: np.ndarray, b: np.ndarray):
    return a.dot(b)


def einsum(indexes: str, *arrays: np.ndarray):
    return np.einsum(indexes, *arrays)


def sum_to_shape(array: np.ndarray, output_shape: tuple = ()):
    return np.sum(np.reshape(array, [-1] + list(output_shape)), axis=0)


def reduce_mean(array: np.ndarray, axis: int = None):
    return np.mean(array, axis=axis)


def reduce_sum(array: np.ndarray, axis: int = None):
    return np.sum(array, axis=axis)


def reduce_max(array: np.ndarray, axis: int = None):
    return np.max(array, axis=axis)


def merge(axis: int, *arrays: np.ndarray):
    splits = np.cumsum([a.shape[axis] for a in arrays])
    return np.concatenate(arrays, axis), splits


def unmerge(axis: int, splits: List[int], array: np.ndarray):
    return np.split(array, splits, axis)


def stack(axis: int, *arrays: np.ndarray):
    return np.stack(arrays, axis)


def unstack(axis: int, array: np.ndarray):
    number = array.shape[axis]
    return [element[:, 0] for element in np.split(array, number, axis)]


def take(axis: int, index: int, array: np.ndarray):
    return np.take(array, index, axis)


def replace(axis: int, index: int, array: np.ndarray, destination: np.ndarray):
    mask = tuple([index if actual_axis == axis else slice(None) for actual_axis in range(destination.ndim)])
    destination[mask] = array
    return destination


def reshape(array: np.ndarray, shape: tuple):
    return np.reshape(array, shape)


def add_dimension(array: np.ndarray, right: bool = True):
    if right:
        return array[..., np.newaxis]
    return array[np.newaxis, ...]


# creational operations
def zeros(shape: tuple):
    return np.zeros(shape)


def ones(shape: tuple):
    return np.ones(shape)


def normal(shape: tuple, mean: float, std: float):
    return np.random.normal(loc=mean, scale=std, size=shape)


def random(shape: tuple):
    return np.random.random_sample(shape)


def empty():
    return np.array([])
