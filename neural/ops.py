from typing import List

import numpy as np


# arithmetic operations
def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    return a / b


# matrix operations
def dot(a: np.ndarray, b: np.ndarray):
    return a.dot(b)


def reduce_sum(array: np.ndarray, output_shape: tuple = ()):
    return np.sum(np.reshape(array, [-1] + list(output_shape)), axis=0)


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
    calculated_gradient = np.delete(destination, index, axis)
    return np.insert(calculated_gradient, index, array, axis)


def reshape(array: np.ndarray, shape: tuple):
    return np.reshape(array, shape)


# creational operations
def zeros(shape: tuple):
    return np.zeros(shape)


def ones(shape: tuple):
    return np.ones(shape)


def normal(shape: tuple, mean: float, std: float):
    return np.random.normal(loc=mean, scale=std, size=shape)


def random(shape: tuple):
    return np.random.random_sample(shape)
