import numpy as np


class Initializer:
    def generate(self, shape: tuple):
        raise NotImplementedError("Should have implemented this")


class Random(Initializer):
    def generate(self, shape: tuple):
        return np.random.random_sample(shape)


class Normal(Initializer):
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std

    def generate(self, shape: tuple):
        return np.random.normal(loc=self.mean, scale=self.std, size=shape)


class Zeros(Initializer):
    def generate(self, shape: tuple):
        return np.zeros(shape)


class Ones(Initializer):
    def generate(self, shape: tuple):
        return np.ones(shape)
