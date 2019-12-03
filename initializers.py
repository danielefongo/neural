import numpy as np


class Initializer:
    def generate(self, shape: tuple):
        raise NotImplementedError("Should have implemented this")


class Random(Initializer):
    def generate(self, shape: tuple):
        return np.random.random_sample(shape)


class Normal(Initializer):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def generate(self, shape: tuple):
        return np.random.normal(loc=self.mean, scale=self.std, size=shape)
