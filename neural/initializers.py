from neural.ops import zeros, ones, normal, random


class Initializer:
    def generate(self, shape: tuple):
        raise NotImplementedError("Should have implemented this")


class Random(Initializer):
    def generate(self, shape: tuple):
        return random(shape)


class Normal(Initializer):
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std

    def generate(self, shape: tuple):
        return normal(shape, self.mean, self.std)


class Zeros(Initializer):
    def generate(self, shape: tuple):
        return zeros(shape)


class Ones(Initializer):
    def generate(self, shape: tuple):
        return ones(shape)
