from neural.configurables import Config
from neural.ops import zeros, ones, normal, random


class Initializer(Config):
    def __init__(self, init: list = []):
        super().__init__(init)

    def generate(self, shape: tuple):
        raise NotImplementedError("Should have implemented this")


class Random(Initializer):
    def generate(self, shape: tuple):
        return random(shape)


class Normal(Initializer):
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std
        super().__init__([mean, std])

    def generate(self, shape: tuple):
        return normal(shape, self.mean, self.std)


class Zeros(Initializer):
    def generate(self, shape: tuple):
        return zeros(shape)


class Ones(Initializer):
    def generate(self, shape: tuple):
        return ones(shape)
