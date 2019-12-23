import numpy as np

from units import Unit
from initializers import Initializer, Zeros, Normal
from weighted_sum import WeightedSum


class Layer(Unit):
    def __init__(self, unit, activation, shape: tuple, weights_initializer: Initializer = Normal(),
                 biases_initializer: Initializer = Zeros()):
        self.weighted_sum = WeightedSum(unit, shape, weights_initializer, biases_initializer)
        self.activation = activation([self.weighted_sum])
        super().__init__([self.activation])

    def compute(self, data: np.ndarray):
        return data

    def apply(self, gradient, optimizer):
        return gradient
