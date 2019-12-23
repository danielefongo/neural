import numpy as np

from arrays import bias_shape
from initializers import Initializer, Zeros, Normal
from units import Unit, Weight, Add, MatMul


class Layer(Unit):
    def __init__(self, unit, activation, shape: tuple, weights_initializer: Initializer = Normal(),
                 biases_initializer: Initializer = Zeros()):
        biases = Weight(bias_shape(shape), biases_initializer)
        weights = Weight(shape, weights_initializer)
        self.weighted_sum = Add(MatMul(unit, weights), biases)
        self.activation = activation(self.weighted_sum)
        super().__init__(self.activation)

    def compute(self, data: np.ndarray):
        return data

    def apply(self, gradient, optimizer):
        return gradient
