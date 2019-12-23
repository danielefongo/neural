import numpy as np

import arrays
from initializers import Initializer
from units import Unit
from weights import Weights


class WeightedSum(Unit):
    def __init__(self, unit, shape, weights_initializer: Initializer, biases_initializer: Initializer):
        super().__init__([unit])
        self.weights = Weights(shape, weights_initializer, biases_initializer)

    def compute(self, data):
        self.tmp = self._add_ones_for_biases(data)
        return np.matmul(self.tmp, self.weights)

    def apply(self, gradient, optimizer):
        inputs_gradient = np.matmul(gradient, self.weights.w.T)
        weights_gradient = np.matmul(self.tmp.T, gradient)
        self.weights -= optimizer.on(self, weights_gradient)
        return inputs_gradient

    def _add_ones_for_biases(self, x):
        return arrays.add_column(x, axis=-1, values=1)
