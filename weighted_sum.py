import numpy as np

import arrays
from initializers import Initializer
from units import Unit
from weights import Weights


class WeightedSum(Unit):
    def __init__(self, shape, weights_initializer: Initializer, biases_initializer: Initializer):
        self.weights = Weights(shape, weights_initializer, biases_initializer)

    def run(self, x: np.ndarray):
        self.input = self._add_ones_for_biases(x)
        self.result = np.matmul(self.input, self.weights)
        return self.result

    def apply(self, d_loss: np.ndarray, learning_rate: float):
        gradient = np.matmul(d_loss.T, self.input) / self.input.shape[0]
        self.weights -= gradient.T * learning_rate

    def derivative_loss(self, next_d_loss: np.ndarray = 1):
        return np.matmul(next_d_loss, self.weights.w.T)

    def _add_ones_for_biases(self, x):
        return arrays.add_column(x, axis=-1, values=1)
