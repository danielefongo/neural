import numpy as np

import arrays
from graphs import Unit
from initializers import Initializer
from optimizers import Optimizer
from weights import Weights


class WeightedSum(Unit):
    def compute(self, data):
        self.tmp = self._add_ones_for_biases(data)
        return np.matmul(self.tmp, self.weights)

    def apply(self, d_loss, optimizer):
        d_loss_new = np.matmul(d_loss, self.weights.w.T)
        self.weights -= optimizer.on(self, self.tmp, d_loss)
        return d_loss_new

    def __init__(self, shape, weights_initializer: Initializer, biases_initializer: Initializer):
        super().__init__([])
        self.weights = Weights(shape, weights_initializer, biases_initializer)

    def _add_ones_for_biases(self, x):
        return arrays.add_column(x, axis=-1, values=1)
