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

    def apply(self, d_loss, optimizer):
        d_loss_inputs = np.matmul(d_loss, self.weights.w.T)
        d_loss_weights = np.matmul(self.tmp.T, d_loss)
        self.weights -= optimizer.on(self, d_loss_weights)
        return d_loss_inputs

    def _add_ones_for_biases(self, x):
        return arrays.add_column(x, axis=-1, values=1)
