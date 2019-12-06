import numpy as np

import arrays
from initializers import Initializer
from optimizers import Optimizer
from units import Unit
from weights import Weights


class WeightedSum(Unit):
    def __init__(self, shape, weights_initializer: Initializer, biases_initializer: Initializer):
        super().__init__()
        self.weights = Weights(shape, weights_initializer, biases_initializer)

    def forward(self, x: np.ndarray):
        self.input = self._add_ones_for_biases(x)
        self.result = np.matmul(self.input, self.weights)
        return self.result

    def backward(self, d_loss: np.ndarray, optimizer: Optimizer):
        d_loss_new = np.matmul(d_loss, self.weights.w.T)
        self.weights -= optimizer.on(self, self.input, d_loss)
        return d_loss_new

    def _add_ones_for_biases(self, x):
        return arrays.add_column(x, axis=-1, values=1)
