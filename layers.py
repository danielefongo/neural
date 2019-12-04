import numpy as np

import arrays
from activations import Activation
from initializers import Initializer, Random, Zeros
from weights import Weights


class Layer:
    def __init__(self, shape: tuple, activation: Activation, weights_initializer: Initializer = Random(),
                 biases_initializer: Initializer = Zeros()):
        self.activation = activation
        self.weights = Weights(shape, weights_initializer, biases_initializer)
        self.shape = self.weights.shape

    def predict(self, x: np.ndarray):
        x = self._add_ones_for_biases(x)

        return self.activation.activate(np.matmul(x, self.weights))

    def update(self, x: np.ndarray, predicted: np.ndarray, d_loss: np.ndarray, learning_rate: float):
        d_activation = d_loss * self.activation.derivative(predicted)
        previous_d_loss = np.matmul(d_activation, self.weights.w.T)

        x = self._add_ones_for_biases(x)
        gradient = np.matmul(d_activation.T, x) / x.shape[0]
        self.weights -= gradient.T * learning_rate

        return previous_d_loss

    def _add_ones_for_biases(self, x):
        return arrays.add_column(x, axis=-1, values=1)
