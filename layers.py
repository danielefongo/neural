import numpy as np
import shapes

from activations import Activation
from initializers import Initializer, Random, Zeros


class Layer:
    def __init__(self, shape: tuple, activation: Activation, weights_initializer: Initializer = Random(),
                 biases_initializer: Initializer = Zeros()):
        self.activation = activation
        self.weights = weights_initializer.generate(shape)
        self.biases = biases_initializer.generate(shapes.change(shape, axis=0, value=1))
        self.shape = self.weights.shape

    def predict(self, x: np.ndarray):
        x = np.append(x, self._ones_for_bias(x), axis=-1)
        weights = self._weights_and_biases()

        return self.activation.activate(np.matmul(x, weights))

    def update(self, x: np.ndarray, predicted: np.ndarray, d_loss: np.ndarray, learning_rate: int):
        d_activation = d_loss * self.activation.derivative(predicted)
        previous_d_loss = np.matmul(d_activation, self.weights.T)

        gradient_w = self._multiply(d_activation, x)
        gradient_b = self._multiply(d_activation, self._ones_for_bias(x))

        self.weights -= gradient_w.T * learning_rate
        self.biases -= gradient_b.T * learning_rate

        return previous_d_loss

    def _multiply(self, d_activation: np.ndarray, x: np.ndarray):
        return np.matmul(d_activation.T, x) / x.shape[0]

    def _ones_for_bias(self, x: np.ndarray):
        return np.ones(shapes.change(x.shape, axis=-1, value=1))

    def _weights_and_biases(self):
        return np.append(self.weights, self.biases, axis=0)
