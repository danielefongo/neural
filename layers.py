import numpy as np

from activations import Activation
from initializers import Initializer, Random

class Layer:
    def __init__(self, shape: tuple, activation: Activation, initializer: Initializer = Random()):
        self.shape = shape
        self.activation = activation
        self.weights = initializer.generate(shape)

    def predict(self, x: np.ndarray):
        return self.activation.activate(np.matmul(x, self.weights))

    def update(self, x: np.ndarray, predicted: np.ndarray, d_loss: np.ndarray, learning_rate: int):
        d_activation = d_loss * self.activation.derivative(predicted)
        previous_d_loss = np.matmul(d_activation, self.weights.T)

        gradient = np.matmul(d_activation.T, x) / x.shape[0]
        self.weights -= gradient.T * learning_rate

        return previous_d_loss
