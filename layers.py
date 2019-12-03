import numpy as np

from activations import Activation


class Layer:
    def __init__(self, input_size: int, output_size: int, activation: Activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = np.random.rand(output_size, input_size)

    def predict(self, x: np.ndarray):
        return self.activation.activate(np.matmul(x, self.weights.T))

    def update(self, x: np.ndarray, predicted: np.ndarray, d_loss: np.ndarray, learning_rate: int):
        d_activation = d_loss * self.activation.derivative(predicted)
        previous_d_loss = np.matmul(d_activation, self.weights)

        gradient = np.matmul(x.T, d_activation) / x.shape[0]
        self.weights -= gradient.T * learning_rate

        return previous_d_loss
