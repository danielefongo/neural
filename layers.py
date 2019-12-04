import numpy as np

from activations import Activation
from initializers import Initializer, Random, Zeros
from weighted_sum import WeightedSum


class Layer:
    def __init__(self, shape: tuple, activation: Activation, weights_initializer: Initializer = Random(),
                 biases_initializer: Initializer = Zeros()):
        self.activation = activation
        self.weighted_sum = WeightedSum(shape, weights_initializer, biases_initializer)

    def predict(self, x: np.ndarray):
        return self.activation.activate(self.weighted_sum.run(x))

    def update(self, predicted: np.ndarray, d_loss: np.ndarray, learning_rate: float):
        d_activation = d_loss * self.activation.derivative(predicted)
        previous_d_loss = self.weighted_sum.derivative_loss(d_activation)

        self.weighted_sum.apply(d_activation, learning_rate)

        return previous_d_loss
