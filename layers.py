import numpy as np

from activations import Activation
from initializers import Initializer, Random, Zeros
from units import UnitChain
from weighted_sum import WeightedSum


class Layer:
    def __init__(self, shape: tuple, activation: Activation, weights_initializer: Initializer = Random(),
                 biases_initializer: Initializer = Zeros()):
        self.chain: UnitChain = UnitChain()
        self.chain.add([WeightedSum(shape, weights_initializer, biases_initializer), activation])

    def predict(self, x: np.ndarray):
        return self.chain.run(x)

    def update(self, d_loss: np.ndarray, learning_rate: float):
        previous_d_loss = self.chain.derivative_loss(d_loss)
        self.chain.apply(d_loss, learning_rate)
        return previous_d_loss
