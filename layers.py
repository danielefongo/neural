import numpy as np

from activations import Activation
from initializers import Initializer, Random, Zeros
from units import UnitChain, Unit
from weighted_sum import WeightedSum


class Layer(UnitChain):
    def __init__(self, shape: tuple, activation: Activation, weights_initializer: Initializer = Random(),
                 biases_initializer: Initializer = Zeros()):
        super().__init__()
        self.add_list([WeightedSum(shape, weights_initializer, biases_initializer), activation])
