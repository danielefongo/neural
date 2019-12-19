from graphs import Unit
from initializers import Initializer, Zeros, Normal
from weighted_sum import WeightedSum


class Layer(Unit):
    def compute(self, arg):
        return arg

    def apply(self, d_loss, optimizer):
        return d_loss

    def __init__(self, activation, shape: tuple, weights_initializer: Initializer = Normal(),
                 biases_initializer: Initializer = Zeros()):
        self.weighted_sum = WeightedSum(shape, weights_initializer, biases_initializer)
        self.activation = activation([self.weighted_sum])
        super().__init__([self.activation])
