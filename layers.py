import numpy as np

from activations import Activation
from initializers import Initializer, Zeros, Normal
from units import Weight, Wrapper, InputPlaceholder, MatMul, Add


class WeighedSum(Wrapper):
    def __init__(self, size, weight_initializer, bias_initializer):
        self.size = size
        self.weights = Weight(weight_initializer)
        self.biases = Weight(bias_initializer)

        matmul = MatMul()(InputPlaceholder(), self.weights)
        weighted_sum = Add()(matmul, self.biases)
        super().__init__(weighted_sum)

    def compute(self, args: np.ndarray):
        if self.weights.is_empty():
            self.weights.set((args.shape[-1], self.size))
            self.biases.set((1, self.size))
        return super().compute(args)


class Layer(Wrapper):
    def __init__(self, activation: Activation, size: int, weights_initializer: Initializer = Normal(),
                 biases_initializer: Initializer = Zeros()):
        weighted_sum = WeighedSum(size, weights_initializer, biases_initializer)(InputPlaceholder())
        activation = activation(weighted_sum)
        super().__init__(activation)
