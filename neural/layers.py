import numpy as np

from neural.activations import Activation, Linear
from neural.initializers import Initializer, Zeros, Normal
from neural.units import Weight, Wrapper, Input, MatMul, Add, Recurrent, Merge


class WeightedSum(Wrapper):
    def __init__(self, size, weight_initializer, bias_initializer):
        self.size = size
        self.weights = Weight(weight_initializer)
        self.biases = Weight(bias_initializer)

        matmul = MatMul()(Input(), self.weights)
        weighted_sum = Add()(matmul, self.biases)
        super().__init__(weighted_sum)

    def compute(self, args: np.ndarray):
        if self.weights.is_empty():
            self.weights.set((args.shape[-1], self.size))
            self.biases.set((1, self.size))
        return super().compute(args)


class Layer(Wrapper):
    def __init__(self, size: int, activation: Activation = Linear(), weights_initializer: Initializer = Normal(),
                 biases_initializer: Initializer = Zeros()):
        weighted_sum = WeightedSum(size, weights_initializer, biases_initializer)(Input())
        activation = activation(weighted_sum)
        super().__init__(activation)


class SimpleRNN(Recurrent):
    def __init__(self, size, timeseries_length, activation=Linear(), weight_initializer=Normal(), bias_initializer=Zeros(), return_sequences=False):
        merge = Merge()(Input(), Input())
        layer = Layer(size, activation, weight_initializer, bias_initializer)(merge)
        super().__init__(layer, size, timeseries_length, return_sequences)
