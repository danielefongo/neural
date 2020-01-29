import numpy as np

from neural.activations import Activation, Linear
from neural.initializers import Initializer, Zeros, Normal
from neural.units import Weight, Wrapper, InputPlaceholder, MatMul, Add, Recurrent, Merge


class WeightedSum(Wrapper):
    def __init__(self, size, weight_initializer, bias_initializer):
        self.size = size
        self.weights = Weight(weight_initializer)
        self.biases = Weight(bias_initializer)

        matmul = MatMul()(InputPlaceholder(), self.weights)
        weighted_sum = Add()(matmul, self.biases)
        super().__init__(weighted_sum, [size, weight_initializer, bias_initializer])

    def compute(self, args: np.ndarray):
        if self.weights.is_empty():
            self.weights.set((args.shape[-1], self.size))
            self.biases.set((1, self.size))
        return super().compute(args)

class Layer(Wrapper):
    def __init__(self, size: int, activation: Activation = Linear(), weights_initializer: Initializer = Normal(),
                 biases_initializer: Initializer = Zeros()):
        self.weighted_sum = WeightedSum(size, weights_initializer, biases_initializer)(InputPlaceholder())
        activation = activation(self.weighted_sum)
        super().__init__(activation, size, activation, weights_initializer, biases_initializer)


class SimpleRNN(Recurrent):
    def __init__(self, size, timeseries_length, activation=Linear(), weight_initializer=Normal(), bias_initializer=Zeros(), return_sequences=False):
        merge = Merge()(InputPlaceholder(), InputPlaceholder())
        layer = Layer(size, activation, weight_initializer, bias_initializer)(merge)
        super().__init__(layer, size, timeseries_length, return_sequences)
