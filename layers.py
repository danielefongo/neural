from activations import Activation
from arrays import bias_shape
from initializers import Initializer, Zeros, Normal
from units import Weight, Add, Wrapper, MatMul, UnitPlaceholder


class Layer(Wrapper):
    def __init__(self, activation: Activation, shape: tuple, weights_initializer: Initializer = Normal(),
                 biases_initializer: Initializer = Zeros()):
        self.weights = Weight(shape, weights_initializer)
        self.biases = Weight(bias_shape(shape), biases_initializer)

        matmul = MatMul()(UnitPlaceholder(), self.weights)
        weighted_sum = Add()(matmul, self.biases)
        activation = activation(weighted_sum)
        super(Layer, self).__init__(activation)
