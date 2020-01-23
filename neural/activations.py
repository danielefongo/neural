import numpy as np

from neural.units import Unit
from neural.ops import multiply, divide, add, subtract, ones, exp, tanh, add_dimension, reduce_max, reduce_sum, einsum, \
    square


class Activation(Unit):
    pass


class Linear(Activation):
    def compute(self, x: np.ndarray):
        return x

    def apply(self, gradient: np.ndarray, optimizer):
        return multiply(gradient, ones(self.output.shape))


class Sigmoid(Activation):
    def compute(self, x: np.ndarray):
        return divide(1.0, add(1.0, exp(-x)))

    def apply(self, gradient: np.ndarray, optimizer):
        return multiply(gradient, multiply(self.output, subtract(1.0, self.output)))


class Tanh(Activation):
    def compute(self, x: np.ndarray):
        return tanh(x)

    def apply(self, gradient: np.ndarray, optimizer):
        return multiply(gradient, subtract(1.0, square(self.output)))


class Softmax(Activation):
    def compute(self, x: np.ndarray):
        max_on_axis = add_dimension(reduce_max(x, axis=-1))
        exp_value = exp(x - max_on_axis)
        total = add_dimension(reduce_sum(exp_value, axis=-1))
        return divide(exp_value, total)

    def apply(self, gradient: np.ndarray, optimizer):
        result = einsum('ij,ik->ijk', self.output, -self.output)
        einsum('ijj->ij', result)[...] += self.output
        return einsum('ij,ijk->ik', gradient, result)
