import numpy as np

from neural.units import Unit
from neural.ops import multiply, divide, add, subtract


class Activation(Unit):
    pass


class Linear(Activation):
    def compute(self, x: np.ndarray):
        return x

    def apply(self, gradient: np.ndarray, optimizer):
        return multiply(gradient, np.ones(self.output.shape))


class Sigmoid(Activation):
    def compute(self, x: np.ndarray):
        return divide(1.0, add(1.0, np.exp(-x)))

    def apply(self, gradient: np.ndarray, optimizer):
        return multiply(gradient, multiply(self.output, subtract(1.0, self.output)))


class Tanh(Activation):
    def compute(self, x: np.ndarray):
        return np.tanh(x)

    def apply(self, gradient: np.ndarray, optimizer):
        return multiply(gradient, subtract(1.0, np.power(self.output, 2)))


class Softmax(Activation):
    def compute(self, x: np.ndarray):
        max_on_axis = np.max(x, axis=-1)[:, np.newaxis]
        exp = np.exp(x - max_on_axis)
        total = np.sum(exp, axis=-1)[:, np.newaxis]
        return divide(exp, total)

    def apply(self, gradient: np.ndarray, optimizer):
        result = np.einsum('ij,ik->ijk', self.output, -self.output)
        np.einsum('ijj->ij', result)[...] += self.output
        return np.einsum('ij,ijk->ik', gradient, result)
