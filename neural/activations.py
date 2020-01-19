import numpy as np

from neural.units import Unit


class Activation(Unit):
    pass


class Linear(Activation):
    def compute(self, x: np.ndarray):
        return x

    def apply(self, gradient: np.ndarray, optimizer):
        return gradient * np.ones(self.output.shape)


class Sigmoid(Activation):
    def compute(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def apply(self, gradient: np.ndarray, optimizer):
        return gradient * self.output * (1 - self.output)


class Tanh(Activation):
    def compute(self, x: np.ndarray):
        return np.tanh(x)

    def apply(self, gradient: np.ndarray, optimizer):
        return gradient * (1.0 - np.power(self.output, 2))


class Softmax(Activation):
    def compute(self, x: np.ndarray):
        max_on_axis = np.max(x, axis=-1)[:, np.newaxis]
        exp = np.exp(x - max_on_axis)
        total = np.sum(exp, axis=-1)[:, np.newaxis]
        return exp / total

    def apply(self, gradient: np.ndarray, optimizer):
        result = np.einsum('ij,ik->ijk', self.output, -self.output)
        np.einsum('ijj->ij', result)[...] += self.output
        return np.einsum('ij,ijk->ik', gradient, result)
