import numpy as np

from units import Unit


class Activation(Unit):
    def compute(self, data: np.ndarray):
        return self._activate(data)

    def apply(self, gradient: np.ndarray, optimizer):
        return gradient * self._derivative()

    def _activate(self, x: np.ndarray):
        raise NotImplementedError("Should have implemented this")

    def _derivative(self):
        raise NotImplementedError("Should have implemented this")


class Linear(Activation):
    def _activate(self, x: np.ndarray):
        return x

    def _derivative(self):
        return np.ones(self.output.shape)


class Sigmoid(Activation):
    def _activate(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def _derivative(self):
        return self.output * (1 - self.output)


class Tanh(Activation):
    def _activate(self, x: np.ndarray):
        return np.tanh(x)

    def _derivative(self):
        return 1.0 - np.power(self.output, 2)
