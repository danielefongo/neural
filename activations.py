import numpy as np

from optimizers import Optimizer
from units import Unit


class Activation(Unit):
    def run(self, x: np.ndarray):
        self.input = x
        self.result = self._activate(self.input)
        return self.result

    def apply(self, d_loss: np.ndarray, optimizer: Optimizer):
        pass

    def derivative_loss(self, next_d_loss: np.ndarray = 1):
        return next_d_loss * self._derivative()

    def _activate(self, x: np.ndarray):
        raise NotImplementedError("Should have implemented this")

    def _derivative(self):
        raise NotImplementedError("Should have implemented this")


class Linear(Activation):
    def _activate(self, x: np.ndarray):
        return x

    def _derivative(self):
        return np.ones(self.result.shape)


class Sigmoid(Activation):
    def _activate(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def _derivative(self):
        return self.result * (1 - self.result)
