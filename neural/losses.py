import numpy as np

from neural.units import Unit, Placeholder
from neural.ops import square, reduce_mean, log, multiply, add, subtract, divide


class Loss(Unit):
    def __call__(self, out: Unit, y: Placeholder):
        return super().__call__(out, y)

    def compute(self, predicted: np.ndarray, y: np.ndarray):
        return self._compute(predicted, y)

    def apply(self, gradient: np.ndarray, optimizer):
        return [self._apply(gradient, optimizer), 0]

    def _compute(self, predicted: np.ndarray, y: np.ndarray):
        raise NotImplementedError("Should have implemented this")

    def _apply(self, gradient: np.ndarray, optimizer):
        raise NotImplementedError("Should have implemented this")


class MSE(Loss):
    def _compute(self, predicted: np.ndarray, y: np.ndarray):
        return reduce_mean(square(predicted - y))

    def _apply(self, gradient: np.ndarray, optimizer):
        return divide(multiply(2.0, subtract(self.inputs[0], self.inputs[1])), self.inputs[0].shape[0])


class CrossEntropy(Loss):
    def _compute(self, predicted: np.ndarray, y: np.ndarray):
        first_term = multiply(y, log(predicted))
        second_term = multiply(subtract(1.0, y), log(1 - predicted))
        return multiply(-1.0, reduce_mean(add(first_term, second_term)))

    def _apply(self, gradient: np.ndarray, optimizer):
        return divide(subtract(self.inputs[0], self.inputs[1]), self.inputs[0].shape[0])
