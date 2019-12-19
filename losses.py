import numpy as np

from graphs import Unit, Placeholder


class Loss(Unit):
    def __init__(self, out: Unit, y: Placeholder):
        super().__init__([out, y])

    def compute(self, predicted, y):
        return [self._compute(predicted, y), 0]
        raise NotImplementedError("Should have implemented this")

    def apply(self, d_loss, optimizer):
        return [self._apply(d_loss, optimizer), 0]
        raise NotImplementedError("Should have implemented this")


class MSE(Loss):
    def compute(self, predicted: np.ndarray, y: np.ndarray):
        return np.power(predicted - y, 2).mean()

    def _apply(self, d_loss, optimizer):
        return 2 * (self.inputs[0] - self.inputs[1])


class CrossEntropy(Loss):
    def compute(self, predicted: np.ndarray, y: np.ndarray):
        first_term = y * self._safe_log(predicted)
        second_term = (1 - y) * self._safe_log(1 - predicted)
        return -1 * np.average(first_term + second_term)

    def _apply(self, d_loss, optimizer):
        return self.inputs[0] - self.inputs[1]

    def _safe_log(self, array):
        return np.log(array, out=np.zeros_like(array), where=(array != 0))
