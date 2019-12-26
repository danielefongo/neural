import numpy as np

from units import Unit, Placeholder


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
        return np.power(predicted - y, 2).mean()

    def _apply(self, gradient: np.ndarray, optimizer):
        return 2 * (self.inputs[0] - self.inputs[1]) / self.inputs[0].shape[0]


class CrossEntropy(Loss):
    def _compute(self, predicted: np.ndarray, y: np.ndarray):
        first_term = y * self._safe_log(predicted)
        second_term = (1 - y) * self._safe_log(1 - predicted)
        return -1 * np.average(first_term + second_term)

    def _apply(self, gradient: np.ndarray, optimizer):
        return (self.inputs[0] - self.inputs[1]) / self.inputs[0].shape[0]

    def _safe_log(self, array: np.ndarray):
        return np.log(array, out=np.zeros_like(array), where=(array != 0))
