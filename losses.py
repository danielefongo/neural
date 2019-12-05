import numpy as np


class Loss:
    def calculate(self, predicted: np.ndarray, y: np.ndarray):
        raise NotImplementedError("Should have implemented this")

    def derivative(self, predicted: np.ndarray, y: np.ndarray):
        raise NotImplementedError("Should have implemented this")


class MSE(Loss):
    def calculate(self, predicted: np.ndarray, y: np.ndarray):
        return np.power(predicted - y, 2).mean()

    def derivative(self, predicted: np.ndarray, y: np.ndarray):
        return 2 * (predicted - y)


class CrossEntropy(Loss):
    def calculate(self, predicted: np.ndarray, y: np.ndarray):
        first_term = y * self._safe_log(predicted)
        second_term = (1 - y) * self._safe_log(1 - predicted)
        return -1 * np.average(first_term + second_term)

    def _safe_log(self, array):
        return np.log(array, out=np.zeros_like(array), where=(array != 0))

    def derivative(self, predicted: np.ndarray, y: np.ndarray):
        return predicted - y
