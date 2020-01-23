import numpy as np

from neural.ops import divide, subtract, power, multiply, add, sqrt, square
from neural.units import Unit


class Optimizer:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.iteration = 0

    def set_epoch(self, iteration: int):
        self.iteration = iteration

    def on(self, unit, gradient: np.ndarray):
        raise NotImplementedError("Should have implemented this")


class GradientDescent(Optimizer):
    def on(self, unit: Unit, gradient: np.ndarray):
        return gradient * self.learning_rate


class Adam(GradientDescent):
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.99, epsilon: float = 1.E-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.units = {}

    def on(self, unit: Unit, gradient: np.ndarray):
        if unit not in self.units:
            self.units[unit] = dict(mean=0, variance=0)

        gradient = super().on(unit, gradient)

        mean, variance = self._mean_and_variance(unit, gradient)
        self._update(unit, mean, variance)

        mean_hat = divide(mean, subtract(1.0, power(self.beta1, self.iteration)))
        variance_hat = divide(variance, subtract(1.0, power(self.beta2, self.iteration)))

        return divide(multiply(self.learning_rate, mean_hat), add(sqrt(variance_hat), self.epsilon))

    def _update(self, unit, mean, variance):
        self.units[unit]["mean"] = mean
        self.units[unit]["variance"] = variance

    def _mean_and_variance(self, unit, gradient):
        mean = self.units[unit]["mean"]
        variance = self.units[unit]["variance"]

        mean = add(multiply(self.beta1, mean), multiply(subtract(1.0, self.beta1), gradient))
        variance = add(multiply(self.beta2, variance), multiply(subtract(1.0, self.beta2), square(gradient)))

        return mean, variance
