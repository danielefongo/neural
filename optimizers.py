import numpy as np


class Optimizer:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.iteration = 0

    def set_iteration(self, iteration: int):
        self.iteration = iteration

    def on(self, layer, inputs, d_loss):
        raise NotImplementedError("Should have implemented this")


class GradientDescent(Optimizer):
    def on(self, layer, inputs, d_loss):
        gradient = (np.matmul(d_loss.T, inputs) / inputs.shape[0])
        return gradient.T * self.learning_rate


class Adam(GradientDescent):
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.99, epsilon: float = 1.E-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.units = {}

    def on(self, layer, inputs, d_loss):
        if layer not in self.units:
            self.units[layer] = dict(mean = 0, variance = 0)

        gradient = super().on(layer, inputs, d_loss)

        mean, variance = self._mean_and_variance(layer, gradient)
        self._update(layer, mean, variance)

        mean_hat = mean / (1 - np.power(self.beta1, self.iteration))
        variance_hat = variance / (1 - np.power(self.beta2, self.iteration))

        return (self.learning_rate * mean_hat) / (np.sqrt(variance_hat) + self.epsilon)

    def _update(self, layer, mean, variance):
        self.units[layer]["mean"] = mean
        self.units[layer]["variance"] = variance

    def _mean_and_variance(self, layer, gradient):
        mean = self.units[layer]["mean"]
        variance = self.units[layer]["variance"]
        mean = self.beta1 * mean + (1 - self.beta1) * gradient
        variance = self.beta2 * variance + (1 - self.beta2) * np.power(gradient, 2)
        return mean, variance
