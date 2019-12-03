import numpy as np


class Activation:
    def activate(self, x: np.ndarray):
        raise NotImplementedError("Should have implemented this")

    def derivative(self, predicted: np.ndarray):
        raise NotImplementedError("Should have implemented this")


class Linear(Activation):
    def activate(self, x: np.ndarray):
        return x

    def derivative(self, predicted: np.ndarray):
        return np.ones(predicted.shape)


class Sigmoid(Activation):
    def activate(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def derivative(self, predicted: np.ndarray):
        return predicted * (1 - predicted)
