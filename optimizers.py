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
