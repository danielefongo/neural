import numpy as np


class Unit:
    def run(self, x: np.ndarray):
        raise NotImplementedError("Should have implemented this")

    def apply(self, d_loss: np.ndarray, learning_rate: float):
        raise NotImplementedError("Should have implemented this")

    def derivative_loss(self, next_d_loss: np.ndarray = 1):
        raise NotImplementedError("Should have implemented this")
