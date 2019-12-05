from typing import List

import numpy as np

from layers import Layer
from losses import Loss
from optimizers import Optimizer
from units import UnitChain, Unit


class Network:
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.chain = UnitChain()

    def add(self, unit: Unit):
        self.chain.add(unit)

    def predict(self, x: np.ndarray):
        return self.chain.run(x)

    def train(self, x: np.ndarray, y: np.ndarray, iterations: int, loss_function: Loss, optimizer: Optimizer):
        for i in np.arange(1, iterations+1):
            optimizer.set_iteration(i)

            self.predict(x)

            loss, d_loss = self.calculate_loss(y, loss_function)
            print(loss)

            self.backpropagate(d_loss, optimizer)

    def calculate_loss(self, y: np.ndarray, loss_function: Loss):
        predicted = self.chain.units[-1].result
        loss_value = loss_function.calculate(predicted, y)
        d_loss = loss_function.derivative(predicted, y)
        return loss_value, d_loss

    def backpropagate(self, d_loss: np.ndarray, optimizer: Optimizer):
        self.chain.apply(d_loss, optimizer)
