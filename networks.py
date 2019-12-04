from typing import List

import numpy as np

from layers import Layer
from losses import Loss
from units import UnitChain


class Network:
    def __init__(self, input_size: int, learning_rate: float):
        self.layers: List[Layer] = []
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.chain = UnitChain()

    def add_layer(self, layer: Layer):
        self.chain.add_chain(layer)

    def predict(self, x: np.ndarray):
        return self.chain.run(x)

    def train(self, x: np.ndarray, y: np.ndarray, iterations: int, loss_function: Loss):
        for i in range(iterations):
            self.predict(x)

            loss, d_loss = self.calculate_loss(y, loss_function)
            print(loss)

            self.backpropagate(d_loss)

    def calculate_loss(self, y: np.ndarray, loss_function: Loss):
        predicted = self.chain.units[-1].result
        loss_value = loss_function.calculate(predicted, y)
        d_loss = loss_function.derivative(predicted, y)
        return loss_value, d_loss

    def backpropagate(self, d_loss: np.ndarray):
        self.chain.apply(d_loss, self.learning_rate)
