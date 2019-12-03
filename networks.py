import numpy as np

from typing import List
from layers import Layer
from activations import Activation
from losses import Loss


class Network:
    def __init__(self, input_size: int, learning_rate: float):
        self.layers: List[Layer] = []
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.outputs = []

    def add_layer(self, layer: Layer):
        if len(self.layers) == 0:
            input_size = self.input_size
        else:
            input_size = self.layers[len(self.layers) - 1].shape[-1]

        self.layers.append(layer)
        self.outputs.append([])

    def train(self, x: np.ndarray, y: np.ndarray, iterations: int, loss_function: Loss):
        for i in range(iterations):
            self.predict(x)

            loss, d_loss = self.calculate_loss(y, loss_function)
            print(loss)

            self.backpropagate(x, d_loss)

    def calculate_loss(self, y: np.ndarray, loss_function: Loss):
        predicted = self._out(len(self.layers) - 1)
        loss_value = loss_function.calculate(predicted, y)
        d_loss = loss_function.derivative(predicted, y)
        return loss_value, d_loss

    def backpropagate(self, x: np.ndarray, d_loss: np.ndarray):
        for layer_id in np.arange(len(self.layers) - 1, 0, -1):
            d_loss = self._update(layer_id, self.outputs[layer_id - 1], d_loss)

        self._update(0, x, d_loss)

    def predict(self, x: np.ndarray):
        self.outputs[0] = self._predict(x, 0)
        for layer_id in np.arange(1, len(self.layers)):
            self.outputs[layer_id] = self._predict(self._out(layer_id - 1), layer_id)

        return self._out(len(self.layers) - 1)

    def _update(self, layer_id: int, x: np.ndarray, d_loss: np.ndarray):
        y = self.outputs[layer_id]
        return self.layers[layer_id].update(x, y, d_loss, self.learning_rate)

    def _out(self, layer_id: int):
        return self.outputs[layer_id]

    def _predict(self, x: np.ndarray, layer_id: int):
        return self.layers[layer_id].predict(x)
