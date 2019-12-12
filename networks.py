import numpy as np

from arrays import shuffle_arrays, to_batches
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
        return self.chain.forward(x)

    def train(self, x: np.ndarray, y: np.ndarray, batch_size: int, epochs: int, loss_function: Loss, optimizer: Optimizer, shuffle=True):
        for epoch in range(1, epochs + 1):
            optimizer.set_epoch(epoch)

            if shuffle:
                x, y = shuffle_arrays(x, y)

            batched_x = to_batches(x, batch_size)
            batched_y = to_batches(y, batch_size)

            loss = self._train_epoch(batched_x, batched_y, loss_function, optimizer)
            print("epoch %i: loss = %f" %(epoch, loss))

    def _train_epoch(self, batched_x, batched_y, loss_function, optimizer):
        loss_mean = 0
        batches_number = batched_x.shape[0]

        for batch in range(batches_number):
            predicted = self.predict(batched_x[batch])

            loss, d_loss = self.calculate_loss(predicted, batched_y[batch], loss_function)
            loss_mean += loss

            self.backpropagate(d_loss, optimizer)

        return loss_mean / batches_number

    def calculate_loss(self, predicted: np.ndarray, y: np.ndarray, loss_function: Loss):
        loss_value = loss_function.calculate(predicted, y)
        d_loss = loss_function.derivative(predicted, y)
        return loss_value, d_loss

    def backpropagate(self, d_loss: np.ndarray, optimizer: Optimizer):
        self.chain.backward(d_loss, optimizer)
