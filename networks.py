from typing import Type

import numpy as np

from arrays import shuffle_arrays, to_batches
from units import Placeholder, Unit
from losses import Loss
from optimizers import Optimizer


class Network:
    def __init__(self, x: Placeholder, output_unit: Unit):
        self.x = x
        self.y = Placeholder()
        self.unit = output_unit

    def train(self, x: np.ndarray, y: np.ndarray, batch_size: int, epochs: int, loss_function: Type[Loss], optimizer: Optimizer, shuffle=True):
        loss = loss_function(self.unit, self.y)

        for epoch in range(1, epochs):
            optimizer.set_epoch(epoch)

            if shuffle:
                x, y = shuffle_arrays(x, y)

            batched_x = to_batches(x, batch_size)
            batched_y = to_batches(y, batch_size)

            loss_value = self._train_epoch(batched_x, batched_y, optimizer, loss)
            print("epoch %i: loss = %f" % (epoch, loss_value))

    def _train_epoch(self, batched_x, batched_y, optimizer, loss):
        loss_mean = 0
        batches_number = batched_x.shape[0]

        for batch in range(batches_number):
            self.x.use(batched_x[batch])
            self.y.use(batched_y[batch])

            loss_mean += loss.evaluate()

            loss.error(optimizer)

        return loss_mean / batches_number

    def evaluate(self, x, y):
        self.x.use(x)
        self.y.use(y)

        return self.y.evaluate()
