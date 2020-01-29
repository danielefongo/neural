import time

import numpy as np

from neural.arrays import shuffle_arrays, to_batches
from neural.units import Placeholder, InputPlaceholder, Unit
from neural.losses import Loss
from neural.optimizers import Optimizer


class Network:
    def __init__(self):
        self.x = InputPlaceholder()
        self.y = Placeholder()
        self.unit = self.x

    def add(self, unit):
        self.unit = unit(self.unit)

    def train(self, x: np.ndarray, y: np.ndarray, batch_size: int, epochs: int, loss_function: Loss, optimizer: Optimizer, shuffle=True):
        loss = loss_function(self.unit, self.y)

        for epoch in range(1, epochs+1):
            optimizer.set_epoch(epoch)

            if shuffle:
                x, y = shuffle_arrays(x, y)

            batched_x = to_batches(x, batch_size)
            batched_y = to_batches(y, batch_size)

            epoch_start = time.time()
            loss_value = self._train_epoch(batched_x, batched_y, optimizer, loss)
            epoch_end = time.time()
            print("epoch %i: loss = %f, execution time = %is" % (epoch, loss_value, epoch_end - epoch_start))

    def _train_epoch(self, batched_x, batched_y, optimizer, loss):
        loss_mean = 0
        batches_number = batched_x.shape[0]

        for batch in range(batches_number):
            self.x(batched_x[batch])
            self.y(batched_y[batch])

            loss_mean += loss.evaluate()

            loss.error(optimizer)

        return loss_mean / batches_number

    def evaluate(self, x):
        self.x(x)

        return self.y.evaluate()

    def structure(self):
        return self.unit.structure()

    def from_structure(self, configs):
        self.unit = Unit.create(configs)
        self.x = self.obtain_placeholders(self.unit)[0]

    def obtain_placeholders(self, unit):
        candidates = []
        for candidate in unit.plain_graph():
            if candidate not in candidates and isinstance(candidate, InputPlaceholder):
                candidates.append(candidate)
        return candidates


