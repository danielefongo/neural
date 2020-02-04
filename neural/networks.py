from typing import List

import numpy as np

from neural.arrays import shuffle_arrays, to_batches
from neural.losses import Loss
from neural.optimizers import Optimizer
from neural.units import Placeholder, Unit, Variable, Graph


class Network:
    def __init__(self):
        super().__init__()
        self.x = Placeholder()
        self.y = Placeholder()
        self.unit = self.x

    def add(self, unit):
        self.unit = unit(self.unit)

    def train(self, x: np.ndarray, y: np.ndarray, batch_size: int, epochs: int, loss_function: Loss,
              optimizer: Optimizer, shuffle=True):
        loss = Graph(loss_function(self.unit, self.y))

        for epoch in range(1, epochs + 1):
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
            self.x(batched_x[batch])
            self.y(batched_y[batch])

            loss_mean += loss.evaluate()

            loss.error(optimizer)

        return loss_mean / batches_number

    def evaluate(self, x):
        self.x(x)

        return Graph(self.unit).evaluate()

    def export(self):
        graph = Graph(self.unit)
        return graph.export(), graph.all_vars()

    def use(self, configs, variables: List[Variable] = []):
        units = Graph.use(configs)
        self.x = units[0]
        self.unit = units[-1]
        graph = Graph(self.unit)
        if not len(variables):
            return
        for new_variable, old_variable in zip(graph.all_vars(), variables):
            new_variable.value = old_variable.value