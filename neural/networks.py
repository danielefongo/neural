from typing import List

import numpy as np

from neural.ops import shuffle_arrays, to_batches
from neural.losses import Loss
from neural.optimizers import Optimizer
from neural.units import Variable, Graph, Input


class Network:
    def __init__(self, graph: Graph = None):
        if graph is None:
            graph = Graph(Input())
        self.graph = graph
        self.y = Input()

    def add(self, unit):
        unit = unit(self.graph.unit)
        self.graph = Graph(unit)

    def train(self, x: np.ndarray, y: np.ndarray, batch_size: int, epochs: int, loss_function: Loss,
              optimizer: Optimizer, shuffle=True):
        loss = Graph(loss_function(self.graph.unit, self.y))

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
            loss_mean += loss.evaluate(batched_x[batch], batched_y[batch])

            loss.error(optimizer)

        return loss_mean / batches_number

    def evaluate(self, x):
        return self.graph.evaluate(x)

    def export_all(self):
        return self.export_configuration(), self.export_variables()

    def export_configuration(self):
        return self.graph.export()

    def export_variables(self):
        return self.graph.all_vars()

    @staticmethod
    def use(configs, variables: List[Variable] = []):
        graph = Graph.use(configs)

        if not len(variables):
            return
        for new_variable, old_variable in zip(graph.all_vars(), variables):
            new_variable.value = old_variable.value

        return Network(graph)
