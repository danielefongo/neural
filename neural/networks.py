import numpy as np

from neural.arrays import shuffle_arrays, to_batches
from neural.losses import Loss
from neural.optimizers import Optimizer
from neural.units import Placeholder, Unit


class Network:
    def __init__(self):
        super().__init__()
        self.x = Placeholder()
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

        return self.y.evaluate()

    def export(self):
        return self.unit.export_graph()

    def use(self, configs):
        units = Unit.generate_graph(configs)
        self.x = units[0]
        self.unit = units[-1]
