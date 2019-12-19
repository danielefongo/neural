from typing import Type

from arrays import shuffle_arrays, to_batches
from graphs import Placeholder, Unit
from losses import Loss
from optimizers import Optimizer


class Network:
    def __init__(self):
        self.x = Placeholder()
        self.y = Placeholder()
        self.last = self.x

    def add(self, unit: Unit):
        first = unit.plain_graph()[0]
        first.add_input_units([self.last])
        self.last = unit.plain_graph()[-1]

    def train(self, x, y, batch_size: int, epochs: int, loss_function: Type[Loss], optimizer: Optimizer, shuffle=True):
        self.loss = loss_function(self.last, self.y)
        self.optimizer = optimizer

        for epoch in range(1, epochs):
            self.optimizer.set_epoch(epoch)

            if shuffle:
                x, y = shuffle_arrays(x, y)

            batched_x = to_batches(x, batch_size)
            batched_y = to_batches(y, batch_size)

            self.x.use(x)
            self.y.use(y)

            self._train_epoch(batched_x, batched_y)

    def _train_epoch(self, batched_x, batched_y):
        loss_mean = 0
        batches_number = batched_x.shape[0]

        for batch in range(batches_number):
            self.x.use(batched_x[batch])
            self.y.use(batched_y[batch])

            loss_mean += self.loss.evaluate()

            self.loss.error(self.optimizer)

        print(loss_mean / batches_number)

    def evaluate(self, x, y):
        self.x.use(x)
        self.y.use(y)

        return self.y.evaluate()
