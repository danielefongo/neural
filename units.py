from typing import List

import numpy as np

from optimizers import Optimizer


class Unit:
    def __init__(self):
        self.input: np.ndarray = np.empty([])
        self.result: np.ndarray = np.empty([])

    def forward(self, x: np.ndarray):
        raise NotImplementedError("Should have implemented this")

    def backward(self, d_loss: np.ndarray, optimizer: Optimizer):
        raise NotImplementedError("Should have implemented this")


class UnitChain(Unit):
    def __init__(self):
        super().__init__()
        self.units: List[Unit] = []

    def forward(self, x: np.ndarray):
        for unit in self.units:
            x = unit.forward(x)
        return x

    def backward(self, d_loss: np.ndarray, optimizer: Optimizer):
        for unit in reversed(self.units):
            d_loss = unit.backward(d_loss, optimizer)

    def add(self, unit: Unit):
        if isinstance(unit, UnitChain):
            for unit in unit.units:
                self.units.append(unit)
        else:
            self.units.append(unit)

    def add_list(self, units: List[Unit]):
        for unit in units:
            self.units.append(unit)
