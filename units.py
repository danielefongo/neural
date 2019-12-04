from typing import List

import numpy as np


class Unit:
    def __init__(self):
        self.input: np.ndarray = np.empty([])
        self.result: np.ndarray = np.empty([])

    def run(self, x: np.ndarray):
        raise NotImplementedError("Should have implemented this")

    def apply(self, d_loss: np.ndarray, learning_rate: float):
        raise NotImplementedError("Should have implemented this")

    def derivative_loss(self, next_d_loss: np.ndarray = 1):
        raise NotImplementedError("Should have implemented this")


class UnitChain(Unit):
    def __init__(self):
        super().__init__()
        self.units: List[Unit] = []

    def run(self, x: np.ndarray):
        for unit in self.units:
            x = unit.run(x)
        return x

    def apply(self, d_loss: np.ndarray, learning_rate: float):
        for unit in reversed(self.units):
            d_loss_new = unit.derivative_loss(d_loss)
            unit.apply(d_loss, learning_rate)
            d_loss = d_loss_new

    def derivative_loss(self, next_d_loss: np.ndarray = 1):
        d_loss = next_d_loss
        for unit in reversed(self.units):
            d_loss = unit.derivative_loss(d_loss)

        return d_loss

    def add(self, unit: Unit):
        self.units.append(unit)

    def add_list(self, units: List[Unit]):
        for unit in units:
            self.units.append(unit)

    def add_chain(self, unitChain):
        if isinstance(unitChain, UnitChain):
            for unit in unitChain.units:
                self.units.append(unit)
