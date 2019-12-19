from typing import List

import numpy as np


class Unit:
    def __init__(self, input_units: List = []):
        self.output_units = []
        self.inputs = []
        self.output = []
        self.d_loss = None

        self.add_input_units(input_units)

    def add_input_units(self, units: List):
        self.input_units: List[Unit] = units
        for element in units:
            if isinstance(element, Unit):
                element.then(self)

    def plain_graph(self):
        nodes_postorder: List[Unit] = []

        def recurse(node):
            if isinstance(node, Unit):
                for input_node in node.input_units:
                    recurse(input_node)
            if node not in nodes_postorder:
                nodes_postorder.append(node)

        recurse(self)
        return nodes_postorder

    def evaluate(self):
        nodes = self.plain_graph()
        [node._forward() for node in nodes]
        return self.output

    def error(self, optimizer):
        nodes = self.plain_graph()[::-1]
        [node._backward(optimizer) for node in nodes]

    def _forward(self):
        self.inputs = [input_node.output for input_node in self.input_units]
        self.output = self.compute(*self.inputs)

    def _backward(self, optimizer):
        d_loss = 0
        for after in self.output_units:
            if len(after.input_units) == 1:
                d_loss += after.d_loss
            else:
                index = after.input_units.index(self)
                d_loss += after.d_loss[index]

        self.d_loss = self.apply(d_loss, optimizer)

    def compute(self, *args):
        raise NotImplementedError()

    def apply(self, d_loss, optimizer):
        raise NotImplementedError()

    def then(self, self1):
        self.output_units.append(self1)


class Placeholder(Unit):
    def apply(self, d_loss, optimizer):
        return d_loss

    def compute(self):
        return self.real_data

    def __init__(self):
        super().__init__([])
        self.real_data = None

    def use(self, x):
        self.real_data = x
