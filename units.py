from typing import List

import numpy as np


class Unit:
    def __init__(self, input_units: List = []):
        self.input_units: List[Unit] = input_units
        self.output_units = []
        self.inputs = []
        self.output = []
        self.gradient = None

        for element in input_units:
            if isinstance(element, Unit):
                element.output_units.append(self)

    def evaluate(self):
        nodes = self.plain_graph()
        [node._forward() for node in nodes]
        return self.output

    def error(self, optimizer):
        nodes = self.plain_graph()[::-1]
        [node._backward(optimizer) for node in nodes]

    def plain_graph(self):
        node_list: List[Unit] = []

        def recurse(node):
            if isinstance(node, Unit):
                for input_node in node.input_units:
                    recurse(input_node)
            if node not in node_list:
                node_list.append(node)

        recurse(self)
        return node_list

    def _forward(self):
        self.inputs = [input_node.output for input_node in self.input_units]
        self.output = self.compute(*self.inputs)

    def _backward(self, optimizer):
        gradient = np.array([0])
        for after in self.output_units:
            if len(after.input_units) == 1:
                gradient += after.gradient
            else:
                index = after.input_units.index(self)
                gradient += after.gradient[index]

        self.gradient = self.apply(gradient, optimizer)

    def compute(self, *args: np.ndarray):
        raise NotImplementedError()

    def apply(self, gradient: np.ndarray, optimizer):
        raise NotImplementedError()


class Placeholder(Unit):
    def apply(self, gradient: np.ndarray, optimizer):
        return gradient

    def compute(self):
        return self.real_data

    def __init__(self):
        super().__init__([])
        self.real_data = np.array([])

    def use(self, x):
        self.real_data = x
