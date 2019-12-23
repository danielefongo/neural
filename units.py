from typing import List

import numpy as np

from arrays import sum_to_shape


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
        gradient = 0
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


class Weight(Unit):
    def __init__(self, shape, initializer):
        super().__init__()
        self.weights = initializer.generate(shape)

    def compute(self):
        return self.weights

    def apply(self, gradient: np.ndarray, optimizer):
        self.weights -= optimizer.on(self, gradient)


class Add(Unit):
    def __init__(self, a, b):
        super().__init__([a, b])

    def compute(self, a_val: np.ndarray, b_val: np.ndarray):
        return a_val + b_val

    def apply(self, gradient: np.ndarray, optimizer):
        a_val = self.inputs[0]
        b_val = self.inputs[1]

        a_gradient = sum_to_shape(gradient, a_val.shape)
        b_gradient = sum_to_shape(gradient, b_val.shape)

        return [a_gradient, b_gradient]


class MatMul(Unit):
    def __init__(self, a, b):
        super().__init__([a, b])

    def compute(self, a_val: np.ndarray, b_val: np.ndarray):
        return np.matmul(a_val, b_val)

    def apply(self, gradient: np.ndarray, optimizer):
        a_val = self.inputs[0]
        b_val = self.inputs[1]

        a_gradient = np.matmul(gradient, b_val.T)
        b_gradient = np.matmul(a_val.T, gradient)

        return [a_gradient, b_gradient]
