import copy
from typing import List

import numpy as np

from arrays import sum_to_shape


class Unit:
    def __init__(self):
        self.input_units = []
        self.output_units = []
        self.inputs = []
        self.output = []
        self.gradient = None

    def __call__(self, *input_units):
        self._remove_all_inputs()

        self.input_units: List[Unit] = list(input_units)
        for element in input_units:
            if isinstance(element, Unit):
                element.output_units.append(self)
        return self

    def _remove_all_inputs(self):
        for element in copy.copy(self.input_units):
            element.output_units.remove(self)
            self.input_units.remove(element)

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


class Variable:
    def __init__(self, value):
        self._value = value

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value

    value = property(get_value, set_value)


class Placeholder(Unit):
    def __init__(self):
        super().__init__()
        self.real_data = np.array([])

    def __call__(self, x):
        self.real_data = x
        return self

    def apply(self, gradient: np.ndarray, optimizer):
        return gradient

    def compute(self):
        return self.real_data


class InputPlaceholder(Placeholder):
    pass


class Weight(Unit):
    def __init__(self, initializer):
        super().__init__()
        self.initializer = initializer
        self.weights = None

    def is_empty(self):
        return self.weights is None

    def set(self, shape):
        if not self.is_empty():
            return

        self.weights = Variable(self.initializer.generate(shape))

    def compute(self):
        return self.weights.value

    def apply(self, gradient: np.ndarray, optimizer):
        self.weights.value -= optimizer.on(self, gradient)


class Add(Unit):
    def __call__(self, a, b):
        return super().__call__(a, b)

    def compute(self, a_val: np.ndarray, b_val: np.ndarray):
        return a_val + b_val

    def apply(self, gradient: np.ndarray, optimizer):
        a_val = self.inputs[0]
        b_val = self.inputs[1]

        a_gradient = sum_to_shape(gradient, a_val.shape)
        b_gradient = sum_to_shape(gradient, b_val.shape)

        return [a_gradient, b_gradient]


class MatMul(Unit):
    def __call__(self, a, b):
        return super().__call__(a, b)

    def compute(self, a_val: np.ndarray, b_val: np.ndarray):
        return np.matmul(a_val, b_val)

    def apply(self, gradient: np.ndarray, optimizer):
        a_val = self.inputs[0]
        b_val = self.inputs[1]

        a_gradient = np.matmul(gradient, b_val.T)
        b_gradient = np.matmul(a_val.T, gradient)

        return [a_gradient, b_gradient]


class Wrapper(Unit):
    def __init__(self, unit):
        self.fake_output: Unit = Unit()
        self.fake_inputs = self.obtain_placeholders(unit)

        self.unit: Unit = unit
        self.fake_output(self.unit)

        super().__init__()

    def obtain_placeholders(self, unit):
        candidates = []
        for candidate in unit.plain_graph():
            if candidate not in candidates and isinstance(candidate, InputPlaceholder):
                candidates.append(candidate)
        return candidates

    def compute(self, *args: np.ndarray):
        for index in range(len(self.fake_inputs)):
            self.fake_inputs[index](args[index])

        return self.unit.evaluate()

    def apply(self, gradient: np.ndarray, optimizer):
        self.fake_output.gradient = gradient

        self.unit.error(optimizer)

        gradients = [unit.gradient for unit in self.fake_inputs]
        return gradients if len(gradients) > 1 else gradients[0]
