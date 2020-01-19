import copy
from typing import List

import numpy as np

from neural.arrays import sum_to_shape


class Unit:
    def __init__(self):
        self.input_units = []
        self.output_units = []
        self.inputs = []
        self.output = []
        self.gradient = None
        self.plain = []

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __call__(self, *input_units):
        self._remove_all_inputs()

        self.input_units: List[Unit] = list(input_units)
        for element in input_units:
            if isinstance(element, Unit):
                element.output_units.append(self)

        self.plain = self.plain_graph()
        return self

    def _remove_all_inputs(self):
        for element in copy.copy(self.input_units):
            element.output_units.remove(self)
            self.input_units.remove(element)

    def evaluate(self):
        [node._forward() for node in self.plain]
        return self.output

    def error(self, optimizer):
        [node._backward(optimizer) for node in self.plain[::-1]]

    def copy(self):
        return copy.deepcopy(self)

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
    def __init__(self, value=None):
        self._value = value

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value

    def is_empty(self):
        return self._value is None

    def __deepcopy__(self, memo):
        return self

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
        self.weights = Variable()

    def is_empty(self):
        return self.weights.is_empty()

    def set(self, shape):
        if not self.is_empty():
            return

        self.weights.value = self.initializer.generate(shape)

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


class Multiply(Unit):
    def __call__(self, a, b):
        return super().__call__(a, b)

    def compute(self, a_val: np.ndarray, b_val: np.ndarray):
        return a_val * b_val

    def apply(self, gradient: np.ndarray, optimizer):
        a_val = self.inputs[0]
        b_val = self.inputs[1]

        return [gradient * a_val, gradient * b_val]


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


class Merge(Unit):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def compute(self, *args):
        self.number = len(args)
        if len(args) > 1:
            self.splits = np.cumsum([a.shape[self.axis] for a in args])
            return np.concatenate(args, self.axis)
        return args[0]

    def apply(self, gradient: np.ndarray, optimizer):
        if self.number > 1:
            return np.split(gradient, self.splits, self.axis)
        return gradient


class Stack(Unit):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def compute(self, *args):
        self.number = len(args)
        if self.number > 1:
            return np.stack(args, self.axis)
        return args[0]

    def apply(self, gradient: np.ndarray, optimizer):
        if self.number > 1:
            return [array[:, 0] for array in np.split(gradient, self.number, self.axis)]
        return gradient


class Take(Unit):
    def __init__(self, index, axis=1):
        super().__init__()
        self.index = index
        self.axis = axis

    def compute(self, args: np.ndarray):
        self.shape = args.shape
        return np.take(args, self.index, self.axis)

    def apply(self, gradient: np.ndarray, optimizer):
        calculated_gradient = np.delete(np.zeros(self.shape), self.index, self.axis)
        return np.insert(calculated_gradient, self.index, gradient, self.axis)


class Flatten(Unit):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def compute(self, args: np.ndarray):
        self.shape = args.shape
        return np.reshape(args, args.shape[:self.axis] + (-1,))

    def apply(self, gradient: np.ndarray, optimizer):
        return np.reshape(gradient, self.shape)


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


class Recurrent(Wrapper):
    def __init__(self, unit, size, timeseries_length, return_sequences=False):
        self.size = size
        self.timeseries_length = timeseries_length
        self.return_sequences = return_sequences
        self.zero = Placeholder()

        self.units = self._unroll(unit, self.timeseries_length)

        if self.return_sequences:
            self.concat = Stack()(*self.units)
        else:
            self.concat = Stack()(self.units[-1])

        super().__init__(self.concat)

    def compute(self, args: np.ndarray):
        self.zero(np.zeros((args.shape[0], self.size)))
        return super().compute(args)

    def _unroll(self, unit, timeseries_length):
        timeframed_input = InputPlaceholder()
        recurrent_unit = Wrapper(unit.copy())(self.zero, Take(0)(timeframed_input))

        units = [recurrent_unit]
        for i in range(1, timeseries_length):
            recurrent_unit2 = Wrapper(unit.copy())(recurrent_unit, Take(i)(timeframed_input))
            units.append(recurrent_unit2)
            recurrent_unit = recurrent_unit2

        return units