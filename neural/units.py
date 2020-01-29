import copy
import re
from pydoc import locate
from typing import List

import numpy as np

from neural.configurables import Config
from neural.ops import multiply, add, dot, sum_to_shape, merge, unmerge, stack, unstack, take, replace, reshape, zeros, \
    empty


class Unit(Config):
    def __init__(self, init: list = []):
        self.input_units = []
        self.output_units = []
        self.inputs = []
        self.output = []
        self.gradient = None
        self.plain = []
        super().__init__(*init)

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

    def self_structure(self):
        a = super().self_structure()
        a["input_units"] = [hash(a) for a in self.input_units if isinstance(a, Unit)]
        return a

    def structure(self):
        return [unit.self_structure() for unit in self.plain_graph()]

    @staticmethod
    def create(configs):
        hash = {}
        for conf in configs:
            unittype: Config = locate(conf["clazz"])
            input_units = [hash[id] for id in conf["input_units"]]
            hashino = conf["hash"]
            if hashino not in hash.keys():
                unit_1 = unittype.self_create(conf)
                if len(input_units):
                    unit_1(*input_units)

                hash[hashino] = unit_1

        return hash[hashino]

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
        gradient = None
        for after in self.output_units:
            if len(after.input_units) == 1:
                next_gradient = after.gradient
            else:
                index = after.input_units.index(self)
                next_gradient = after.gradient[index]

            if gradient is None:
                gradient = next_gradient
            else:
                gradient += next_gradient

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
        self.real_data = empty()

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
    def __init__(self, var=None):
        super().__init__()
        if var is None:
            var = Variable()
        self.weights = var

    def is_empty(self):
        return self.weights.is_empty()

    def set(self, data):
        self.weights.value = data

    def compute(self):
        return self.weights.value

    def apply(self, gradient: np.ndarray, optimizer):
        self.weights.value -= optimizer.on(self, gradient)

    def __deepcopy__(self, memo):
        return self


class Add(Unit):
    def __call__(self, a, b):
        return super().__call__(a, b)

    def compute(self, a_val: np.ndarray, b_val: np.ndarray):
        return add(a_val, b_val)

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
        return multiply(a_val, b_val)

    def apply(self, gradient: np.ndarray, optimizer):
        a_val = self.inputs[0]
        b_val = self.inputs[1]

        return [multiply(gradient, a_val), multiply(gradient, b_val)]


class MatMul(Unit):
    def __call__(self, a, b):
        return super().__call__(a, b)

    def compute(self, a_val: np.ndarray, b_val: np.ndarray):
        return dot(a_val, b_val)

    def apply(self, gradient: np.ndarray, optimizer):
        a_val = self.inputs[0]
        b_val = self.inputs[1]

        a_gradient = dot(gradient, b_val.T)
        b_gradient = dot(a_val.T, gradient)

        return [a_gradient, b_gradient]


class Merge(Unit):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def compute(self, *args):
        self.number = len(args)
        if self.number > 1:
            result, self.splits = merge(self.axis, *args)
            return result
        return args[0]

    def apply(self, gradient: np.ndarray, optimizer):
        if self.number > 1:
            return unmerge(self.axis, self.splits, gradient)
        return gradient


class Stack(Unit):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def compute(self, *args):
        self.number = len(args)
        if self.number > 1:
            return stack(self.axis, *args)
        return args[0]

    def apply(self, gradient: np.ndarray, optimizer):
        if self.number > 1:
            return unstack(self.axis, gradient)
        return gradient


class Take(Unit):
    def __init__(self, index, axis=1):
        super().__init__()
        self.index = index
        self.axis = axis

    def compute(self, args: np.ndarray):
        self.shape = args.shape
        return take(self.axis, self.index, args)

    def apply(self, gradient: np.ndarray, optimizer):
        return replace(self.axis, self.index, gradient, zeros(self.shape))


class Flatten(Unit):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def compute(self, args: np.ndarray):
        self.shape = args.shape
        return reshape(args, args.shape[:self.axis] + (-1,))

    def apply(self, gradient: np.ndarray, optimizer):
        return reshape(gradient, self.shape)


class Wrapper(Unit):
    def __init__(self, unit, init = []):
        self.fake_output: Unit = Unit()
        self.fake_inputs = self.obtain_placeholders(unit)

        self.unit: Unit = unit
        self.fake_output(self.unit)

        super().__init__(init)

    def obtain_placeholders(self, unit):
        candidates = []
        for candidate in unit.plain_graph():
            if candidate not in candidates and isinstance(candidate, InputPlaceholder):
                candidates.append(candidate)
        return candidates

    def internalities(self):
        return self.unit.structure()

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
    def __init__(self, unit, size, timeseries_length, return_sequences=False, init=None):
        self.size = size
        self.timeseries_length = timeseries_length
        self.return_sequences = return_sequences
        self.zero = Placeholder()

        self.units = self._unroll(unit, self.timeseries_length)

        if self.return_sequences:
            self.concat = Stack()(*self.units)
        else:
            self.concat = Stack()(self.units[-1])

        super().__init__(self.concat, init)

    def compute(self, args: np.ndarray):
        self.zero(zeros((args.shape[0], self.size)))
        return super().compute(args)

    def _unroll(self, unit, timeseries_length):
        timeframed_input = InputPlaceholder()
        recurrent_unit = Wrapper(unit.copy())(self.zero, Take(0)(timeframed_input))

        units = [recurrent_unit]
        for i in range(1, timeseries_length):
            recurrent_unit = Wrapper(unit.copy())(recurrent_unit, Take(i)(timeframed_input))
            units.append(recurrent_unit)

        print([i.unit.weighted_sum.biases.weights for i in units])
        return units
