import copy
from typing import List

import numpy as np

from neural.exportable import Exportable
from neural.ops import multiply, add, dot, sum_to_shape, merge, unmerge, stack, unstack, take, replace, reshape, zeros, \
    empty


class Unit(Exportable):
    def __init__(self):
        self.input_units = []
        self.output_units = []
        self.inputs = []
        self.output = []
        self.gradient = None
        self.plain = []
        super().__init__()

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

        return self

    def _remove_all_inputs(self):
        for element in copy.copy(self.input_units):
            element.output_units.remove(self)
            self.input_units.remove(element)

    def copy(self):
        return copy.deepcopy(self)

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

    def vars(self):
        return []


class Graph:
    def __init__(self, unit):
        self.unit = unit
        self.inputs = self.find(Input)

    def evaluate(self, *args):
        for index in range(len(args)):
            self.inputs[index](args[index])

        [node._forward() for node in self.plain()]
        return self.unit.output

    def error(self, optimizer):
        [node._backward(optimizer) for node in self.plain()[::-1]]

        gradients = [unit.gradient for unit in self.inputs]
        return gradients if len(gradients) > 1 else gradients[0]

    def plain(self):
        node_list: List[Unit] = []

        def recurse(node):
            if isinstance(node, Unit):
                for input_node in node.input_units:
                    recurse(input_node)
            if node not in node_list:
                node_list.append(node)

        recurse(self.unit)
        return node_list

    def get_vars(self):
        all_vars = []
        for unit in self.plain():
            all_vars.extend(unit.vars())
        return all_vars

    def set_vars(self, variables=[]):
        if len(variables):
            for new_variable, old_variable in zip(self.get_vars(), variables):
                new_variable.value = old_variable.value

    def export(self):
        exports = []
        for unit in self.plain():
            unit_export = unit.export()
            unit_export["input_units"] = [hash(input_unit) for input_unit in unit.input_units]
            exports.append(unit_export)
        return exports

    @staticmethod
    def use(configs):
        created_units = {}
        for config in configs:
            input_units = [created_units[id] for id in config["input_units"]]
            actual_unit_hash = config["hash"]
            if actual_unit_hash not in created_units.keys():
                new_unit = Unit.use(config)
                if len(input_units):
                    new_unit(*input_units)

                created_units[actual_unit_hash] = new_unit

        return Graph(created_units[actual_unit_hash])

    def find(self, unitType):
        candidates = []
        for candidate in self.plain():
            if candidate not in candidates and isinstance(candidate, unitType):
                candidates.append(candidate)
        return candidates


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


class Input(Placeholder):
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

    def vars(self):
        return [self.weights]


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
    def __init__(self, unit):
        self.inner_graph = Graph(unit)
        self.fake_output: Unit = Unit()(unit)

        super().__init__()

    def compute(self, *args: np.ndarray):
        return self.inner_graph.evaluate(*args)

    def apply(self, gradient: np.ndarray, optimizer):
        self.fake_output.gradient = gradient

        return self.inner_graph.error(optimizer)

    def vars(self):
        return self.inner_graph.get_vars()


class Recurrent(Wrapper):
    def __init__(self, unit, size, timeseries_length, return_sequences=False, truncated_backprop_size=10):
        self.size = size
        self.truncated_backprop_size = truncated_backprop_size
        self.timeseries_length = timeseries_length
        self.return_sequences = return_sequences
        self.zero = Placeholder()
        self.single_unit_graph = Graph(unit)

        self.units = self._unroll(unit, self.timeseries_length)

        if self.return_sequences:
            self.concat = Stack()(*self.units)
        else:
            self.concat = Stack()(self.units[-1])

        super().__init__(self.concat)

    def compute(self, args: np.ndarray):
        self.zero(zeros((args.shape[0], self.size)))
        return super().compute(args)

    def apply(self, gradient: np.ndarray, optimizer):
        if len(self.units) < self.truncated_backprop_size:
            return super().apply(gradient, optimizer)

        truncated_unit_index = len(self.units) - self.truncated_backprop_size
        truncated_unit_inputs = [unit for unit in self.units[truncated_unit_index].input_units]

        self.units[truncated_unit_index]()
        new_gradient = super().apply(gradient, optimizer)
        self.units[truncated_unit_index](*truncated_unit_inputs)

        return new_gradient

    def _unroll(self, unit, timeseries_length):
        timeframed_input = Input()
        recurrent_unit = Wrapper(unit.copy())(self.zero, Take(0)(timeframed_input))

        units = [recurrent_unit]
        for i in range(1, timeseries_length):
            recurrent_unit = Wrapper(unit.copy())(recurrent_unit, Take(i)(timeframed_input))
            units.append(recurrent_unit)

        return units

    def vars(self):
        return self.single_unit_graph.get_vars()
