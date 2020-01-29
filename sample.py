from typing import List, Set

import numpy as np

from neural.activations import Softmax, Tanh, Linear
from neural.initializers import Ones, Zeros
from neural.layers import Layer, SimpleRNN
from neural.losses import CrossEntropy
from neural.networks import Network
from neural.optimizers import Adam

# Load RANDOM data
from neural.units import InputPlaceholder, Wrapper, Variable, Weight, Unit, Recurrent

X = np.ones((2, 2, 2))

# Train
epochs = 1
batch_size = 8
learning_rate = 0.001
optimizer = Adam(learning_rate)

input_features = X.shape[-1]



def obtain_placeholders(unit):
    candidates = []
    for candidate in unit.plain_graph():
        if candidate not in candidates and isinstance(candidate, InputPlaceholder):
            candidates.append(candidate)
    return candidates

def getWeight(unit):
    candidates = []
    for candidate in unit.plain_graph():
        if isinstance(candidate, Recurrent):
            candidates.extend(getWeight(candidate.units[-1]))
        elif isinstance(candidate, Wrapper):
            candidates.extend(getWeight(candidate.unit))
        if candidate not in candidates and isinstance(candidate, Weight):
            candidates.append(candidate)
    def f7(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]
    return f7(candidates)


def setVariables(unit, vars):
    lis: Set[Weight] = getWeight(unit)
    for l, v in zip(lis, vars):
        l.set(v.weights.value)


unitina = SimpleRNN(1,2)(InputPlaceholder()(X)) #Layer(1, Linear(), Ones(), Zeros())(InputPlaceholder()(X))
# setVariables(network.unit, a)
print(unitina.evaluate())
print(unitina.structure())
unitina2 = Unit.create(unitina.structure())
setVariables(unitina2, getWeight(unitina))

obtain_placeholders(unitina2)[0](X)
print(unitina.evaluate()[:3])
print(unitina2.evaluate()[:3])
