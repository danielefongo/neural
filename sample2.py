import json

import numpy as np

from neural.activations import Linear
from neural.configurables import Config
from neural.initializers import Ones, Normal, Zeros
from neural.layers import WeightedSum, Layer
from neural.optimizers import Adam
# Load RANDOM data
from neural.units import InputPlaceholder, Unit, MatMul


def obtain_placeholders(unit):
    candidates = []
    for candidate in unit.plain_graph():
        if candidate not in candidates and isinstance(candidate, InputPlaceholder):
            candidates.append(candidate)
    return candidates

X = np.ones((2, 2))

# Train
epochs = 20
batch_size = 8
learning_rate = 0.001
optimizer = Adam(learning_rate)

# print(matmul.evaluate())
# y = matmul.structure()
#
# new_matmul = Unit.create(y)
# placeholder = obtain_placeholders(new_matmul)[0]
# placeholder(X)
# print(new_matmul.structure())
# print(new_matmul.evaluate())

a = InputPlaceholder()
b = Layer(1, Linear(), Ones(), Zeros())(a)

d = Unit.create(b.structure())

#print(b.self_structure())
print(b.structure())
print(d.structure())
placeholder_b = obtain_placeholders(b)[0]
placeholder_d = obtain_placeholders(d)[0]
placeholder_b(X)
placeholder_d(X)
print(b.evaluate())
print(d.evaluate())
#print(d.self_structure())
#print(Unit.create(b.structure()))