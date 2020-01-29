import json

import numpy as np

from neural.initializers import Ones, Normal
from neural.layers import WeightedSum
from neural.optimizers import Adam
# Load RANDOM data
from neural.units import InputPlaceholder


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

matmul = WeightedSum(1, Ones(), Normal())(InputPlaceholder()(X))
a = json.dumps(matmul.structure())
print(a)
# print(matmul.evaluate())
# y = matmul.structure()
#
# new_matmul = Unit.create(y)
# placeholder = obtain_placeholders(new_matmul)[0]
# placeholder(X)
# print(new_matmul.structure())
# print(new_matmul.evaluate())

# a = Config([1])
# b = Config([2, a])
# print(b.self_structure())