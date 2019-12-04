from typing import Union

import numpy as np


def add_column(array: np.ndarray, axis: int, values: Union[tuple, int]):
    return np.insert(array, array.shape[axis], values=values, axis=axis)
