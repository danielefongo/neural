import numpy as np

import arrays
from initializers import Initializer, Random


class Weights(np.ndarray):
    def __new__(subtype, shape, weights_initializer: Initializer = Random(), biases_initializer: Initializer = Random(),
                dtype=float, offset=0,
                strides=None, order=None, info=None):
        weights = weights_initializer.generate(shape)
        biases = biases_initializer.generate(Weights.bias_shape(shape))

        data = arrays.add_column(weights, axis=0, values=biases)

        return super(Weights, subtype).__new__(subtype, data.shape, dtype,
                                               data, offset, strides,
                                               order)

    @staticmethod
    def bias_shape(shape):
        bias_shape = list(shape)
        bias_shape[0] = 1
        return tuple(bias_shape)

    def _get_bias(self):
        return self[-1]

    def _get_weights(self):
        return self[:-1]

    b = property(fget=_get_bias)
    w = property(fget=_get_weights)
