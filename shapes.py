def change(shape: tuple, axis=-1, value: int = 1):
    tmp_shape = list(shape)
    tmp_shape[axis] = value
    return tuple(tmp_shape)

