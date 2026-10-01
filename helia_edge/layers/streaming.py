"""
# Streaming Layers API

Recurrent cells for streaming models whose state is an explicit model input and output. One model call
processes one step: the caller feeds each ``state_out_k`` output back as the ``state_in_k`` input of the next
call, and feeds zeros at the start of an independent sequence.

Classes:
    StreamingLSTMCell: One LSTM step with explicit hidden and cell state

Functions:
    state_input: Model input for state pair ``k``
    state_output: Model output for state pair ``k``
"""

import keras

from ..export.spec import state_input_name, state_output_name
from ..utils.export import helia_export


@helia_export(path="helia_edge.layers.StreamingLSTMCell")
class StreamingLSTMCell(keras.layers.Layer):
    """One LSTM step: ``[x, h, c] -> [h', c']``.

    The weights have the layout of ``keras.layers.LSTMCell``: ``kernel`` (features, 4 * units),
    ``recurrent_kernel`` (units, 4 * units) and ``bias`` (4 * units), with gates in the order input,
    forget, cell, output. Weights of a ``keras.layers.LSTM`` or ``LSTMCell`` load unchanged.

    c' = f * c + i * tanh(z_c) and h' = o * tanh(c'), where z = x W + h R + b and i, f, o are the sigmoid
    of their slices of z.

    Args:
        units: Size of the hidden and cell state.
        use_bias: Add ``bias`` to the gates.
    """

    def __init__(self, units: int, use_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape):
        x_shape, h_shape, c_shape = input_shape
        units = self.units
        if h_shape[-1] != units or c_shape[-1] != units:
            raise ValueError(f"State shapes {h_shape} and {c_shape} must end in units={units}")
        self.kernel = self.add_weight(name="kernel", shape=(x_shape[-1], 4 * units), initializer="glorot_uniform")
        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel", shape=(units, 4 * units), initializer="orthogonal"
        )
        self.bias = self.add_weight(name="bias", shape=(4 * units,), initializer="zeros") if self.use_bias else None

    def call(self, inputs):
        x, h, c = inputs
        units = self.units
        z = keras.ops.matmul(x, self.kernel) + keras.ops.matmul(h, self.recurrent_kernel)
        if self.bias is not None:
            z = z + self.bias
        i = keras.ops.sigmoid(z[..., :units])
        f = keras.ops.sigmoid(z[..., units : 2 * units])
        g = keras.ops.tanh(z[..., 2 * units : 3 * units])
        o = keras.ops.sigmoid(z[..., 3 * units :])
        c_next = f * c + i * g
        return o * keras.ops.tanh(c_next), c_next

    def compute_output_shape(self, input_shape):
        _, h_shape, c_shape = input_shape
        return tuple(h_shape), tuple(c_shape)

    def get_config(self):
        return {**super().get_config(), "units": self.units, "use_bias": self.use_bias}


def state_input(k: int, shape: tuple[int, ...], batch_size: int | None = None) -> keras.KerasTensor:
    """Model input ``state_in_k`` for state pair ``k``.

    Args:
        k: State pair index; ``state_out_k`` is the matching output.
        shape: State shape without the batch dimension.
        batch_size: Fixed batch size, or None.

    Returns:
        keras.KerasTensor: The input tensor.
    """
    return keras.Input(shape=shape, batch_size=batch_size, name=state_input_name(k))


def state_output(k: int, x: keras.KerasTensor) -> keras.KerasTensor:
    """Name ``x`` as the model output ``state_out_k`` of state pair ``k``.

    Args:
        k: State pair index; ``state_in_k`` is the matching input.
        x: The next state.

    Returns:
        keras.KerasTensor: ``x`` through an identity layer named ``state_out_k``.
    """
    return keras.layers.Identity(name=state_output_name(k))(x)
