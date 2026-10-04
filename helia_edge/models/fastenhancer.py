# Copyright (c) 2025 AHN Sung Hwan
# Licensed under MIT; see licenses/fastenhancer-mit.txt.
"""FastEnhancer folded-inference model for streaming spectral enhancement.

Adapted from aask1357/fastenhancer (revision e74cab1, models/fastenhancer/
default/model.py), the revision of the onnx-vd-v1.0.0 release. One call processes one STFT frame: ``spec_in`` is
(n_fft // 2 + 1, 1, 2) real/imag, and each RNNFormer block carries a GRU state
(freq, channels). Callers zero the states only at independent sequence starts
and feed each ``state_out_k`` back as ``state_in_k``. The graph matches the
released ONNX inference form, with BatchNorm and weight normalization folded.
STFT/iSTFT framing, weight files and weight terms are the caller's concern.
"""

import keras
import numpy as np

from ..layers.streaming import state_input, state_output
from ..utils import helia_export
from .fastenhancer_params import FastEnhancerParams


def linear_filterbanks(n_freq: int, n_filter: int) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed triangular (pre, post) frequency projections as [in, out] kernels.

    Follows upstream revision e74cab1, which produced the onnx-vd-v1.0.0
    weights; upstream changed this formula in 8d2d419 (2026-02-25).
    """
    half = 8000.0  # sr // 2 for the default 16 kHz; the scale cancels.
    f_filter = np.linspace(0, half, n_filter, dtype=np.float32)
    f_freqs = np.linspace(0, half, n_freq, dtype=np.float32)
    delta = half / n_filter
    down = np.pad((f_filter[1:, None] - f_freqs[None, :]) / delta, ((0, 1), (0, 0)), constant_values=1.0)
    up = np.pad((f_freqs[None, :] - f_filter[:-1, None]) / delta, ((1, 0), (0, 0)), constant_values=1.0)
    weight = np.maximum(0.0, np.minimum(down, up))  # [n_filter, n_freq]
    pre = weight / weight.sum(axis=1, keepdims=True)
    post = weight.T / weight.T.sum(axis=1, keepdims=True)
    return pre.T.astype(np.float32), post.T.astype(np.float32)


@helia_export()
class FastEnhancerCompression(keras.layers.Layer):
    """Drop the Nyquist bin and compress magnitude: x * max(|x|, 1e-5)^(c - 1)."""

    def __init__(self, compression: float, **kwargs):
        super().__init__(**kwargs)
        self.compression = compression

    def call(self, spec):
        x = spec[:, :-1, 0, :]
        mag = keras.ops.sqrt(keras.ops.sum(keras.ops.square(x), axis=-1, keepdims=True))
        return x * keras.ops.power(keras.ops.maximum(mag, 1e-5), self.compression - 1.0)

    def get_config(self):
        return {**super().get_config(), "compression": self.compression}


@helia_export()
class FastEnhancerMaskOutput(keras.layers.Layer):
    """Apply a complex mask, undo compression and restore a zero Nyquist bin."""

    def __init__(self, compression: float, **kwargs):
        super().__init__(**kwargs)
        self.compression = compression

    def call(self, inputs):
        x, mask = inputs
        real = x[..., 0] * mask[..., 0] - x[..., 1] * mask[..., 1]
        imag = x[..., 0] * mask[..., 1] + x[..., 1] * mask[..., 0]
        y = keras.ops.stack([real, imag], axis=-1)
        mag = keras.ops.sqrt(keras.ops.sum(keras.ops.square(y), axis=-1, keepdims=True))
        y = y * keras.ops.power(mag, 1.0 / self.compression - 1.0)
        y = keras.ops.pad(y, ((0, 0), (0, 1), (0, 0)))
        return keras.ops.expand_dims(y, 2)

    def get_config(self):
        return {**super().get_config(), "compression": self.compression}


@helia_export()
class FastEnhancerFrequencyProjection(keras.layers.Layer):
    """Project the frequency axis of (batch, freq, channels) with a [in, out] kernel."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel", shape=(input_shape[1], self.units), initializer="zeros")

    def call(self, x):
        return keras.ops.einsum("bfc,fg->bgc", x, self.kernel)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


@helia_export()
class FastEnhancerGRUStep(keras.layers.Layer):
    """One GRU step per frequency band, gates z, r, h and reset after matmul.

    Matches ONNX GRU with ``linear_before_reset=1`` and PyTorch ``nn.GRU``:
    h' = z * h + (1 - z) * tanh(Wx + bw + r * (Rh + br)). Returns h'.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        x_shape, _ = input_shape
        units = self.units
        self.kernel = self.add_weight(name="kernel", shape=(x_shape[-1], 3 * units), initializer="glorot_uniform")
        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel", shape=(units, 3 * units), initializer="orthogonal"
        )
        self.bias = self.add_weight(name="bias", shape=(2, 3 * units), initializer="zeros")

    def call(self, inputs):
        x, h = inputs
        units = self.units
        gx = keras.ops.matmul(x, self.kernel) + self.bias[0]
        gh = keras.ops.matmul(h, self.recurrent_kernel) + self.bias[1]
        z = keras.ops.sigmoid(gx[..., :units] + gh[..., :units])
        r = keras.ops.sigmoid(gx[..., units : 2 * units] + gh[..., units : 2 * units])
        n = keras.ops.tanh(gx[..., 2 * units :] + r * gh[..., 2 * units :])
        return z * h + (1.0 - z) * n

    def get_config(self):
        return {**super().get_config(), "units": self.units}


@helia_export()
class FastEnhancerPositionalEmbedding(keras.layers.Layer):
    """Add a learned (freq, channels) embedding."""

    def build(self, input_shape):
        self.embedding = self.add_weight(name="embedding", shape=tuple(input_shape[1:]), initializer="zeros")

    def call(self, x):
        return x + self.embedding


@helia_export()
class FastEnhancerFrequencyAttention(keras.layers.Layer):
    """Multi-head self-attention across frequency bands, no output projection.

    The qkv kernel is [channels, 3 * channels] with per-head interleaved
    columns: head h uses q = 3dh*h + [0, dh), k = + [dh, 2dh), v = + [2dh, 3dh).
    """

    def __init__(self, num_heads: int, use_bias: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.use_bias = use_bias

    def build(self, input_shape):
        channels = input_shape[-1]
        self.kernel = self.add_weight(name="kernel", shape=(channels, 3 * channels), initializer="glorot_uniform")
        self.bias = self.add_weight(name="bias", shape=(3 * channels,), initializer="zeros") if self.use_bias else None

    def call(self, x):
        freq, channels = x.shape[1], x.shape[2]
        head_dim = channels // self.num_heads
        qkv = keras.ops.matmul(x, self.kernel)
        if self.bias is not None:
            qkv = qkv + self.bias
        qkv = keras.ops.transpose(keras.ops.reshape(qkv, (-1, freq, self.num_heads, 3 * head_dim)), (0, 2, 1, 3))
        q = qkv[..., :head_dim]
        k = qkv[..., head_dim : 2 * head_dim]
        v = qkv[..., 2 * head_dim :]
        scores = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2))) * (head_dim**-0.5)
        out = keras.ops.matmul(keras.ops.softmax(scores, axis=-1), v)
        return keras.ops.reshape(keras.ops.transpose(out, (0, 2, 1, 3)), (-1, freq, channels))

    def get_config(self):
        return {**super().get_config(), "num_heads": self.num_heads, "use_bias": self.use_bias}


def _conv(x, filters: int, kernel_size: int, name: str):
    return keras.layers.Conv1D(filters, kernel_size, padding="same", name=name)(x)


def _act(x, params: FastEnhancerParams, name: str):
    return keras.layers.Activation(params.activation, name=name)(x)


def build(
    params: FastEnhancerParams,
    input_shape: tuple[int | None, ...] | None = None,
    *,
    batch_size: int | None = None,
    name: str | None = None,
) -> keras.Model:
    """Construct an untrained one-frame FastEnhancer with named streaming inputs and outputs.

    Inputs: ``spec_in`` (bins, 1, 2) and ``state_in_k`` (freq, channels) per block. Outputs: ``spec_out``
    and ``state_out_k``. Fixed linear filterbanks are initialized and frozen; other weights are untrained
    until imported with a mapping from ``fastenhancer_params`` (``FASTENHANCER_T_ONNX`` or
    ``fastenhancer_mapping``).

    Args:
        params (FastEnhancerParams): Model parameters.
        input_shape (tuple[int | None, ...] | None): None, or ``spec_in``'s fixed shape ``(bins, 1, 2)``.
        batch_size (int | None): Static batch size; None for a dynamic batch.
        name (str | None): Model name; the family when None.

    Returns:
        keras.Model: The model, named ``fastenhancer`` unless ``name`` is given.
    """
    shape = params.input_shape
    if input_shape is not None and tuple(input_shape) != shape:
        raise ValueError(f"FastEnhancer takes spec_in shape {shape}, not {tuple(input_shape)}")
    rf = params.rnnformer
    channels = params.channels
    k0, stride = params.kernel_size[0], params.stride
    pad = (k0 - stride) // 2

    spec_in = keras.Input(shape=shape, batch_size=batch_size, name="spec_in")
    caches_in = [state_input(i, (rf.freq, rf.channels), batch_size) for i in range(rf.num_blocks)]

    compressed = FastEnhancerCompression(params.input_compression, name="compress")(spec_in)
    x = keras.layers.ZeroPadding1D(pad, name="enc_pre_pad")(compressed)
    x = keras.layers.Conv1D(channels, k0, strides=stride, name="enc_pre_conv")(x)
    x = _act(x, params, "enc_pre_act")
    skips = [x]
    for index, kernel in enumerate(params.kernel_size[1:]):
        y = _act(_conv(x, channels, kernel, f"encoder_{index}_conv"), params, f"encoder_{index}_act")
        # The source adds the residual in place, so the skip includes it.
        x = keras.layers.Add(name=f"encoder_{index}_res")([y, x]) if params.resnet else y
        skips.append(x)

    rf_in = x
    pre, post = linear_filterbanks(params.encoder_bins, rf.freq)
    x = FastEnhancerFrequencyProjection(rf.freq, trainable=False, name="rf_pre_proj")(x)
    x = _conv(x, rf.channels, 1, "rf_pre_conv")
    caches_out = []
    for index, cache in enumerate(caches_in):
        block = f"rf_block_{index}"
        h = FastEnhancerGRUStep(rf.channels, name=f"{block}_gru")([x, cache])
        caches_out.append(state_output(index, h))
        y = keras.layers.Dense(rf.channels, name=f"{block}_rnn_fc")(h)
        if rf.post_act:
            y = _act(y, params, f"{block}_rnn_act")
        x = keras.layers.Add(name=f"{block}_rnn_res")([y, x])
        if rf.positional_embedding and index == 0:
            x = FastEnhancerPositionalEmbedding(name=f"{block}_pe")(x)
        y = FastEnhancerFrequencyAttention(rf.num_heads, rf.attn_bias, name=f"{block}_attn")(x)
        y = keras.layers.Dense(rf.channels, name=f"{block}_attn_fc")(y)
        if rf.post_act:
            y = _act(y, params, f"{block}_attn_act")
        x = keras.layers.Add(name=f"{block}_attn_res")([y, x])
    x = FastEnhancerFrequencyProjection(params.encoder_bins, trainable=False, name="rf_post_proj")(x)
    x = _conv(x, channels, 1, "rf_post_conv")
    if params.resnet:
        x = keras.layers.Add(name="rf_res")([x, rf_in])

    for index, kernel in enumerate(reversed(params.kernel_size[1:])):
        x_in = x
        x = keras.layers.Concatenate(name=f"decoder_{index}_cat")([x, skips.pop()])
        x = _act(_conv(x, channels, 1, f"decoder_{index}_conv0"), params, f"decoder_{index}_act0")
        x = _act(_conv(x, channels, kernel, f"decoder_{index}_conv1"), params, f"decoder_{index}_act1")
        if params.resnet:
            x = keras.layers.Add(name=f"decoder_{index}_res")([x, x_in])
    x = keras.layers.Concatenate(name="dec_post_cat")([x, skips.pop()])
    x = _act(_conv(x, channels, 1, "dec_post_conv"), params, "dec_post_act")
    x = keras.layers.Conv1DTranspose(2, k0, strides=stride, name="dec_post_upsample")(x)
    x = keras.layers.Cropping1D(pad, name="dec_post_crop")(x)
    if params.mask != "none":
        x = keras.layers.Activation(params.mask, name="mask_act")(x)
    spec_out = FastEnhancerMaskOutput(params.input_compression, name="spec_out")([compressed, x])

    model = keras.Model(inputs=[spec_in, *caches_in], outputs=[spec_out, *caches_out], name=name or params.family)
    model.get_layer("rf_pre_proj").kernel.assign(pre)
    model.get_layer("rf_post_proj").kernel.assign(post)
    return model
