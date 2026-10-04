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

from collections.abc import Mapping

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
    until :func:`load_fastenhancer_weights` hydrates matching tensors.

    Args:
        params (FastEnhancerParams): Model parameters.
        input_shape (tuple[int | None, ...] | None): None, or ``spec_in``'s fixed shape ``(bins, 1, 2)``.
        batch_size (int | None): Static batch size; None for a dynamic batch.
        name (str | None): Model name; the family when None.

    Returns:
        keras.Model: The model, named ``fastenhancer`` unless ``name`` is given.
    """
    shape = (params.spectral_bins, 1, 2)
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


def fastenhancer_weight_shapes(params: FastEnhancerParams) -> dict[str, tuple[int, ...]]:
    """Folded tensors in ONNX/PyTorch layout, keyed by module path.

    Conv weights are [out, in, kernel]; ``*.kernel`` MatMul weights are
    [in, out]; ``rnn.W``/``rnn.R`` are [1, 3H, in] and ``rnn.B`` is [1, 6H]
    with ONNX gate order z, r, h; the upsampler is [in, out, kernel].
    """
    rf = params.rnnformer
    c, rc, k0, s = params.channels, rf.channels, params.kernel_size[0], params.stride
    shapes: dict[str, tuple[int, ...]] = {
        "enc_pre.0.weight": (c, 2 * s, k0 // s),
        "enc_pre.0.bias": (c,),
        "rf_pre.0.kernel": (params.encoder_bins, rf.freq),
        "rf_pre.1.weight": (rc, c, 1),
        "rf_pre.1.bias": (rc,),
        "rf_post.0.kernel": (rf.freq, params.encoder_bins),
        "rf_post.1.weight": (c, rc, 1),
        "rf_post.1.bias": (c,),
        "dec_post.0.weight": (c, 2 * c, 1),
        "dec_post.0.bias": (c,),
        "dec_post.2.weight": (c, 2, k0),
        "dec_post.2.bias": (2,),
    }
    for i, k in enumerate(params.kernel_size[1:]):
        shapes[f"encoder.{i}.0.weight"] = (c, c, k)
        shapes[f"encoder.{i}.0.bias"] = (c,)
    for i, k in enumerate(reversed(params.kernel_size[1:])):
        shapes[f"decoder.{i}.0.weight"] = (c, 2 * c, 1)
        shapes[f"decoder.{i}.0.bias"] = (c,)
        shapes[f"decoder.{i}.2.weight"] = (c, c, k)
        shapes[f"decoder.{i}.2.bias"] = (c,)
    for i in range(rf.num_blocks):
        b = f"rf_block.{i}"
        shapes[f"{b}.rnn.W"] = (1, 3 * rc, rc)
        shapes[f"{b}.rnn.R"] = (1, 3 * rc, rc)
        shapes[f"{b}.rnn.B"] = (1, 6 * rc)
        shapes[f"{b}.rnn_fc.kernel"] = (rc, rc)
        shapes[f"{b}.rnn_fc.bias"] = (rc,)
        shapes[f"{b}.attn.qkv.kernel"] = (rc, 3 * rc)
        if rf.attn_bias:
            shapes[f"{b}.attn.qkv.bias"] = (3 * rc,)
        shapes[f"{b}.attn_fc.kernel"] = (rc, rc)
        shapes[f"{b}.attn_fc.bias"] = (rc,)
    if rf.positional_embedding:
        shapes["rf_block.0.pe"] = (rf.freq, rc)
    return shapes


# Anonymous initializer names in the onnx-vd-v1.0.0 fastenhancer_t.spec.onnx release.
FASTENHANCER_T_ONNX_NAMES: Mapping[str, str] = {
    "onnx::MatMul_638": "rf_pre.0.kernel",
    "onnx::GRU_662": "rf_block.0.rnn.W",
    "onnx::GRU_663": "rf_block.0.rnn.R",
    "onnx::GRU_664": "rf_block.0.rnn.B",
    "onnx::MatMul_675": "rf_block.0.rnn_fc.kernel",
    "onnx::MatMul_680": "rf_block.0.attn.qkv.kernel",
    "onnx::MatMul_702": "rf_block.0.attn_fc.kernel",
    "onnx::GRU_724": "rf_block.1.rnn.W",
    "onnx::GRU_725": "rf_block.1.rnn.R",
    "onnx::GRU_726": "rf_block.1.rnn.B",
    "onnx::MatMul_737": "rf_block.1.rnn_fc.kernel",
    "onnx::MatMul_742": "rf_block.1.attn.qkv.kernel",
    "onnx::MatMul_764": "rf_block.1.attn_fc.kernel",
    "onnx::MatMul_766": "rf_post.0.kernel",
}


def fastenhancer_t_onnx_weights(initializers: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rename released FastEnhancer-T FP32 initializers to module-path keys.

    Graph constants (names starting with "/") are dropped; every other name
    must be known, so a different export is rejected instead of guessed.
    """
    known = set(fastenhancer_weight_shapes(FastEnhancerParams()))
    renamed = {}
    for name, value in initializers.items():
        if name.startswith("/"):
            continue
        key = FASTENHANCER_T_ONNX_NAMES.get(name, name)
        if key not in known:
            raise ValueError(f"unrecognized FastEnhancer-T initializer {name!r}")
        renamed[key] = np.asarray(value)
    return renamed


def load_fastenhancer_weights(
    model: keras.Model, params: FastEnhancerParams, tensors: Mapping[str, np.ndarray]
) -> None:
    """Hydrate a model from folded tensors keyed as in :func:`fastenhancer_weight_shapes`.

    All names and shapes must match exactly; nothing is loaded otherwise.
    """
    expected = fastenhancer_weight_shapes(params)
    missing, extra = sorted(set(expected) - set(tensors)), sorted(set(tensors) - set(expected))
    if missing or extra:
        raise ValueError(f"FastEnhancer tensors mismatch: missing {missing}, unexpected {extra}")
    arrays = {}
    for name, shape in expected.items():
        array = np.asarray(tensors[name])
        if array.shape != shape or not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{name}: expected float {shape}, received {array.dtype} {array.shape}")
        arrays[name] = array.astype(np.float32)

    def conv(w):
        return w.transpose(2, 1, 0)

    s, rf = params.stride, params.rnnformer
    enc = arrays["enc_pre.0.weight"]  # [out, phase * 2 + ri, j] -> [out, ri, j * stride + phase]
    enc = enc.reshape(enc.shape[0], s, 2, -1).transpose(0, 2, 3, 1).reshape(enc.shape[0], 2, -1)
    assignments = {
        "enc_pre_conv": [conv(enc), arrays["enc_pre.0.bias"]],
        "rf_pre_proj": [arrays["rf_pre.0.kernel"]],
        "rf_pre_conv": [conv(arrays["rf_pre.1.weight"]), arrays["rf_pre.1.bias"]],
        "rf_post_proj": [arrays["rf_post.0.kernel"]],
        "rf_post_conv": [conv(arrays["rf_post.1.weight"]), arrays["rf_post.1.bias"]],
        "dec_post_conv": [conv(arrays["dec_post.0.weight"]), arrays["dec_post.0.bias"]],
        "dec_post_upsample": [conv(arrays["dec_post.2.weight"]), arrays["dec_post.2.bias"]],
    }
    for i in range(len(params.kernel_size) - 1):
        assignments[f"encoder_{i}_conv"] = [conv(arrays[f"encoder.{i}.0.weight"]), arrays[f"encoder.{i}.0.bias"]]
        assignments[f"decoder_{i}_conv0"] = [conv(arrays[f"decoder.{i}.0.weight"]), arrays[f"decoder.{i}.0.bias"]]
        assignments[f"decoder_{i}_conv1"] = [conv(arrays[f"decoder.{i}.2.weight"]), arrays[f"decoder.{i}.2.bias"]]
    for i in range(rf.num_blocks):
        b, layer = f"rf_block.{i}", f"rf_block_{i}"
        assignments[f"{layer}_gru"] = [
            arrays[f"{b}.rnn.W"][0].T,
            arrays[f"{b}.rnn.R"][0].T,
            arrays[f"{b}.rnn.B"].reshape(2, -1),
        ]
        assignments[f"{layer}_rnn_fc"] = [arrays[f"{b}.rnn_fc.kernel"], arrays[f"{b}.rnn_fc.bias"]]
        qkv = [arrays[f"{b}.attn.qkv.kernel"]]
        if rf.attn_bias:
            qkv.append(arrays[f"{b}.attn.qkv.bias"])
        assignments[f"{layer}_attn"] = qkv
        assignments[f"{layer}_attn_fc"] = [arrays[f"{b}.attn_fc.kernel"], arrays[f"{b}.attn_fc.bias"]]
    if rf.positional_embedding:
        assignments["rf_block_0_pe"] = [arrays["rf_block.0.pe"]]

    layers = {layer.name: layer for layer in model.layers}
    for name, values in assignments.items():
        if name not in layers or [w.shape for w in layers[name].get_weights()] != [v.shape for v in values]:
            raise ValueError(f"model layer {name} does not match params; build the model from the same params")
    for name, values in assignments.items():
        layers[name].set_weights(values)
