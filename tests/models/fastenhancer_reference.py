"""NumPy reference for one FastEnhancer frame, following the released ONNX graph.

Operates on ONNX/PyTorch-layout tensors (NCW convolutions, ONNX GRU z/r/h
gates with linear_before_reset=1, strided convolution by pad/reshape/transpose)
so it shares no layout conversion with the Keras hydration under test.
"""

import numpy as np


def _conv1d(x, weight, bias, pad):
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    windows = np.lib.stride_tricks.sliding_window_view(x, weight.shape[2], axis=2)  # [B, I, T, K]
    return np.einsum("bitk,oik->bot", windows, weight) + bias[None, :, None]


def _conv_transpose1d(x, weight, bias, stride, pad):
    batch, _, length = x.shape
    kernel = weight.shape[2]
    out = np.zeros((batch, weight.shape[1], (length - 1) * stride + kernel), dtype=np.float64)
    for t in range(length):
        out[:, :, t * stride : t * stride + kernel] += np.einsum("bi,iok->bok", x[:, :, t], weight)
    return out[:, :, pad : out.shape[2] - pad] + bias[None, :, None]


def _act(x, name):
    return x / (1.0 + np.exp(-x)) if name == "silu" else np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gru(x, h, w, r, b):
    units = h.shape[-1]
    gx = x @ w[0].T + b[0, : 3 * units]
    gh = h @ r[0].T + b[0, 3 * units :]
    z = _sigmoid(gx[..., :units] + gh[..., :units])
    reset = _sigmoid(gx[..., units : 2 * units] + gh[..., units : 2 * units])
    n = np.tanh(gx[..., 2 * units :] + reset * gh[..., 2 * units :])
    return (1.0 - z) * n + z * h


def _attention(x, kernel, bias, heads):
    batch, freq, channels = x.shape
    head_dim = channels // heads
    qkv = x @ kernel + (0.0 if bias is None else bias)
    qkv = qkv.reshape(batch, freq, heads, 3 * head_dim).transpose(0, 2, 1, 3)
    q, k, v = qkv[..., :head_dim], qkv[..., head_dim : 2 * head_dim], qkv[..., 2 * head_dim :]
    scores = q @ k.transpose(0, 1, 3, 2) * head_dim**-0.5
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    out = (scores / scores.sum(axis=-1, keepdims=True)) @ v
    return out.transpose(0, 2, 1, 3).reshape(batch, freq, channels)


def reference_frame(params, t, spec, caches):
    """Return (spec_out [B, bins, 1, 2], new caches) for one frame in float64."""
    rf, c = params.rnnformer, params.input_compression
    t = {k: np.asarray(v, dtype=np.float64) for k, v in t.items()}
    spec = np.asarray(spec, dtype=np.float64)
    batch = spec.shape[0]
    x = spec[:, :-1]
    mag = np.maximum(np.sqrt((x**2).sum(-1, keepdims=True)), 1e-5)
    compressed = x * mag ** (c - 1.0)  # [B, F, 1, 2]
    bins = compressed.shape[1]

    s, k0 = params.stride, params.kernel_size[0]
    pad = (k0 - s) // 2
    y = compressed.transpose(0, 2, 3, 1).reshape(batch, 2, bins)
    y = np.pad(y, ((0, 0), (0, 0), (pad, pad)))
    y = y.reshape(batch, 2, -1, s).transpose(0, 3, 1, 2).reshape(batch, 2 * s, -1)
    y = _act(_conv1d(y, t["enc_pre.0.weight"], t["enc_pre.0.bias"], 0), params.activation)
    skips = [y]
    for i, k in enumerate(params.kernel_size[1:]):
        z = _act(_conv1d(y, t[f"encoder.{i}.0.weight"], t[f"encoder.{i}.0.bias"], (k - 1) // 2), params.activation)
        y = z + y if params.resnet else z
        skips.append(y)

    rf_in = y
    y = y @ t["rf_pre.0.kernel"]
    y = _conv1d(y, t["rf_pre.1.weight"], t["rf_pre.1.bias"], 0).transpose(0, 2, 1)  # [B, F', C]
    new_caches = []
    for i, cache in enumerate(caches):
        b = f"rf_block.{i}"
        h = _gru(y, np.asarray(cache, dtype=np.float64), t[f"{b}.rnn.W"], t[f"{b}.rnn.R"], t[f"{b}.rnn.B"])
        new_caches.append(h)
        z = h @ t[f"{b}.rnn_fc.kernel"] + t[f"{b}.rnn_fc.bias"]
        y = (_act(z, params.activation) if rf.post_act else z) + y
        if rf.positional_embedding and i == 0:
            y = y + t["rf_block.0.pe"]
        z = _attention(y, t[f"{b}.attn.qkv.kernel"], t.get(f"{b}.attn.qkv.bias"), rf.num_heads)
        z = z @ t[f"{b}.attn_fc.kernel"] + t[f"{b}.attn_fc.bias"]
        y = (_act(z, params.activation) if rf.post_act else z) + y
    y = y.transpose(0, 2, 1) @ t["rf_post.0.kernel"]
    y = _conv1d(y, t["rf_post.1.weight"], t["rf_post.1.bias"], 0)
    if params.resnet:
        y = y + rf_in

    for i, k in enumerate(reversed(params.kernel_size[1:])):
        y_in = y
        y = np.concatenate([y, skips.pop()], axis=1)
        y = _act(_conv1d(y, t[f"decoder.{i}.0.weight"], t[f"decoder.{i}.0.bias"], 0), params.activation)
        y = _act(_conv1d(y, t[f"decoder.{i}.2.weight"], t[f"decoder.{i}.2.bias"], (k - 1) // 2), params.activation)
        if params.resnet:
            y = y + y_in
    y = np.concatenate([y, skips.pop()], axis=1)
    y = _act(_conv1d(y, t["dec_post.0.weight"], t["dec_post.0.bias"], 0), params.activation)
    y = _conv_transpose1d(y, t["dec_post.2.weight"], t["dec_post.2.bias"], s, pad)  # [B, 2, F]
    mask = y.reshape(batch, 1, 2, bins).transpose(0, 3, 1, 2)
    if params.mask == "sigmoid":
        mask = _sigmoid(mask)
    elif params.mask == "tanh":
        mask = np.tanh(mask)

    xr, xi, mr, mi = compressed[..., 0], compressed[..., 1], mask[..., 0], mask[..., 1]
    out = np.stack([xr * mr - xi * mi, xr * mi + xi * mr], axis=-1)
    out = out * np.sqrt((out**2).sum(-1, keepdims=True)) ** (1.0 / c - 1.0)
    return np.pad(out, ((0, 0), (0, 1), (0, 0), (0, 0))), new_caches
