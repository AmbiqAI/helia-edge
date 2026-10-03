# Copyright (c) 2020-present Silero Team
# Licensed under MIT; see licenses/silero-vad-mit.txt.
"""The Silero VAD v6 NPU lowering: the same model rebuilt from operators an Ethos-U NPU runs.

``lower_silero_vad_v6_npu`` takes a ``silero_vad_v6`` model and returns a model with the same inputs,
outputs and state pairs, built from its weights:

- The 576 samples become 9 blocks of 64. Frames 0 to 2 are a convolution over blocks (kernel 4,
  stride 2); a 128-sample stride is beyond the NPU.
- Frame 3 overlaps the right reflect padding. Reflection is linear, so it is folded into a second
  convolution's weights over blocks 6 to 8.
- The magnitude of each bin is the largest of 9 projections of (|re|, |im|) onto directions from 0 to
  90 degrees, with a gain that centres the error within +-0.24%. Square root and power spectra do
  not survive 16-bit integers.
- The third and fourth encoder convolutions see real frames through some taps only, so they are fully
  connected layers over those taps.

Everything except the magnitude is exact in float.
"""

import keras
import numpy as np

from ..layers.streaming import StreamingLSTMCell, state_output
from ..utils.export import helia_export

BLOCK = 64
BINS = 129
DIRECTIONS = 9


def _projection_kernel(directions: int = DIRECTIONS) -> np.ndarray:
    """(1, 1, 2, directions) unit directions over 0 to 90 degrees, scaled to centre the error."""
    theta = np.linspace(0, np.pi / 2, directions)
    half_gap = np.pi / 4 / (directions - 1)
    gain = 2 / (1 + np.cos(half_gap))
    return (gain * np.stack([np.cos(theta), np.sin(theta)]))[None, None].astype(np.float32)


def _block_kernels(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convolution kernels over 64-sample blocks from the semantic STFT basis (256, 1, 258).

    Returns kernels for frames 0 to 2, (4, 1, 64, 258), and for frame 3 with the reflection folded in,
    (3, 1, 64, 258). Output channels are interleaved: 2b is the real part of bin b, 2b + 1 the imaginary.
    """
    taps = basis[:, 0, :]  # (256, 258): real parts of every bin, then imaginary parts
    taps = taps[:, np.stack([np.arange(BINS), np.arange(BINS) + BINS], axis=1).reshape(-1)]
    main = taps.reshape(4, BLOCK, 2 * BINS)[:, None]
    # Frame 3 reads padded samples 384..639; samples 576..639 reflect x[574], x[573], ..., x[511]
    folded = taps[:192].copy()
    for j in range(64):
        folded[574 - j - 384] += taps[192 + j]
    return main.astype(np.float32), folded.reshape(3, BLOCK, 2 * BINS)[:, None].astype(np.float32)


@helia_export(path="helia_edge.models.SileroNpuFrontend")
class SileroNpuFrontend(keras.layers.Layer):
    """STFT magnitude of one Silero call from block convolutions and projections: (1, 576) -> (1, 4, 1, 129)."""

    def build(self, input_shape):
        self.main = self.add_weight(name="main", shape=(4, 1, BLOCK, 2 * BINS), initializer="zeros", trainable=False)
        self.folded = self.add_weight(
            name="folded", shape=(3, 1, BLOCK, 2 * BINS), initializer="zeros", trainable=False
        )
        self.projection = self.add_weight(
            name="projection", shape=(1, 1, 2, DIRECTIONS), initializer="zeros", trainable=False
        )

    def call(self, audio):
        blocks = keras.ops.reshape(audio, (-1, 9, 1, BLOCK))
        first = keras.ops.conv(blocks, self.main, strides=(2, 1), padding="valid")  # (1, 3, 1, 258)
        last = keras.ops.conv(blocks[:, 6:9], self.folded, padding="valid")  # (1, 1, 1, 258)
        pairs = keras.ops.reshape(keras.ops.concatenate([first, last], axis=1), (-1, 4, BINS, 2))
        projected = keras.ops.conv(keras.ops.abs(pairs), self.projection, padding="valid")  # (1, 4, 129, 9)
        return keras.ops.reshape(keras.ops.max(projected, axis=-1), (-1, 4, 1, BINS))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4, 1, BINS)


def lower_silero_vad_v6_npu(model: keras.Model) -> keras.Model:
    """Rebuild a ``silero_vad_v6`` model from NPU operators, with its weights.

    Args:
        model: A ``silero_vad_v6`` model (layers ``stft``, ``encoder0`` to ``encoder3``, ``lstm``, ``prob``).

    Returns:
        keras.Model: The lowered model, with the same input and output names, batch size 1.
    """
    weights = {
        name: [keras.ops.convert_to_numpy(w) for w in model.get_layer(name).weights]
        for name in ("stft", "encoder0", "encoder1", "encoder2", "encoder3", "lstm", "prob")
    }
    audio = keras.Input((576,), batch_size=1, name="audio")
    h = keras.Input((128,), batch_size=1, name="state_in_0")
    c = keras.Input((128,), batch_size=1, name="state_in_1")
    frontend = SileroNpuFrontend(name="frontend")
    x = frontend(audio)  # (1, 4 frames, 1, 129 bins)
    conv1 = keras.layers.Conv2D(128, (3, 1), activation="relu", name="conv1")
    conv2 = keras.layers.Conv2D(64, (3, 1), strides=(2, 1), activation="relu", name="conv2")
    x = conv1(keras.layers.ZeroPadding2D(((1, 1), (0, 0)), name="conv1_pad")(x))
    x = conv2(keras.layers.ZeroPadding2D(((1, 1), (0, 0)), name="conv2_pad")(x))
    x = keras.layers.Reshape((128,), name="frames")(x)  # frame 0 channels, then frame 1 channels
    conv3 = keras.layers.Dense(64, activation="relu", name="conv3")
    conv4 = keras.layers.Dense(128, activation="relu", name="conv4")
    x = conv4(conv3(x))
    lstm = StreamingLSTMCell(128, name="lstm")
    h_next, c_next = lstm([x, h, c])
    head = keras.layers.Dense(1, activation="sigmoid", name="prob")
    prob = head(keras.layers.ReLU(name="decoder_relu")(h_next))
    lowered = keras.Model(
        [audio, h, c], [prob, state_output(0, h_next), state_output(1, c_next)], name=f"{model.name}_npu"
    )

    (basis,) = weights["stft"]
    main, folded = _block_kernels(basis)
    frontend.set_weights([main, folded, _projection_kernel()])
    (k1, b1), (k2, b2) = weights["encoder0"], weights["encoder1"]
    conv1.set_weights([k1[:, None], b1])
    conv2.set_weights([k2[:, None], b2])
    (k3, b3), (k4, b4) = weights["encoder2"], weights["encoder3"]
    conv3.set_weights([np.concatenate([k3[1], k3[2]], axis=0), b3])  # taps 1 and 2 see frames 0 and 1
    conv4.set_weights([k4[1], b4])  # the centre tap sees the only frame
    lstm.set_weights(weights["lstm"])
    head.set_weights(weights["prob"])
    return lowered
