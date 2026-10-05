"""Fixtures shared across test packages."""

import json
import struct

import numpy as np
import pytest


@pytest.fixture
def write_safetensors():
    """Write ``{name: array}`` as a little-endian ``.safetensors`` file."""

    def write(path, tensors):
        header, offset, blobs = {}, 0, []
        codes = {
            np.dtype(np.float64): "F64",
            np.dtype(np.float32): "F32",
            np.dtype(np.float16): "F16",
            np.dtype(np.int64): "I64",
        }
        for name, value in tensors.items():
            blob = np.ascontiguousarray(value).astype(value.dtype.newbyteorder("<")).tobytes()
            header[name] = {
                "dtype": codes[value.dtype],
                "shape": list(value.shape),
                "data_offsets": [offset, offset + len(blob)],
            }
            offset += len(blob)
            blobs.append(blob)
        text = json.dumps({"__metadata__": {"format": "np"}, **header}).encode()
        path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))

    return write


SILERO_V6_SHAPES = {
    "model.stft.forward_basis_buffer": (258, 1, 256),
    "model.encoder.0.reparam_conv.weight": (128, 129, 3),
    "model.encoder.0.reparam_conv.bias": (128,),
    "model.encoder.1.reparam_conv.weight": (64, 128, 3),
    "model.encoder.1.reparam_conv.bias": (64,),
    "model.encoder.2.reparam_conv.weight": (64, 64, 3),
    "model.encoder.2.reparam_conv.bias": (64,),
    "model.encoder.3.reparam_conv.weight": (128, 64, 3),
    "model.encoder.3.reparam_conv.bias": (128,),
    "model.decoder.rnn.weight_ih": (512, 128),
    "model.decoder.rnn.weight_hh": (512, 128),
    "model.decoder.rnn.bias_ih": (512,),
    "model.decoder.rnn.bias_hh": (512,),
    "model.decoder.decoder.2.weight": (1, 128, 1),
    "model.decoder.decoder.2.bias": (1,),
}
"""Initializer names and shapes of silero_vad_16k_op15.onnx (v6.2.2)."""


@pytest.fixture
def silero_tensors():
    """Synthetic Silero VAD v6 initializers: ``silero_tensors(seed)`` returns ``{name: float32 array}``."""

    def make(seed=0):
        rng = np.random.default_rng(seed)
        tensors = {
            name: (rng.standard_normal(shape) / np.sqrt(np.prod(shape[1:]) or 1)).astype(np.float32)
            for name, shape in SILERO_V6_SHAPES.items()
        }
        tensors["model.stft.forward_basis_buffer"] *= 4
        return tensors

    return make
