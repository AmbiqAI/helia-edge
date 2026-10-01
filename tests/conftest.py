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
        codes = {np.dtype(np.float32): "F32", np.dtype(np.float16): "F16", np.dtype(np.int64): "I64"}
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
