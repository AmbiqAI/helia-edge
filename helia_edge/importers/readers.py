"""Read named tensors from source files: ``path -> {name: array}``.

The ``onnx`` reader needs the ``onnx`` extra and the ``torch`` reader the ``torch`` extra; both are
imported only when called. The ``safetensors`` reader uses NumPy alone.
"""

import json
import struct
from pathlib import Path

import numpy as np
import numpy.typing as npt

_SAFETENSORS_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def read_onnx(path: Path) -> dict[str, npt.NDArray]:
    """The initializers of a self-contained ``.onnx`` model, by name.

    Weights held in ``Constant`` nodes rather than initializers are not read; such a source tensor is
    reported as not in the file.

    Raises:
        ValueError: If an initializer is stored in a separate data file, which the source's sha256 does
            not cover.
    """
    try:
        import onnx
        from onnx import numpy_helper
        from onnx.external_data_helper import uses_external_data
    except ModuleNotFoundError as exc:
        raise ImportError("Reading ONNX files needs the optional dependency 'onnx'. Install helia-edge[onnx].") from exc
    model = onnx.load(str(path), load_external_data=False)
    external = [tensor.name for tensor in model.graph.initializer if uses_external_data(tensor)]
    if external:
        raise ValueError(
            f"{path} stores initializers {external} in separate data files; import a self-contained .onnx file"
        )
    return {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}


def read_safetensors(path: Path) -> dict[str, npt.NDArray]:
    """The tensors of a ``.safetensors`` file, by name (little-endian; BF16 is not supported)."""
    data = Path(path).read_bytes()
    (header_size,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + header_size])
    header.pop("__metadata__", None)
    body = memoryview(data)[8 + header_size :]
    tensors = {}
    for name, entry in header.items():
        if entry["dtype"] not in _SAFETENSORS_DTYPES:
            raise ValueError(f"Tensor {name!r} has unsupported dtype {entry['dtype']}")
        start, end = entry["data_offsets"]
        dtype = np.dtype(_SAFETENSORS_DTYPES[entry["dtype"]]).newbyteorder("<")
        tensors[name] = np.frombuffer(body[start:end], dtype=dtype).reshape(entry["shape"]).copy()
    return tensors


def read_torch(path: Path) -> dict[str, npt.NDArray]:
    """The tensors of a PyTorch state dict saved with ``torch.save``, loaded with ``weights_only=True``."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ImportError("Reading PyTorch files needs torch. Install helia-edge[torch].") from exc
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"{path} is not a flat state dict of tensors")
    tensors = {}
    for name, value in state.items():
        try:
            tensors[name] = value.detach().cpu().numpy()
        except TypeError as exc:
            raise ValueError(f"Tensor {name!r} has dtype {value.dtype}, which NumPy cannot hold") from exc
    return tensors
