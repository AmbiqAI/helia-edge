"""Golden files (``helia-model-zoo/golden@2``) from an exported streaming model."""

import io
from collections.abc import Collection
from typing import Any, cast

import numpy as np
import numpy.typing as npt


def golden_npz(content: bytes, frames: npt.NDArray, resets: Collection[int] = ()) -> bytes:
    """A golden@2 sequence for a streaming model: ``frames`` are consecutive calls of its signal input.

    The model runs with LiteRT's reference kernels, each ``state_out_k`` fed back as ``state_in_k`` and
    the states reset to zero at the steps in ``resets``. The NPZ holds ``input_i`` and ``output_i`` in
    subgraph order, as raw (quantized) values.

    Raises:
        ValueError: If the model has more than one input that is not a state.
    """
    from .runner import LiteRTStreamRunner

    runner = LiteRTStreamRunner(content, reference_kernels=True)
    if len(runner.signals) != 1:
        raise ValueError(
            f"A streaming reference needs exactly one input that is not a state; got {list(runner.signals)}"
        )
    (signal,) = runner.signals
    encoded = runner.encode(signal, np.asarray(frames).reshape(len(frames), *runner.inputs[signal]["shape"]))
    fed, produced = runner.run({signal: encoded}, resets)
    by_index = {details["index"]: name for name, details in runner.inputs.items()}
    out_by_index = {details["index"]: name for name, details in runner.outputs.items()}
    order_in = [by_index[d["index"]] for d in runner.interpreter.get_input_details()]
    order_out = [out_by_index[d["index"]] for d in runner.interpreter.get_output_details()]
    arrays = {f"input_{i}": fed[name] for i, name in enumerate(order_in)}
    arrays |= {f"output_{i}": produced[name] for i, name in enumerate(order_out)}
    buffer = io.BytesIO()
    np.savez(buffer, **cast(dict[str, Any], arrays))  # plain savez: zlib versions cannot change the bytes
    return buffer.getvalue()
