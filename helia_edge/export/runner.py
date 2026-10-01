"""Run exported LiteRT model bytes with ai-edge-litert."""

import importlib
from collections.abc import Collection, Mapping
from typing import cast

import numpy as np
import numpy.typing as npt

from .spec import state_input_name, state_output_name, state_pair


def _litert_interpreter():
    try:
        module = importlib.import_module("ai_edge_litert.interpreter")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "LiteRTRunner requires the optional dependency 'ai-edge-litert'. Install helia-edge[litert]."
        ) from exc
    return module


class LiteRTRunner:
    """Run a single-input, single-output ``.tflite`` model one sample at a time.

    The input tensor is resized to each sample's shape along the model's dynamic dimensions only;
    fixed dimensions must match. Native float16 graphs run when the runtime has float16 kernels for
    every operator; otherwise the runtime's error is raised unchanged.

    Args:
        content: Model flatbuffer bytes.
        reference_kernels: Use LiteRT's reference kernels instead of its optimized kernels.
        num_threads: Interpreter threads.
    """

    def __init__(self, content: bytes, *, reference_kernels: bool = False, num_threads: int = 1) -> None:
        litert = _litert_interpreter()
        resolver = litert.OpResolverType.BUILTIN_REF if reference_kernels else litert.OpResolverType.AUTO
        self.interpreter = litert.Interpreter(
            model_content=content, num_threads=num_threads, experimental_op_resolver_type=resolver
        )
        self.interpreter.allocate_tensors()
        inputs, outputs = self.interpreter.get_input_details(), self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"LiteRTRunner supports one input and one output; got {len(inputs)} and {len(outputs)}")
        self.input, self.output = inputs[0], outputs[0]

    def run(self, x: npt.NDArray) -> npt.NDArray:
        """Invoke on samples along axis 0 already in the model's input dtype; return raw outputs."""
        if x.dtype != self.input["dtype"]:
            raise ValueError(f"Input dtype {x.dtype} does not match the model input {np.dtype(self.input['dtype'])}")
        outputs = []
        for sample in x:
            if tuple(self.interpreter.get_input_details()[0]["shape"]) != (1, *sample.shape):
                self.interpreter.resize_tensor_input(self.input["index"], (1, *sample.shape), strict=True)
                self.interpreter.allocate_tensors()
            self.interpreter.set_tensor(self.input["index"], sample[None])
            self.interpreter.invoke()
            outputs.append(self.interpreter.get_tensor(self.output["index"])[0])
        return np.stack(outputs)

    def encode(self, x: npt.NDArray) -> npt.NDArray:
        """Convert real-valued samples to the input dtype, rounding and saturating integer inputs."""
        dtype = np.dtype(self.input["dtype"])
        if dtype.kind == "f":
            return x.astype(dtype)
        scale, zero_point = self.input["quantization"]
        info = np.iinfo(cast(type[np.integer], dtype.type))
        return np.clip(np.rint(x / scale + zero_point), info.min, info.max).astype(dtype)

    def decode(self, y: npt.NDArray) -> npt.NDArray:
        """Convert raw outputs to float32 values."""
        if np.dtype(self.output["dtype"]).kind == "f":
            return y.astype(np.float32)
        scale, zero_point = self.output["quantization"]
        return ((y.astype(np.float32) - zero_point) * scale).astype(np.float32)

    def predict(self, x: npt.NDArray) -> npt.NDArray:
        """Encode real-valued samples, run them and decode the outputs."""
        return self.decode(self.run(self.encode(x)))


def _encode(details: dict, x: npt.NDArray) -> npt.NDArray:
    dtype = np.dtype(details["dtype"])
    if dtype.kind == "f":
        return x.astype(dtype)
    scale, zero_point = details["quantization"]
    info = np.iinfo(cast(type[np.integer], dtype.type))
    return np.clip(np.rint(x / scale + zero_point), info.min, info.max).astype(dtype)


def _decode(details: dict, y: npt.NDArray) -> npt.NDArray:
    if np.dtype(details["dtype"]).kind == "f":
        return y.astype(np.float32)
    scale, zero_point = details["quantization"]
    return ((y.astype(np.float32) - zero_point) * scale).astype(np.float32)


class LiteRTStreamRunner:
    """Run a streaming ``.tflite`` model one step at a time, carrying its state as raw tensors.

    Inputs and outputs are addressed by signature name. State inputs ``state_in_k`` start at zero, and at
    each later step take the raw ``state_out_k`` of the previous step unchanged, so an integer state keeps
    its value only if the pair has one scale and zero point (``export_model`` ties them).

    Args:
        content: Model flatbuffer bytes with one signature.
        reference_kernels: Use LiteRT's reference kernels instead of its optimized kernels.
        num_threads: Interpreter threads.

    Raises:
        ValueError: If the model does not have exactly one signature, or its state inputs and outputs
            do not pair up.
    """

    def __init__(self, content: bytes, *, reference_kernels: bool = False, num_threads: int = 1) -> None:
        litert = _litert_interpreter()
        resolver = litert.OpResolverType.BUILTIN_REF if reference_kernels else litert.OpResolverType.AUTO
        self.interpreter = litert.Interpreter(
            model_content=content, num_threads=num_threads, experimental_op_resolver_type=resolver
        )
        self.interpreter.allocate_tensors()
        signatures = list(self.interpreter.get_signature_list())
        if len(signatures) != 1:
            raise ValueError(f"LiteRTStreamRunner needs a model with one signature; got {len(signatures)}")
        runner = self.interpreter.get_signature_runner(signatures[0])
        self.inputs: dict[str, dict] = dict(runner.get_input_details())
        self.outputs: dict[str, dict] = dict(runner.get_output_details())
        ins = sorted(p[1] for name in self.inputs if (p := state_pair(name)) and p[0] == "in")
        outs = sorted(p[1] for name in self.outputs if (p := state_pair(name)) and p[0] == "out")
        if ins != outs:
            raise ValueError(f"State inputs {ins} and state outputs {outs} do not pair up")
        self.pairs: tuple[int, ...] = tuple(ins)
        self.signals: tuple[str, ...] = tuple(n for n in self.inputs if n not in map(state_input_name, self.pairs))

    def initial_state(self) -> dict[str, npt.NDArray]:
        """Each state input at zero: the zero point for an integer state."""
        return {
            name: _encode(self.inputs[name], np.zeros(self.inputs[name]["shape"], np.float32))
            for name in map(state_input_name, self.pairs)
        }

    def step(self, inputs: Mapping[str, npt.NDArray]) -> dict[str, npt.NDArray]:
        """Invoke once on raw tensors for every input name; return every output by name."""
        if set(inputs) != set(self.inputs):
            raise ValueError(f"Inputs {sorted(inputs)} do not match the model inputs {sorted(self.inputs)}")
        for name, details in self.inputs.items():
            value = np.asarray(inputs[name])
            if value.dtype != details["dtype"] or value.shape != tuple(details["shape"]):
                raise ValueError(
                    f"Input {name!r} is {value.dtype}{list(value.shape)}; the model takes "
                    f"{np.dtype(details['dtype'])}{list(details['shape'])}"
                )
            self.interpreter.set_tensor(details["index"], value)
        self.interpreter.invoke()
        return {name: self.interpreter.get_tensor(details["index"]).copy() for name, details in self.outputs.items()}

    def run(
        self, signals: Mapping[str, npt.NDArray], resets: Collection[int] = ()
    ) -> tuple[dict[str, npt.NDArray], dict[str, npt.NDArray]]:
        """Stream raw signal inputs, carrying the state.

        Args:
            signals: Every input that is not a state, by name, as raw tensors stacked along a leading step
                axis (``[steps, *input shape]``).
            resets: Steps at which the state inputs return to ``initial_state``.

        Returns:
            tuple: Every input as fed and every output as produced, by name, stacked along the step axis.
        """
        if sorted(signals) != sorted(self.signals):
            raise ValueError(
                f"Signals {sorted(signals)} do not match the inputs that are not states {sorted(self.signals)}"
            )
        steps = {len(values) for values in signals.values()}
        if len(steps) != 1:
            raise ValueError("Signals have different numbers of steps")
        state = self.initial_state()
        fed: dict[str, list] = {name: [] for name in self.inputs}
        produced: dict[str, list] = {name: [] for name in self.outputs}
        for t in range(steps.pop()):
            if t in resets:
                state = self.initial_state()
            feed = {name: signals[name][t] for name in self.signals} | state
            outputs = self.step(feed)
            for name in self.inputs:
                fed[name].append(feed[name])
            for name in self.outputs:
                produced[name].append(outputs[name])
            state = {state_input_name(k): outputs[state_output_name(k)] for k in self.pairs}
        return {n: np.stack(v) for n, v in fed.items()}, {n: np.stack(v) for n, v in produced.items()}

    def encode(self, name: str, x: npt.NDArray) -> npt.NDArray:
        """Convert real values to input ``name``'s dtype, rounding and saturating integer inputs."""
        return _encode(self.inputs[name], x)

    def decode(self, name: str, y: npt.NDArray) -> npt.NDArray:
        """Convert raw values of output ``name`` to float32."""
        return _decode(self.outputs[name], y)
