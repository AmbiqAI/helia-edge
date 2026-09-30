"""Run exported LiteRT model bytes with ai-edge-litert."""

import importlib

import numpy as np
import numpy.typing as npt


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
        info = np.iinfo(dtype)
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
