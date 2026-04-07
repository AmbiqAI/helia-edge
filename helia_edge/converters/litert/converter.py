"""
# LiteRT Converter API

This module handles converting models to LiteRT format.

Classes:
    LiteRTKerasConverter: LiteRT model converter.

"""

import importlib

import numpy as np
import numpy.typing as npt

from ..tflite import ConversionType, QuantizationType, TfLiteKerasConverter


def _load_litert_interpreter():
    try:
        module = importlib.import_module("ai_edge_litert.interpreter")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "LiteRT support requires the optional dependency 'ai-edge-litert>=2.1.3'. "
            "Install helia-edge[litert] to enable LiteRT conversion and runtime validation."
        ) from exc
    return module.Interpreter


class LiteRTKerasConverter(TfLiteKerasConverter):
    """Converts Keras model to LiteRT model content."""

    def convert(
        self,
        test_x: npt.NDArray | None = None,
        quantization: QuantizationType = QuantizationType.FP32,
        io_type: str | None = None,
        mode: ConversionType = ConversionType.KERAS,
        strict: bool = True,
        verbose: int = 2,
    ) -> bytes:
        """Convert TF model into LiteRT model content."""
        _load_litert_interpreter()
        quantization = QuantizationType(quantization)
        ConversionType(mode)
        if test_x is None and quantization in {QuantizationType.INT8, QuantizationType.INT16X8}:
            raise ValueError("LiteRT quantized conversion requires representative data passed via test_x.")
        return super().convert(
            test_x=test_x,
            quantization=quantization,
            io_type=io_type,
            mode=mode,
            strict=strict,
            verbose=verbose,
        )

    def predict(
        self,
        x: npt.NDArray,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Run LiteRT inference for the converted model."""
        if self._tflite_content is None:
            raise ValueError("No LiteRT content to predict. Run convert() first.")

        interpreter_cls = _load_litert_interpreter()

        inputs = x.copy().astype(np.float32)
        interpreter = interpreter_cls(model_content=self._tflite_content)
        interpreter.allocate_tensors()

        if len(interpreter.get_signature_list()) == 0:
            output_details = interpreter.get_output_details()[0]
            input_details = interpreter.get_input_details()[0]

            input_scale: list[float] = input_details["quantization_parameters"]["scales"]
            input_zero_point: list[int] = input_details["quantization_parameters"]["zero_points"]
            output_scale: list[float] = output_details["quantization_parameters"]["scales"]
            output_zero_point: list[int] = output_details["quantization_parameters"]["zero_points"]

            inputs = inputs.reshape([-1] + input_details["shape_signature"].tolist())
            if len(input_scale) and len(input_zero_point):
                inputs = inputs / input_scale[0] + input_zero_point[0]
                inputs = inputs.astype(input_details["dtype"])

            outputs = []
            for sample in inputs:
                interpreter.set_tensor(input_details["index"], sample)
                interpreter.invoke()
                y = interpreter.get_tensor(output_details["index"])
                outputs.append(y)
            outputs = np.concatenate(outputs, axis=0)

            if len(output_scale) and len(output_zero_point):
                outputs = outputs.astype(np.float32)
                outputs = (outputs - output_zero_point[0]) * output_scale[0]

            return outputs

        model_sig = interpreter.get_signature_runner()
        inputs_details = model_sig.get_input_details()
        outputs_details = model_sig.get_output_details()
        if input_name is None:
            input_name = list(inputs_details.keys())[0]
        if output_name is None:
            output_name = list(outputs_details.keys())[0]
        input_details = inputs_details[input_name]
        output_details = outputs_details[output_name]
        input_scale: list[float] = input_details["quantization_parameters"]["scales"]
        input_zero_point: list[int] = input_details["quantization_parameters"]["zero_points"]
        output_scale: list[float] = output_details["quantization_parameters"]["scales"]
        output_zero_point: list[int] = output_details["quantization_parameters"]["zero_points"]

        inputs = inputs.reshape([-1] + input_details["shape_signature"].tolist()[1:])
        if len(input_scale) and len(input_zero_point):
            inputs = inputs / input_scale[0] + input_zero_point[0]
            inputs = inputs.astype(input_details["dtype"])

        outputs = np.array(
            [model_sig(**{input_name: inputs[i : i + 1]})[output_name][0] for i in range(inputs.shape[0])],
            dtype=output_details["dtype"],
        )

        if len(output_scale) and len(output_zero_point):
            outputs = outputs.astype(np.float32)
            outputs = (outputs - output_zero_point[0]) * output_scale[0]

        return outputs
