"""
# TFLite Converter API

This module handles converting models to TensorFlow Lite format.

Classes:
    QuantizationType: Enum class for quantization types.
    TfLiteKerasConverter: TensorFlow Lite model converter.
    ConversionType: Enum class for conversion types.

"""

import io
import tempfile
import warnings
from enum import StrEnum
from pathlib import Path

import keras
import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf

from ...export.litert import convert_litert
from ...export.spec import LEGACY_MODE, LEGACY_PRECISION
from ...models import load_model
from ..cpp import xxd_c_dump


def _reject_native_fp16(interpreter) -> None:
    details = interpreter.get_input_details() + interpreter.get_output_details()
    if any(detail["dtype"] == np.float16 for detail in details):
        raise ValueError("Native float16 models need an engine with float16 kernels; predict() does not run them.")


def _warn_deprecated(name: str, stacklevel: int) -> None:
    warnings.warn(
        f"{name} is deprecated; use helia_edge.export.export_model with an ExportSpec, "
        "and helia_edge.export.LiteRTRunner to run the result.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


class QuantizationType(StrEnum):
    """Supported quantization formats

    Attributes:
        FP32: FP32 quantization
        FP16: float16 weight storage; weights are dequantized and compute stays FP32
        FP16_NATIVE: native float16 graph; inputs, weights, activations and outputs are float16
        INT8: INT8 quantization
        INT16X8: INT16X8 quantization

    """

    FP32 = "FP32"
    FP16 = "FP16"
    FP16_NATIVE = "FP16_NATIVE"
    INT8 = "INT8"
    INT16X8 = "INT16X8"


class ConversionType(StrEnum):
    """Supported conversion types

    Attributes:
        KERAS: Use Keras model directly
        SAVED_MODEL: Use TF Saved model format
        CONCRETE: Lower to TF Concrete functions
    """

    KERAS = "KERAS"
    SAVED_MODEL = "SAVED_MODEL"
    CONCRETE = "CONCRETE"


class TfLiteKerasConverter:
    def __init__(
        self,
        model: keras.Model,
    ):
        """Converts Keras model to TFLite model.

        Deprecated: use ``helia_edge.export.export_model`` with an ``ExportSpec``, and
        ``helia_edge.export.LiteRTRunner`` to run the result. Constructing this class emits a
        ``DeprecationWarning``; its behaviour is unchanged.

        Args:
            model (keras.Model): Keras model

        Example:

        ```python
        # Create simple dataset
        test_x = np.random.rand(1000, 64).astype(np.float32)
        test_y = np.random.randint(0, 10, 1000).astype(np.int32)

        # Create a dense model and train
        model = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=(64,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(test_x, test_y, epochs=1, validation_split=0.2)

        import helia_edge as helia

        # Create converter and convert to TFLite w/ FP32 quantization
        converter = helia.converters.tflite.TfLiteKerasConverter(model=model)
        tflite_content = converter.convert(
            test_x,
            quantization=helia.converters.tflite.QuantizationType.FP32,
            io_type="float32"
        )
        y_pred_tfl = converter.predict(test_x)
        y_pred_tf = model.predict(test_x)
        print(np.allclose(y_pred_tf, y_pred_tfl, atol=1e-3))
        ```
        """
        _warn_deprecated(type(self).__name__, stacklevel=3)
        self.model = model
        self.representative_dataset = None
        self._converter: tf.lite.TFLiteConverter | None = None
        self._tflite_content: bytes | None = None
        self.tf_model_path = tempfile.TemporaryDirectory()

    @classmethod
    def from_saved_model(cls, model_path: Path) -> "TfLiteKerasConverter":
        """Create converter from saved keras model

        Args:
            model_path (Path): Path to saved model

        Returns:
            TfLiteKerasConverter: Converter
        """
        model = load_model(model_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            converter = cls(model=model)
        _warn_deprecated(cls.__name__, stacklevel=3)
        return converter

    def convert(
        self,
        test_x: npt.NDArray | None = None,
        quantization: QuantizationType = QuantizationType.FP32,
        io_type: str | None = None,
        mode: ConversionType = ConversionType.KERAS,
        strict: bool = True,
        verbose: int = 2,
    ) -> bytes:
        """Convert TF model into TFLite model content

        Args:
            test_x (npt.NDArray | None, optional): Representative samples, required for INT8 and INT16X8.
                Defaults to None.
            quantization (QuantizationType, optional): Quantization type. Defaults to QuantizationType.FP32.
                FP16_NATIVE produces a float16 graph for engines with float16 kernels. The TensorFlow Lite
                interpreter cannot run its convolution or fully connected operators, and predict() rejects it.
            io_type (str | None, optional): Input/Output type. Defaults to None; FP16_NATIVE is always float16.
            mode (ConversionType, optional): Conversion mode. Defaults to ConversionType.KERAS.
            strict (bool, optional): Strict mode. Defaults to True.
            verbose (int, optional): Verbosity level (0,1,2). Defaults to 2.

        Returns:
            bytes: TFLite content
        """
        quantization = QuantizationType(quantization)
        if quantization == QuantizationType.FP16_NATIVE and io_type not in (None, "float16"):
            raise ValueError("FP16_NATIVE models always use float16 inputs and outputs")
        if quantization in (QuantizationType.INT8, QuantizationType.INT16X8) and test_x is None:
            raise ValueError(f"{quantization.value} conversion requires representative data passed via test_x.")
        conversion = convert_litert(
            self.model,
            precision=LEGACY_PRECISION[quantization],
            io_type=io_type,
            mode=LEGACY_MODE[ConversionType(mode)],
            strict=strict,
            calibration=test_x,
            workdir=self.tf_model_path.name,
        )
        self._converter = conversion.converter
        self.representative_dataset = conversion.representative_dataset
        self._tflite_content = conversion.content
        return self._tflite_content

    def debug_quantization(self) -> pd.DataFrame:
        """Debug quantized TFLite model content."""

        if self._converter is None:
            raise ValueError("No TFLite content to debug. Run convert() first.")

        if self.representative_dataset is None:
            raise ValueError("No representative dataset provided. Run convert() with test_x first.")

        # Debug model
        debugger = tf.lite.experimental.QuantizationDebugger(
            converter=self._converter, debug_dataset=self.representative_dataset
        )
        debugger.run()

        with io.StringIO() as f:
            debugger.layer_statistics_dump(f)
            f.seek(0)
            layer_stats = pd.read_csv(f)
        # END WITH

        # Add custom metrics
        layer_stats["range"] = 255.0 * layer_stats["scale"]
        layer_stats["rmse/scale"] = layer_stats.apply(
            lambda row: np.sqrt(row["mean_squared_error"]) / row["scale"], axis=1
        )
        return layer_stats

    def predict(
        self,
        x: npt.NDArray,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        # Prepare the test data
        inputs = x.copy()
        inputs = inputs.astype(np.float32)

        interpreter = tf.lite.Interpreter(model_content=self._tflite_content)
        _reject_native_fp16(interpreter)
        interpreter.allocate_tensors()

        # No signature
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

        # WITH Signature
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

    def evaluate(
        self,
        x: npt.NDArray,
        y: npt.NDArray,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> npt.NDArray:
        """Evaluate TFLite model

        Args:
            x (npt.NDArray): Input samples
            y (npt.NDArray): Input labels
            input_name (str | None, optional): Input layer name. Defaults to None.
            output_name (str | None, optional): Output layer name. Defaults to None.

        Returns:
            npt.NDArray: Loss values
        """
        y_pred = self.predict(
            x=x,
            input_name=input_name,
            output_name=output_name,
        )
        loss_function = keras.losses.get(self.model.loss)
        loss = loss_function(y, y_pred).numpy()
        return loss

    def export(self, tflite_path: str):
        """Export TFLite model content to file

        Args:
            tflite_path (str): TFLite file path
        """
        if self._tflite_content is None:
            raise ValueError("No TFLite content to export. Run convert() first.")

        with open(tflite_path, "wb") as f:
            f.write(self._tflite_content)

    def export_header(self, header_path: str, name: str = "model"):
        """Export TFLite model as C header file.

        Args:
            header_path (str): Header file path
            name (str, optional): Variable name. Defaults to "model".
        """
        with tempfile.NamedTemporaryFile() as f:
            self.export(f.name)
            xxd_c_dump(
                src_path=f.name,
                dst_path=header_path,
                var_name=name,
                chunk_len=20,
                is_header=True,
            )
        # END WITH

    def cleanup(self):
        """Cleanup temporary files"""
        self.tf_model_path.cleanup()
