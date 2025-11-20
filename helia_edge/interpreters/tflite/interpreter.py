"""
# TFLite Interpreter API

This module handles interpreting TensorFlow Lite models.

Classes:
    TfLiteKerasInterpreter: Interprets TensorFlow Lite models converted from Keras.

"""

import numpy as np
import numpy.typing as npt

import tensorflow as tf


class TfLiteKerasInterpreter:
    def __init__(
        self,
        model_content: str,
        input_name: str | None = None,
        output_name: str | None = None,
        signature_key: str | None = None,
    ):
        """TFLite model interpreter that takes care of I/O conversion and prediction.

        Args:
            model_content (str): TFLite model content
            input_name (str | None, optional): Input layer name. Defaults to None.
            output_name (str | None, optional): Output layer name. Defaults to None.
            signature_key (str | None, optional): Signature key. Defaults to None

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
        interpreter = helia.interpreters.tflite.TfLiteKerasInterpreter(model_content=tflite_content)
        interpreter.compile()

        y_pred_tfl = interpreter.predict(test_x)
        y_pred_tf = model.predict(test_x)
        print(np.allclose(y_pred_tf, y_pred_tfl, atol=1e-3))
        ```
        """
        self.model_content = model_content
        self.interpreter = tf.lite.Interpreter(model_content=model_content)
        self.interpreter.allocate_tensors()

        self.signature_key = signature_key
        self._has_signature = False

        self._input_name = input_name
        self._input_shape = None
        self._input_scale = None
        self._input_zero_point = None
        self._input_dtype = "float32"

        self._output_name = output_name
        self._output_scale = None
        self._output_zero_point = None
        self._output_dtype = "float32"

    def compile(self):
        """Compile model and extract input/output details."""

        # Some models may lose signature after converting to tflite due to TF issues.
        # Most prevalent for models lowered to concrete functions.
        self._has_signature = len(self.interpreter.get_signature_list()) > 0

        if not self._has_signature:
            input_details = self.interpreter.get_input_details()[0]
            output_details = self.interpreter.get_output_details()[0]
            self._input_shape = input_details["shape_signature"].tolist()
            self._input_name = input_details["index"]
            self._output_name = output_details["index"]

        else:
            model_sig = self.interpreter.get_signature_runner(self.signature_key)
            inputs_details = model_sig.get_input_details()
            outputs_details = model_sig.get_output_details()
            if self._input_name is None:
                self._input_name = list(inputs_details.keys())[0]
            if self._output_name is None:
                self._output_name = list(outputs_details.keys())[0]
            input_details = inputs_details[self._input_name]
            output_details = outputs_details[self._output_name]
            self._input_shape = input_details["shape_signature"].tolist()[1:]
        # END IF

        input_scale: list[float] = input_details["quantization_parameters"]["scales"]
        input_zero_point: list[int] = input_details["quantization_parameters"]["zero_points"]
        output_scale: list[float] = output_details["quantization_parameters"]["scales"]
        output_zero_point: list[int] = output_details["quantization_parameters"]["zero_points"]

        self._input_dtype = input_details["dtype"]
        if len(input_scale) and len(input_zero_point):
            self._input_scale = input_scale[0]
            self._input_zero_point = input_zero_point[0]
        # END IF

        self._output_dtype = output_details["dtype"]
        if len(output_scale) and len(output_zero_point):
            self._output_scale = output_scale[0]
            self._output_zero_point = output_zero_point[0]
        # END IF

    def convert_input(self, x: npt.NDArray) -> npt.NDArray:
        """Convert input data based on quantization.

        NOTE: predict() will call this method internally.

        Args:
            x (npt.NDArray): Input samples

        Returns:
            npt.NDArray: Prepared input samples
        """
        inputs = x.copy()
        inputs = inputs.reshape([-1] + self._input_shape)
        if self._input_scale and self._input_zero_point:
            inputs = inputs / self._input_scale + self._input_zero_point
        inputs = inputs.astype(self._input_dtype)
        return inputs

    def convert_output(self, outputs: npt.NDArray) -> npt.NDArray:
        """Convert output data based on quantization.

        NOTE: predict() will call this method internally.

        Args:
            outputs (npt.NDArray): Output samples

        Returns:
            npt.NDArray: Prepared output samples
        """
        outputs = outputs.astype(np.float32)
        if self._output_scale and self._output_zero_point:
            outputs = (outputs - self._output_zero_point) * self._output_scale
        return outputs

    def predict(
        self,
        x: npt.NDArray,
    ) -> npt.NDArray:
        """Predict using TFLite model

        Args:
            x (npt.NDArray): Input samples

        Returns:
            npt.NDArray: Predicted values
        """
        inputs = self.convert_input(x)

        if not self._has_signature:
            outputs = []
            for sample in inputs:
                self.interpreter.set_tensor(self._input_name, sample)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(self._output_name)
                outputs.append(y)
            outputs = np.concatenate(outputs, axis=0)
        else:
            model_sig = self.interpreter.get_signature_runner(self.signature_key)
            outputs = np.array(
                [
                    model_sig(**{self._input_name: inputs[i : i + 1]})[self._output_name][0]
                    for i in range(inputs.shape[0])
                ],
                dtype=self._output_dtype,
            )
        # END IF

        outputs = self.convert_output(outputs)

        return outputs
