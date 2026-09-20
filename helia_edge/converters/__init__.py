"""
# :material-export: Converters API

The `converters` module provides classes to convert and export models to various inference engine foramts. For example, you can convert a Keras model to a TensorFlow Lite model using [TfLiteKerasConverter](./tflite).

While `Keras` and `TensorFlow` provide built-in methods to export they often have inconsistencies and require additional steps to convert between formats. The `converters` module provides a consistent interface to convert and export to various formats.

## Available Converters

* **[C++ Converter](./cpp)**: Convert models to C++ executables.
* **[LiteRT Converter](./litert)**: Convert models to LiteRT format.
* **[TFLite Converter](./tflite)**: Convert models to TensorFlow Lite format.
* **[Torch Converter](./torch)**: Convert models to PyTorch formats such as ExecuTorch.

"""

from helia_edge._lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "cpp": (".cpp", None),
        "litert": (".litert", None),
        "tflite": (".tflite", None),
        "torch": (".torch", None),
    },
)
