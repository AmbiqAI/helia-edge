"""
# TFLite Interpreter API

This module handles interpreting TensorFlow Lite models.

Classes:
    TfLiteKerasInterpreter: Interprets TensorFlow Lite models converted from Keras.

"""

from helia_edge._lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "TfLiteKerasInterpreter": (".interpreter", "TfLiteKerasInterpreter"),
        "interpreter": (".interpreter", None),
    },
)
