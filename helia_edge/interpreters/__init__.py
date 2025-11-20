"""
# Interpreters

Interpreters are classes that are used to interpret pre-trained model files such as TensorFlow and TensorFlow Lit.
These classes are used to load the model file and provide a consistent interface for making predictions.
Often a `helia_edge.converters` module is used to convert the model and `helia_edge.interpreters` is used to verify the model results.

## Available Interpreters

* [TFLite Interpreter](./tflite): Interprets TensorFlow Lite models.

"""

from . import tflite
