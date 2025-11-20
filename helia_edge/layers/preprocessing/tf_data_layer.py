"""
# `tf.data.Dataset` Pipeline Layer

This module provides a class to build a `tf.data.Dataset` pipeline layer.

Classes:
    TFDataLayer: Layer that can safely used in a tf.data pipeline

"""

import keras
import tensorflow as tf
from keras.src.utils import backend_utils
from keras.src.utils import tracking

from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.TFDataLayer")
class TFDataLayer(keras.Layer):
    """Layer that can safely be used in a tf.data pipeline.

    The `call()` method must solely rely on `self.backend` ops.

    !!! note

        Only supports a single input tensor argument.
    """

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.backend = backend_utils.DynamicBackend()
        self._allow_non_tensor_positional_args = True

    def __call__(self, inputs, **kwargs):
        """When called, this layer sets the backend and device, and then calls the `call()` method.

        Args:
            inputs: The input tensor.
        """
        if backend_utils.in_tf_graph() and not isinstance(inputs, keras.KerasTensor):
            # We're in a TF graph, e.g. a tf.data pipeline.
            self.backend.set_backend("tensorflow")
            inputs = keras.tree.map_structure(
                lambda x: self.backend.convert_to_tensor(x, dtype=self.compute_dtype),
                inputs,
            )
            switch_convert_input_args = False
            if self._convert_input_args:
                self._convert_input_args = False
                switch_convert_input_args = True
            try:
                with tf.device(self.device):
                    outputs = super().__call__(inputs, **kwargs)
            finally:
                self.backend.reset()
                if switch_convert_input_args:
                    self._convert_input_args = True
            return outputs
        with tf.device(self.device):
            outputs = super().__call__(inputs, **kwargs)
        return outputs

    @tracking.no_automatic_dependency_tracking
    def _get_seed_generator(self, backend=None):
        if backend is None or backend == keras.backend.backend():
            return self.generator
        if not hasattr(self, "_backend_generators"):
            self._backend_generators = {}
        if backend in self._backend_generators:
            return self._backend_generators[backend]
        seed_generator = keras.random.SeedGenerator(self.seed, backend=self.backend)
        self._backend_generators[backend] = seed_generator
        return seed_generator
