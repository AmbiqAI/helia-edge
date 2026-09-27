"""Batchwise random layer choice with captured branches and owned RNG."""

import keras
from .base_augmentation import BaseAugmentation
from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.RandomChoice")
class RandomChoice(BaseAugmentation):
    """Choose one layer for a whole batch; branches must have compatible outputs.

    Per-example layer choice was never implemented and remains unsupported.
    The next-major default is the supported ``batchwise=True`` mode.
    """

    def __init__(self, layers: list[keras.Layer], batchwise: bool = True, **kwargs):
        super().__init__(**kwargs)
        if not layers:
            raise ValueError("At least one layer is required")
        if not batchwise:
            raise NotImplementedError("Per-example layer choice is not supported; use batchwise=True")
        self.layers = layers
        self.batchwise = batchwise

    def build(self, input_shape):
        for layer in self.layers:
            if not layer.built:
                layer.build(input_shape)
        super().build(input_shape)

    def batch_augment(self, inputs, transformations=None):
        index = (
            keras.random.randint((), 0, len(self.layers), seed=self.random_generator, dtype="int32")
            if transformations is None
            else transformations["index"]
        )
        branches = [lambda x, layer=layer: layer(x, training=True) for layer in self.layers]
        return keras.ops.switch(index, branches, inputs)

    def call(self, inputs, training=None, transformations=None):
        inputs = self._identity(inputs, validate_rank=False)
        if training is None or training is False:
            return inputs
        if training is True:
            return self.batch_augment(inputs, transformations)
        return keras.ops.cond(training, lambda: self.batch_augment(inputs, transformations), lambda: inputs)

    def get_config(self):
        return {
            **super().get_config(),
            "layers": [keras.saving.serialize_keras_object(x) for x in self.layers],
            "batchwise": self.batchwise,
        }

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["layers"] = [keras.saving.deserialize_keras_object(x) for x in config["layers"]]
        return cls(**config)
