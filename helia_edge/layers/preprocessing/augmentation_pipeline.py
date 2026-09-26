"""Serializable composition of preprocessing and training-only augmentation."""

import keras
from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.AugmentationPipeline")
class AugmentationPipeline(keras.Layer):
    def __init__(self, layers: list[keras.Layer], name=None, force_training=False, **kwargs):
        kwargs.setdefault("autocast", False)
        super().__init__(name=name, **kwargs)
        self.layers = layers
        self.force_training = force_training

    def call(self, inputs, training=None):
        for layer in self.layers:
            inputs = layer(inputs, training=True if self.force_training else training)
        return inputs

    def get_config(self):
        return {
            **super().get_config(),
            "layers": [keras.saving.serialize_keras_object(x) for x in self.layers],
            "force_training": self.force_training,
        }

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["layers"] = [keras.saving.deserialize_keras_object(x) for x in config["layers"]]
        return cls(**config)
