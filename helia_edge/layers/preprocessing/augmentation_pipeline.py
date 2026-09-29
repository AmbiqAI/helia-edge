"""Serializable composition of preprocessing and training-only augmentation."""

import keras
from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.AugmentationPipeline")
class AugmentationPipeline(keras.Layer):
    """Apply layers in order to a tensor or structured sample.

    Each child receives the preceding child's output and the training flag.
    Layers must accept ``training`` and, when supplied, ``transformations``.
    Calling with transformations requires one entry per layer; use None for
    a child that should sample its own parameters.

    Args:
        layers: Ordered layers to apply. The layer objects are retained.
        name (str | None): Optional Keras layer name.
        force_training (bool): Pass training=True to every child, including at inference.
        **kwargs (Any): Additional Keras Layer arguments.

    Raises:
        ValueError: The supplied transformations count differs from the layer count.
    """

    def __init__(self, layers: list[keras.Layer], name=None, force_training=False, **kwargs):
        kwargs.setdefault("autocast", False)
        super().__init__(name=name, **kwargs)
        self.layers = layers
        self.force_training = force_training

    def call(self, inputs, training=None, transformations=None):
        if transformations is not None and len(transformations) != len(self.layers):
            raise ValueError("transformations must contain one entry per pipeline layer")
        for index, layer in enumerate(self.layers):
            params = None if transformations is None else transformations[index]
            kwargs = {} if params is None else {"transformations": params}
            inputs = layer(inputs, training=True if self.force_training else training, **kwargs)
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
