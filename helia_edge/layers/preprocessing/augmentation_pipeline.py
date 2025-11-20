"""
# Augmentation Pipeline API

Classes:
    AugmentationPipeline: Pipeline of augmentation layers
"""

import keras

from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.AugmentationPipeline")
class AugmentationPipeline(keras.Layer):
    def __init__(
        self,
        layers: list[keras.Layer],
        name: str | None = None,
        force_training: bool = False,
    ):
        """Pipeline of augmentation layers.

        Args:
            layers (list[keras.Layer]): List of augmentation layers.
            force_training (bool, optional): Force training mode. Defaults to False.

        Example:

        ```python
        import helia_edge as helia

        layers = [
            helia.layers.preprocessing.RandomNoiseDistortion1D(sample_rate=100, frequency=(1, 2), amplitude=(0.5, 2)),
            helia.layers.preprocessing.AmplitudeWarp(sample_rate=100, frequency=(1, 2), amplitude=(0.5, 2)),
        ]
        pipeline = helia.layers.preprocessing.AugmentationPipeline(layers)
        x = keras.random.normal((10, 100, 1), dtype="float32")
        x_aug = pipeline(x, training=True)
        plt.plot(x[0].numpy())
        plt.plot(x_aug[0].numpy())
        plt.show()
        ```
        """
        super().__init__(name=name)
        self.layers = layers
        self.force_training = force_training

    def call(self, inputs, training: bool = True, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, training=training or self.force_training, **kwargs)
        return inputs
