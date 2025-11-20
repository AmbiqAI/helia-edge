"""
# Random Choice Layer API

This module provides classes to build random choice layers.

Classes:
    RandomChoice: Random choice

"""

import keras
from .base_augmentation import BaseAugmentation
from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.RandomChoice")
class RandomChoice(BaseAugmentation):
    layers: list[BaseAugmentation]
    batchwise: bool

    def __init__(self, layers: list[BaseAugmentation], batchwise: bool = False, **kwargs):
        """Randomly choose one augmentation layer from a list of augmentation layers.

        Args:
            layers (list[BaseAugmentation]): List of augmentation layers to choose from.
            batchwise (bool): If True, apply same layer to all samples in the batch.
        """
        super().__init__(**kwargs)
        self.layers = layers
        self.batchwise = batchwise

    def batch_augment(self, inputs):
        """Apply random layer(s) to the batch"""
        # If batchwise, apply the same layer to all samples in the batch
        if self.batchwise:
            lyr_idx: int = keras.random.randint(
                shape=(), minval=0, maxval=len(self.layers), dtype="int32", seed=self.random_generator
            )
            branch_fns = [lambda x: layer.call(x, training=self.training) for layer in self.layers]
            return keras.ops.switch(lyr_idx, branch_fns, inputs)
        raise NotImplementedError("Batchwise=False is not implemented yet")

    def get_config(self):
        """Serializes the configuration of the layer."""
        config = super().get_config()
        config.update(
            {
                "layers": self.layers,
                "batchwise": self.batchwise,
            }
        )
        return config
