"""
# Random Augmentation Pipeline Layer API

This module provides classes to build random augmentation pipeline layers.

Classes:
    RandomAugmentation1DPipeline: Random augmentation 1D pipeline
    RandomAugmentation2DPipeline: Random augmentation 2D pipeline

"""

import keras
from .base_augmentation import BaseAugmentation1D
from .random_choice import RandomChoice
from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.RandomAugmentation1DPipeline")
class RandomAugmentation1DPipeline(BaseAugmentation1D):
    layers: list[BaseAugmentation1D]
    augmentations_per_sample: int
    rate: float

    def __init__(
        self,
        layers: list[BaseAugmentation1D],
        augmentations_per_sample: int = 1,
        rate: float = 1.0,
        batchwise: bool = False,
        force_training: bool = False,
        **kwargs,
    ):
        """Apply N random augmentations from a list of augmentation layers to each sample.

        Args:
            layers (list[BaseAugmentation1D]): List of augmentation layers to choose from.
            augmentations_per_sample (int): Number of augmentations to apply to each sample.
            rate (float): Probability of applying the augmentation pipeline.
            batchwise (bool): If True, apply same layer to all samples in the batch.
            force_training (bool, optional): Force training mode. Defaults to False.
        """
        super().__init__(**kwargs)
        self.layers = layers
        self.augmentations_per_sample = augmentations_per_sample
        self.rate = rate
        self.batchwise = batchwise
        kwargs.update({"name": "random_choice"})
        self._random_choice = RandomChoice(layers=layers, batchwise=batchwise, **kwargs)
        self.force_training = force_training
        if not self.layers:
            raise ValueError("At least one layer must be provided.")

    def apply_random_choice(self, inputs):
        skip_augment = keras.random.uniform(
            shape=(), minval=0.0, maxval=1.0, dtype="float32", seed=self.random_generator
        )
        return keras.ops.cond(
            skip_augment > self.rate,
            lambda: inputs,
            lambda: self._random_choice.batch_augment(inputs),
        )

    def batch_augment(self, inputs):
        """Apply N random augmentations to each"""
        return keras.ops.fori_loop(
            lower=0,
            upper=self.augmentations_per_sample,
            body_fun=lambda _, x: self.apply_random_choice(x),
            init_val=inputs,
        )

    def call(self, inputs, training: bool = True, **kwargs):
        self._random_choice.training = training or self.force_training
        super().call(inputs, training=training or self.force_training, **kwargs)

    def get_config(self):
        """Serializes the configuration of the layer."""
        config = super().get_config()
        config.update(
            {
                "layers": [lyr.get_config() for lyr in self.layers],
                "augmentations_per_sample": self.augmentations_per_sample,
                "rate": self.rate,
                "batchwise": self.batchwise,
                "force_training": self.force_training,
            }
        )
        return config


@helia_export(path="helia_edge.layers.preprocessing.RandomAugmentation2DPipeline")
class RandomAugmentation2DPipeline(RandomAugmentation1DPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._random_choice = RandomChoice(layers=self.layers, batchwise=self.batchwise, **kwargs)

    def apply_random_choice(self, inputs):
        skip_augment = keras.random.uniform(
            shape=(), minval=0.0, maxval=1.0, dtype="float32", seed=self.random_generator
        )
        return keras.ops.cond(
            skip_augment > self.rate,
            lambda: inputs,
            lambda: self._random_choice.batch_augment(inputs),
        )

    def batch_augment(self, inputs):
        """Apply N random augmentations to each"""
        return keras.ops.fori_loop(
            lower=0,
            upper=self.augmentations_per_sample,
            body_fun=lambda _, x: self.apply_random_choice(x),
            init_val=inputs,
        )

    def call(self, inputs, training: bool = True, **kwargs):
        self._random_choice.training = training or self.force_training
        super().call(inputs, training=training or self.force_training, **kwargs)

    def get_config(self):
        """Serializes the configuration of the layer."""
        config = super().get_config()
        config.update(
            {
                "layers": [lyr.get_config() for lyr in self.layers],
                "augmentations_per_sample": self.augmentations_per_sample,
                "rate": self.rate,
                "batchwise": self.batchwise,
                "force_training": self.force_training,
            }
        )
        return config
