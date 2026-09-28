"""
# Frequency Mix Style Layer API

This module provides classes to perform frequency mix style augmentation.

Classes:
    FrequencyMixStyle2D: 2D frequency mix style augmentation

"""

import keras

from .base_augmentation import BaseAugmentation2D
from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.FrequencyMixStyle2D")
class FrequencyMixStyle2D(BaseAugmentation2D):
    probability: float
    alpha: float
    epsilon: float

    def __init__(
        self,
        probability: float = 0.5,
        alpha: float = 1.0,
        epsilon: float = 1e-6,
        **kwargs,
    ):
        """Apply frequency mix style augmentation to the 2D input.

        Args:
            probability (float): Probability of applying the augmentation.
            alpha (float): Mixup alpha value.
            epsilon (float): Epsilon value for numerical stability.

        Example:

        ```python
            x = np.random.rand(4, 4, 3)
            lyr = FrequencyMixStyle2D(probability=1.0, alpha=1.0)
            y = lyr(x, training=True)
        ```
        """

        super().__init__(**kwargs)
        if not 0 <= probability <= 1 or alpha <= 0 or epsilon <= 0:
            raise ValueError("probability must be in [0,1]; alpha and epsilon must be positive")
        self.probability = probability
        self.alpha = alpha
        self.epsilon = epsilon

    def get_random_transformations(self, input_shape: tuple[int, int, int]) -> dict:
        """Generate noise distortion tensor

        Args:
            input_shape (tuple[int, ...]): Input shape.

        Returns:
            dict: Dictionary containing the noise tensor.
        """
        batch_size = input_shape[0]
        skip_augment = keras.random.uniform(
            shape=(), minval=0.0, maxval=1.0, dtype="float32", seed=self.random_generator
        )
        lmda = keras.random.beta(
            shape=(batch_size, 1, 1, 1), alpha=self.alpha, beta=self.alpha, seed=self.random_generator
        )
        perm = keras.random.shuffle(keras.ops.arange(batch_size), seed=self.random_generator)
        return {"lmda": lmda, "perm": perm, "skip_augment": skip_augment}

    def apply_mixstyle(
        self, x: keras.KerasTensor, lmda: keras.KerasTensor, perm: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Apply mixstyle augmentation

        Args:
            x (tf.Tensor): Input tensor
            lmda (tf.Tensor): Lambda tensor
            perm (tf.Tensor): Permutation tensor

        Returns:
            tf.Tensor: Augmented tensor
        """
        f_mu = keras.ops.mean(x, axis=[1, 3] if self.data_format == "channels_first" else [2, 3], keepdims=True)
        f_var = keras.ops.var(x, axis=[1, 3] if self.data_format == "channels_first" else [2, 3], keepdims=True)
        f_sig = keras.ops.sqrt(f_var + self.epsilon)

        x_normed = (x - f_mu) / f_sig
        f_mu_perm = keras.ops.take(f_mu, perm, axis=0)
        f_sig_perm = keras.ops.take(f_sig, perm, axis=0)
        x_perm = keras.ops.take(x_normed, perm, axis=0)
        x_mix = lmda * x_normed + (1 - lmda) * x_perm
        x_mix = x_mix * f_sig_perm + f_mu_perm
        return x_mix

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Augment samples

        Args:
            inputs (tf.Tensor): Input tensor

        Returns:
            tf.Tensor: Augmented tensor
        """

        samples = inputs[self.SAMPLES]
        transforms = inputs[self.TRANSFORMS]
        skip_augment = transforms["skip_augment"]
        lmda = transforms["lmda"]
        perm = transforms["perm"]
        return keras.ops.cond(
            skip_augment > self.probability, lambda: samples, lambda: self.apply_mixstyle(samples, lmda, perm)
        )
        return samples

    def get_config(self):
        return {**super().get_config(), "probability": self.probability, "alpha": self.alpha, "epsilon": self.epsilon}
