"""
# Frequency Mix Style Layer API

This module provides classes to perform frequency mix style augmentation.

Classes:
    FrequencyMixStyle2D: 2D frequency mix style augmentation

"""

import keras

from .base_augmentation import BaseAugmentation2D
from ...utils import parse_factor, helia_export


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
        self.probability, _ = parse_factor([None, probability], min_value=0.0, max_value=1.0, param_name="probability")
        self.alpha, _ = parse_factor([None, alpha], min_value=0.0, max_value=None, param_name="alpha")
        self.epsilon, _ = parse_factor([None, epsilon], min_value=0.0, max_value=None, param_name="epsilon")

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

    def apply_mixstyle(self, x, lmda, perm):
        """Apply mixstyle augmentation

        Args:
            x (tf.Tensor): Input tensor
            lmda (tf.Tensor): Lambda tensor
            perm (tf.Tensor): Permutation tensor

        Returns:
            tf.Tensor: Augmented tensor
        """
        f_mu = keras.ops.mean(x, axis=[2, 3], keepdims=True)
        f_var = keras.ops.var(x, axis=[2, 3], keepdims=True)
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
        if self.training:
            lmda = transforms["lmda"]
            perm = transforms["perm"]
            return keras.ops.cond(
                skip_augment > self.probability, lambda: samples, lambda: self.apply_mixstyle(samples, lmda, perm)
            )
        return samples


# def mixstyle(x, p=0.4, alpha=0.3, eps=1e-6):
#     if keras.random.uniform(shape=()) > p:
#         return x
#     batch_size = x.shape[0]

#     # x axis is NxFxTxC (batch_size, frequency, time, channels)

#     f_mu = keras.ops.mean(x, axis=[2, 3], keepdims=True)
#     f_var = keras.ops.var(x, axis=[2, 3], keepdims=True)
#     f_sig = keras.ops.sqrt(f_var + eps)
#     x_normed = (x - f_mu) / f_sig  # normalize input
#     lmda = keras.random.beta((batch_size, 1, 1, 1), alpha, alpha)
#     perm = keras.random.shuffle(keras.ops.arange(batch_size))
#     f_mu_perm = keras.ops.take(f_mu, perm, axis=0)
#     f_sig_perm = keras.ops.take(f_sig, perm, axis=0)

#     mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)  # generate mixed mean
#     sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)  # generate mixed standard deviation

#     x = x_normed * sig_mix + mu_mix  # denormalize input using the mixed frequency statistics
#     return x
