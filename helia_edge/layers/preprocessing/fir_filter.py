"""
# FIR Filter Layer API

This module provides classes to build FIR filter layers.

Classes:
    FirFilter: FIR filter layer

"""

import numpy as np
import numpy.typing as npt
import keras

from .base_augmentation import BaseAugmentation1D
from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.FirFilter")
class FirFilter(BaseAugmentation1D):
    def __init__(
        self,
        b: npt.NDArray[np.float32],
        a: npt.NDArray[np.float32] | None = None,
        forward_backward: bool = False,
        **kwargs,
    ):
        """Apply FIR filter to the input.

        Args:
            b (np.ndarray): Numerator coefficients.
            a (np.ndarray|None): Denominator coefficients. Defaults to None.
            forward_backward (bool): Apply filter forward and backward. Defaults to False.

        Example:

        ```python
        # Create sine wave at 10 Hz with 1000 Hz sampling rate
        t = np.linspace(0, 1, 1000, endpoint=False)
        x = np.sin(2 * np.pi * 10 * t)
        # Add noise at 100 Hz and 2 Hz
        x_noise = x + 0.5 * np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)
        x_noise = x_noise.reshape(-1, 1).astype(np.float32)
        x_noise = keras.ops.convert_to_tensor(x_noise)
        import helia_edge as helia

        # Create bandpass FIR taps
        b = scipy.signal.firwin(101, [5, 15], fs=1000, pass_zero=False)
        lyr = helia.layers.preprocessing.FirFilter(b, forward_backward=True)
        y = lyr(x_noise).numpy().squeeze()
        x_noise = x_noise.numpy().squeeze()
        plt.plot(x, label="Original")
        plt.plot(x_noise, label="Noisy")
        plt.plot(y, label="Filtered")
        plt.legend()
        plt.show()
        ```
        """
        super().__init__(**kwargs)
        b = b.reshape(-1, 1, 1)
        self.b = self.add_weight(
            name="b",
            shape=b.shape,
            trainable=False,
        )
        self.b.assign(b)
        if a is not None:
            a = a.reshape(-1, 1, 1)
            self.a = self.add_weight(
                name="a",
                shape=a.shape,
                trainable=False,
            )
            self.a.assign(a)
        else:
            self.a = None

        self.forward_backward = forward_backward

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Augment a batch of samples during training."""

        samples = inputs[self.SAMPLES]

        # TODO: Handle 'a' coefficients
        if self.a is not None:
            raise NotImplementedError("Denominator coefficients 'a' are not supported yet.")

        outputs = keras.ops.depthwise_conv(samples, self.b, padding="same")

        if self.forward_backward:
            outputs = keras.ops.flip(outputs, axis=self.data_axis)
            outputs = keras.ops.depthwise_conv(outputs, self.b, padding="same")
            outputs = keras.ops.flip(outputs, axis=self.data_axis)
        # END IF
        return outputs

    def get_config(self):
        """Serialize the layer configuration."""
        config = super().get_config()
        config.update(
            {
                "forward_backward": self.forward_backward,
            }
        )
        return config
