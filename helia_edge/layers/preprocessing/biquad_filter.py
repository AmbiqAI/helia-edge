"""
# Biquad Filter Layer API

This module provides classes to build biquad filter layers.

Functions:
    get_butter_sos: Compute biquad filter coefficients as SOS

Classes:
    CascadedBiquadFilter: Cascaded biquad filter layer

"""

import keras
import scipy.signal
import numpy.typing as npt
from functools import lru_cache

from .base_augmentation import BaseAugmentation1D
from ...utils import helia_export


@lru_cache(maxsize=128)
def get_butter_sos(
    lowcut: float | None = None,
    highcut: float | None = None,
    sample_rate: float = 1000,
    order: int = 3,
) -> npt.NDArray:
    """Compute biquad filter coefficients as SOS. This function caches.
    For highpass, lowcut is required and highcut is ignored.
    For lowpass, highcut is required and lowcut is ignored.
    For bandpass, both lowcut and highcut are required.

    Args:
        lowcut (float|None): Lower cutoff in Hz. Defaults to None.
        highcut (float|None): Upper cutoff in Hz. Defaults to None.
        sample_rate (float): Sampling rate in Hz. Defaults to 1000 Hz.
        order (int, optional): Filter order. Defaults to 3.

    Returns:
        npt.NDArray: SOS
    """
    nyq = sample_rate / 2
    if lowcut is not None and highcut is not None:
        freqs = [lowcut / nyq, highcut / nyq]
        btype = "bandpass"
    elif lowcut is not None:
        freqs = lowcut / nyq
        btype = "highpass"
    elif highcut is not None:
        freqs = highcut / nyq
        btype = "lowpass"
    else:
        raise ValueError("At least one of lowcut or highcut must be specified")
    sos = scipy.signal.butter(order, freqs, btype=btype, output="sos")
    return sos


@helia_export(path="helia_edge.layers.preprocessing.CascadedBiquadFilter")
class CascadedBiquadFilter(BaseAugmentation1D):
    def __init__(
        self,
        lowcut: float | None = None,
        highcut: float | None = None,
        sample_rate: float = 1000,
        order: int = 3,
        forward_backward: bool = False,
        **kwargs,
    ):
        """Implements a 2nd order cascaded biquad filter using direct form 1 structure.

        See [here](https://en.wikipedia.org/wiki/Digital_biquad_filter) for more information
        on the direct form 1 structure.

        Args:
            lowcut (float|None): Lower cutoff in Hz. Defaults to None.
            highcut (float|None): Upper cutoff in Hz. Defaults to None.
            sample_rate (float): Sampling rate in Hz. Defaults to 1000 Hz.
            order (int, optional): Filter order. Defaults to 3.
            forward_backward (bool): Apply filter forward and backward.

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

        # Create bandpass filter
        lyr = helia.layers.preprocessing.CascadedBiquadFilter(
            lowcut=5,
            highcut=15,
            sample_rate=1000,
            forward_backward=True,
        )
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

        self.lowcut = lowcut
        self.highcut = highcut
        self.sample_rate = sample_rate
        self.order = order
        self.forward_backward = forward_backward

        sos = get_butter_sos(lowcut, highcut, sample_rate, order)
        # Normalize by a0 to use the canonical recurrence:
        # y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        sos = sos / sos[:, [3]]

        # Canonical SOS layout: [b0, b1, b2, a0(=1), a1, a2]
        self.sos = self.add_weight(
            name="sos",
            shape=sos.shape,
            trainable=False,
        )
        self.sos.assign(sos)
        self.num_stages = keras.ops.shape(self.sos)[0]

    def _apply_sos(self, i, sample: keras.KerasTensor) -> keras.KerasTensor:
        """Applies a single section to the input sample.

        Equation:
           y[n] = b0 * x[n] + b1 * x[n-1] + b2 * x[n-2] - a1 * y[n-1] - a2 * y[n-2]

        Args:
            i (int): Index of the second order section.
            sample (keras.KerasTensor): Input sample with shape (duration, channels)

        Returns:
            keras.KerasTensor: Output sample with shape (duration, channels)
        """
        # Inputs must be channels_last
        duration_size = keras.ops.shape(sample)[0]
        ch_size = keras.ops.shape(sample)[1]
        coeffs = keras.ops.squeeze(keras.ops.slice(self.sos, start_indices=[i, 0], shape=[1, 6]), axis=0)
        b0, b1, b2, _a0, a1, a2 = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5]

        # Previous implementation (depthwise_conv + feedback pass) was kept for reference but
        # replaced because it did not exactly match causal SOS filtering at sequence boundaries.
        # y = keras.ops.depthwise_conv(...)

        zeros_ch = keras.ops.zeros((ch_size,), dtype=sample.dtype)
        output = keras.ops.zeros((duration_size, ch_size), dtype=sample.dtype)

        def tstep_fn(t, state):
            """Applies single time step using causal SOS recurrence."""
            y_seq, x1, x2, y1, y2 = state
            x0 = keras.ops.squeeze(keras.ops.slice(sample, start_indices=[t, 0], shape=[1, ch_size]), axis=0)
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            y_seq = keras.ops.slice_update(
                y_seq,
                start_indices=[t, 0],
                updates=keras.ops.expand_dims(y0, axis=0),
            )
            return y_seq, x0, x1, y0, y1

        output, _, _, _, _ = keras.ops.fori_loop(
            lower=0,
            upper=duration_size,
            body_fun=tstep_fn,
            init_val=(output, zeros_ch, zeros_ch, zeros_ch, zeros_ch),
        )
        return output

    def augment_sample(self, inputs) -> keras.KerasTensor:
        """Applies the cascaded biquad filter to the input samples."""
        samples = inputs[self.SAMPLES]
        # inputs have shape (time, channels)

        # Force to be channels_last
        if self.data_format == "channels_first":
            samples = keras.ops.transpose(samples, axes=[1, 0])

        # Iterate across second order sections
        samples = keras.ops.fori_loop(lower=0, upper=self.num_stages, body_fun=self._apply_sos, init_val=samples)

        if self.forward_backward:
            samples = keras.ops.flip(samples, axis=0)
            samples = keras.ops.fori_loop(lower=0, upper=self.num_stages, body_fun=self._apply_sos, init_val=samples)
            samples = keras.ops.flip(samples, axis=0)
        # END IF

        # Undo the transpose if needed
        if self.data_format == "channels_first":
            samples = keras.ops.transpose(samples, axes=[1, 0])

        return samples

    def get_config(self):
        """Serialize the configuration."""
        config = super().get_config()
        config.update(
            {
                "lowcut": self.lowcut,
                "highcut": self.highcut,
                "sample_rate": self.sample_rate,
                "order": self.order,
                "forward_backward": self.forward_backward,
            }
        )
        return config


# import numpy as np
# class CascadedBiquadFilterNumpy:

#     def __init__(self, sos):
#         """Implements a 2nd order cascaded biquad filter using the provided second order sections (sos) matrix."""
#         # These are the filter coefficients arranged as 2D tensor (n_sections x 6)
#         # We remap them [b0, b1, b2, a0, a1, a2] but mapped as [b0, b1, b2, -a1, -a2]
#         self.sos = sos[:, [0, 1, 2, 4, 5]] * [1, 1, 1, -1, -1]

#     def call(self, inputs):
#         # inputs has shape (batch, time, channels)
#         batches = inputs.shape[0]
#         time = inputs.shape[1]
#         chs = inputs.shape[2]
#         num_stages = self.sos.shape[0]

#         state = np.zeros((batches, 2, num_stages, chs), dtype=inputs.dtype)
#         outputs = np.zeros((batches, time, chs), dtype=inputs.dtype)

#         for b in range(batches):
#             for t in range(time):
#                 x = inputs[b, t]
#                 y = np.zeros(chs)
#                 for s in range(num_stages):
#                     b0, b1, b2, a1n, a2n = self.sos[s]
#                     y = b0*x + state[b, 0, s]
#                     state[b, 0, s] = b1*x + a1n*y + state[b, 1, s]
#                     state[b, 1, s] = b2*x + a2n*y
#                     x = y
#                 # END FOR
#                 outputs[b, t] = y
#             # END FOR
#         # END FOR
#         return outputs
#     # END DEF
