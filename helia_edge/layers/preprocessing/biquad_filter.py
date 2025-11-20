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

from .base_augmentation import BaseAugmentation1D
from ...utils import helia_export


def get_butter_sos(
    lowcut: float | None = None,
    highcut: float | None = None,
    sample_rate: float = 1000,
    order: int = 3,
) -> npt.NDArray:
    """Compute biquad filter coefficients as SOS. This function caches.
    For lowpass, lowcut is required and highcut is ignored.
    For highpass, highcut is required and lowcut is ignored.
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

        sos = get_butter_sos(lowcut, highcut, sample_rate, order)

        # These are the second order coefficients arranged as 2D tensor (n_sections x 6)
        # We remap each section from [b0, b1, b2, a0, a1, a2] to [b0, b1, b2, -a2, -a1, a0]
        sos = sos[:, [0, 1, 2, 5, 4, 3]] * [1, 1, 1, -1, -1, 1]
        self.sos = self.add_weight(
            name="sos",
            shape=sos.shape,
            trainable=False,
        )
        self.sos.assign(sos)
        self.num_stages = keras.ops.shape(self.sos)[0]
        self.forward_backward = forward_backward

    def _apply_sos(self, i, sample: keras.KerasTensor) -> keras.KerasTensor:
        """Applies a single section to the input sample.

        Equation:
           y[n] = b0 * x[n] + b1 * x[n-1] + b2 * x[n-2] + a1 * y[n-1] + a2 * y[n-2]

        Args:
            i (int): Index of the second order section.
            sample (keras.KerasTensor): Input sample with shape (duration, channels)

        Returns:
            keras.KerasTensor: Output sample with shape (duration, channels)
        """
        # Inputs must be channels_last
        duration_size = keras.ops.shape(sample)[0]
        ch_size = keras.ops.shape(sample)[1]

        # Step 1: Convolve input `x` with coefficients [b0, b1, b2]
        # y[n] = b0 * x[n] + b1 * x[n-1] + b2 * x[n-2]

        # Taps will be [b0, b1, b2]
        taps = keras.ops.slice(self.sos, start_indices=[i, 0], shape=[1, 3])
        taps = keras.ops.tile(keras.ops.reshape(taps, (3, 1, 1)), (1, ch_size, 1))

        # keras.ops.depthwise_conv appears to require batch dimension
        y = keras.ops.reshape(sample, (1, duration_size, ch_size))
        y = keras.ops.depthwise_conv(inputs=y, kernel=taps, padding="same", data_format="channels_last")
        y = keras.ops.squeeze(y, axis=0)

        # Step 2: Apply feedback to the output `y`
        # y[n] = a0 * y[n] + a1 * y[n-1] + a2 * y[n-2]

        # Pad y with zeros at the beginning
        y = keras.ops.pad(y, pad_width=[[2, 0], [0, 0]], constant_values=0)

        # Taps will be [a0, a1, a2]
        taps = keras.ops.slice(self.sos, start_indices=[i, 3], shape=[1, 3])
        taps = keras.ops.squeeze(taps, axis=0)

        indices = keras.ops.arange(0, duration_size + 2)
        indices = keras.ops.reshape(indices, (-1, 1))
        indices = keras.ops.tile(indices, (1, ch_size))

        def tstep_fn(t, yl):
            """Applies single time step"""
            yy = keras.ops.slice(yl, start_indices=[t - 2, 0], shape=[3, ch_size])
            yy = keras.ops.transpose(yy, axes=[1, 0])  # (ch_size, 3)
            yy = keras.ops.dot(yy, taps)  # (ch_size, 3) x (3) = (ch_size)
            yl = keras.ops.where(condition=indices == t, x1=yy, x2=yl)
            return yl

        # Iterate over time steps
        y = keras.ops.fori_loop(lower=2, upper=duration_size + 2, body_fun=tstep_fn, init_val=y)

        # Remove the padding
        y = keras.ops.slice(y, start_indices=[2, 0], shape=[duration_size, ch_size])

        return y

    def augment_sample(self, inputs) -> keras.KerasTensor:
        """Applies the cascaded biquad filter to the input samples."""
        samples = inputs[self.SAMPLES]
        # inputs have shape (time, channels)

        # Force to be channels_last
        if self.data_format == "channels_first":
            samples = keras.ops.transpose(samples, perm=[1, 0])

        # Iterate across second order sections
        samples = keras.ops.fori_loop(lower=0, upper=self.num_stages, body_fun=self._apply_sos, init_val=samples)

        if self.forward_backward:
            samples = keras.ops.flip(samples, axis=self.data_axis)
            samples = keras.ops.fori_loop(lower=0, upper=self.num_stages, body_fun=self._apply_sos, init_val=samples)
            samples = keras.ops.flip(samples, axis=self.data_axis)
        # END IF

        # Undo the transpose if needed
        if self.data_format == "channels_first":
            samples = keras.ops.transpose(samples, axes=[1, 0])

        return samples


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
