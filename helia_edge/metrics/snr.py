"""
# SNR Metrics API

Signal-to-Noise Ratio (SNR) metric where y_true: signal, y_pred: signal + noise

Classes:
    Snr: Computes the Signal-to-Noise Ratio (SNR) in dB.
"""

import keras

from ..utils import helia_export


@helia_export(path="helia_edge.metrics.Snr")
class Snr(keras.Metric):
    def __init__(self, name="snr", **kwargs):
        """Signal-to-Noise Ratio (SNR) metric where
            y_true: signal
            y_pred: signal + noise

        Args:
            name (str, optional): Name of the metric. Defaults to 'snr'.

        Example:

        ```python
        # Create 4-second sine wave w/ freq=4, amplitude=1, Fs=1000Hz
        t = np.linspace(0, 4, 4 * 1000, endpoint=False)
        x = np.sin(2 * np.pi * 4 * t)
        # Add random noise with amplitude 0.1
        noise = np.random.normal(0, 0.1, len(t))
        y = x + noise
        import helia_edge as helia

        snr = helia.metrics.Snr()
        snr.update_state(x, y)
        print(snr.result())
        ```
        """
        super().__init__(name=name, **kwargs)
        self.num = self.add_variable(shape=(), initializer="zeros", name="num")
        self.den = self.add_variable(shape=(), initializer="zeros", name="den")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = keras.ops.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = keras.ops.convert_to_tensor(y_pred, dtype=self.dtype)

        # Flatten the input if its rank > 1.
        if len(y_pred.shape) > 1:
            y_pred = keras.ops.reshape(y_pred, [-1])

        if len(y_true.shape) > 1:
            y_true = keras.ops.reshape(y_true, [-1])

        # Numerator is the sum of squares of the true signal
        num = keras.ops.sum(keras.ops.square(y_true))

        # Denominator is the sum of squares of the noise
        den = keras.ops.sum(keras.ops.square(y_pred - y_true))

        self.num.assign_add(self.num + num)
        self.den.assign_add(self.den + den)

    def result(self):
        """Computes the SNR in dB."""
        ratio = keras.ops.divide(self.num, self.den + keras.backend.epsilon())
        ratio = keras.ops.convert_to_tensor(ratio, dtype=self.dtype)
        snr = 10 * keras.ops.log10(ratio + keras.backend.epsilon())
        return snr  # in dB

    def get_config(self):
        return super().get_config()

    def reset_state(self):
        for v in self.variables:
            v.assign(keras.ops.zeros(v.shape, dtype=v.dtype))
