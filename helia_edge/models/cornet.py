"""CorNET heart-rate regressor from a wrist PPG window.

Rebuilt from Biswas et al., "CorNET: Deep Learning Framework for PPG-Based
Heart Rate Estimation and Biometric Identification in Ambulant Environment",
IEEE TBioCAS 13(2), 2019 (Sec. III, Fig. 6, Table III). The reference input
is 8 s of band-passed, z-scored PPG at 125 Hz (1000 x 1). No code or weights
are published, so constructed models are untrained.
"""

import keras

from .cornet_params import CorNetParams

keras.saving.register_keras_serializable(package="helia_edge")(CorNetParams)


class CorNetModel:
    """Build a standard Keras Functional model from typed architecture config."""

    @staticmethod
    def model_from_params(inputs: keras.KerasTensor, params: CorNetParams, *, unroll: bool = False) -> keras.Model:
        """Construct an untrained regressor with one linear output.

        Each convolution stage is Conv1D (valid, stride 1), batch normalization,
        ReLU, max pooling and dropout, following Fig. 6; the paper's text
        places batch normalization after ReLU instead. Stride and padding are
        not stated and are inferred from Table III's MAC counts. Every LSTM but
        the last returns its sequence. ``unroll`` builds the LSTMs as
        per-timestep operations instead of a loop; weights are identical.
        """
        if not isinstance(params, CorNetParams):
            raise TypeError("params must be CorNetParams; use from_config for mappings")
        if not keras.backend.is_keras_tensor(inputs) or len(inputs.shape) != 3:
            raise ValueError("inputs must be a rank-3 (batch, time, channels) Keras tensor")
        if inputs.shape[1] is None or inputs.shape[2] is None:
            raise ValueError("inputs need a known time length and channel count")

        x = inputs
        for stage in range(params.conv_stages):
            name = f"conv{stage}"
            if x.shape[1] < params.kernel_size:
                raise ValueError("input window is too short for the convolution stages")
            x = keras.layers.Conv1D(params.filters, params.kernel_size, name=f"{name}_conv")(x)
            x = keras.layers.BatchNormalization(name=f"{name}_bn")(x)
            x = keras.layers.Activation("relu", name=f"{name}_relu")(x)
            x = keras.layers.MaxPooling1D(params.pool_size, name=f"{name}_pool")(x)
            if params.dropout:
                x = keras.layers.Dropout(params.dropout, name=f"{name}_dropout")(x)
        if x.shape[1] < 1:
            raise ValueError("input window is too short for the convolution stages")
        for index in range(params.lstm_layers):
            last = index == params.lstm_layers - 1
            x = keras.layers.LSTM(
                params.lstm_units,
                recurrent_activation=params.recurrent_activation,
                return_sequences=not last,
                unroll=unroll,
                name=f"lstm{index}",
            )(x)
        outputs = keras.layers.Dense(1, name="hr")(x)
        return keras.Model(inputs=inputs, outputs=outputs, name=params.name)
