"""CorNET heart-rate regressor from a wrist PPG window.

Rebuilt from Biswas et al., "CorNET: Deep Learning Framework for PPG-Based
Heart Rate Estimation and Biometric Identification in Ambulant Environment",
IEEE TBioCAS 13(2), 2019 (Sec. III, Fig. 6, Table III). The reference input
is 8 s of band-passed, z-scored PPG at 125 Hz (1000 x 1). No code or weights
are published, so constructed models are untrained.
"""

import keras

from .cornet_params import CorNetParams


def build(
    params: CorNetParams, input_shape: tuple[int | None, ...], *, batch_size: int | None = None, name: str | None = None
) -> keras.Model:
    """Construct an untrained CorNET regressor with one linear output.

    Each convolution stage is Conv1D (valid, stride 1), batch normalization,
    ReLU, max pooling and dropout, following Fig. 6; the paper's text
    places batch normalization after ReLU instead. Stride and padding are
    not stated and are inferred from Table III's MAC counts. Every LSTM but
    the last returns its sequence.

    Args:
        params (CorNetParams): Model parameters.
        input_shape (tuple[int | None, ...]): ``(time, channels)``, both known.
        batch_size (int | None): Static batch size; None for a dynamic batch.
        name (str | None): Model name; the family when None.

    Returns:
        keras.Model: The model, named ``cornet`` unless ``name`` is given.
    """
    if len(input_shape) != 2:
        raise ValueError("input_shape must be (time, channels)")
    if input_shape[0] is None or input_shape[1] is None:
        raise ValueError("inputs need a known time length and channel count")
    inputs = keras.Input(shape=input_shape, batch_size=batch_size, name="inputs")

    x = inputs
    for stage in range(params.conv_stages):
        prefix = f"conv{stage}"
        if x.shape[1] < params.kernel_size:
            raise ValueError("input window is too short for the convolution stages")
        x = keras.layers.Conv1D(params.filters, params.kernel_size, name=f"{prefix}_conv")(x)
        x = keras.layers.BatchNormalization(name=f"{prefix}_bn")(x)
        x = keras.layers.Activation("relu", name=f"{prefix}_relu")(x)
        x = keras.layers.MaxPooling1D(params.pool_size, name=f"{prefix}_pool")(x)
        if params.dropout:
            x = keras.layers.Dropout(params.dropout, name=f"{prefix}_dropout")(x)
    if x.shape[1] < 1:
        raise ValueError("input window is too short for the convolution stages")
    for index in range(params.lstm_layers):
        last = index == params.lstm_layers - 1
        x = keras.layers.LSTM(
            params.lstm_units,
            recurrent_activation=params.recurrent_activation,
            return_sequences=not last,
            unroll=params.unroll,
            name=f"lstm{index}",
        )(x)
    outputs = keras.layers.Dense(1, name="hr")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=name or params.family)
