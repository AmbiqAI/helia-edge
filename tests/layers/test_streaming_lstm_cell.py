"""StreamingLSTMCell: one LSTM step with explicit state, on every backend."""

import keras
import numpy as np
import pytest

from helia_edge.layers import StreamingLSTMCell, state_input, state_output

UNITS, FEATURES, STEPS = 6, 5, 9


def stepping_model(use_bias=True):
    x = keras.Input((FEATURES,), name="signal")
    h, c = state_input(0, (UNITS,)), state_input(1, (UNITS,))
    h_next, c_next = StreamingLSTMCell(UNITS, use_bias=use_bias, name="lstm")([x, h, c])
    return keras.Model([x, h, c], [state_output(0, h_next), state_output(1, c_next)])


def keras_lstm(model, use_bias=True):
    """A keras.layers.LSTM with the cell's weights, returning the hidden state of every step."""
    x = keras.Input((STEPS, FEATURES))
    lstm = keras.layers.LSTM(UNITS, use_bias=use_bias, return_sequences=True, return_state=True)
    reference = keras.Model(x, lstm(x))
    lstm.set_weights(model.get_layer("lstm").get_weights())
    return reference


def stream(model, xs, h, c):
    hs, cs = [], []
    for t in range(xs.shape[1]):
        h, c = (keras.ops.convert_to_numpy(v) for v in model([xs[:, t], h, c], training=False))
        hs.append(h)
        cs.append(c)
    return np.stack(hs, axis=1), np.stack(cs, axis=1)


@pytest.mark.parametrize("use_bias", [True, False])
def test_stepping_matches_keras_lstm_with_the_same_weights(use_bias):
    keras.utils.set_random_seed(3)
    model = stepping_model(use_bias)
    lstm = model.get_layer("lstm")
    if use_bias:
        lstm.bias.assign(np.random.default_rng(0).normal(size=lstm.bias.shape).astype(np.float32))
    xs = np.random.default_rng(1).normal(size=(2, STEPS, FEATURES)).astype(np.float32)
    zeros = np.zeros((2, UNITS), np.float32)
    hs, cs = stream(model, xs, zeros, zeros)
    want_hs, want_h, want_c = (keras.ops.convert_to_numpy(v) for v in keras_lstm(model, use_bias)(xs))
    np.testing.assert_allclose(hs, want_hs, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(hs[:, -1], want_h, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cs[:, -1], want_c, rtol=1e-5, atol=1e-6)


def test_the_state_carries_and_a_zero_state_restarts_the_sequence():
    keras.utils.set_random_seed(4)
    model = stepping_model()
    xs = np.random.default_rng(2).normal(size=(1, STEPS, FEATURES)).astype(np.float32)
    zeros = np.zeros((1, UNITS), np.float32)
    hs, cs = stream(model, xs, zeros, zeros)
    restarted, _ = stream(model, xs[:, 4:], zeros, zeros)
    carried, _ = stream(model, xs[:, 4:], hs[:, 3], cs[:, 3])
    np.testing.assert_allclose(carried, hs[:, 4:], rtol=1e-6, atol=1e-7)
    assert np.abs(restarted - hs[:, 4:]).max() > 1e-3


def test_model_io_is_named_by_state_pair():
    model = stepping_model()
    assert [t.name for t in model.inputs] == ["signal", "state_in_0", "state_in_1"]
    assert model.output_names == ["state_out_0", "state_out_1"]


def test_weights_have_the_keras_lstm_cell_layout():
    model = stepping_model()
    cell = keras.layers.LSTMCell(UNITS)
    cell.build((None, FEATURES))
    assert [tuple(w.shape) for w in model.get_layer("lstm").weights] == [tuple(w.shape) for w in cell.weights]
    assert [w.name for w in model.get_layer("lstm").weights] == ["kernel", "recurrent_kernel", "bias"]


def test_the_forget_gate_bias_starts_at_one_like_keras():
    model = stepping_model()
    cell = keras.layers.LSTMCell(UNITS)
    cell.build((None, FEATURES))
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(model.get_layer("lstm").bias), keras.ops.convert_to_numpy(cell.bias)
    )


def test_state_shapes_must_match_units():
    x, h, c = keras.Input((FEATURES,)), keras.Input((UNITS + 1,)), keras.Input((UNITS,))
    with pytest.raises(ValueError, match="units=6"):
        StreamingLSTMCell(UNITS)([x, h, c])


def test_saved_model_reloads_with_the_same_outputs(tmp_path):
    keras.utils.set_random_seed(5)
    model = stepping_model()
    path = tmp_path / "cell.keras"
    model.save(path)
    loaded = keras.saving.load_model(path)
    assert isinstance(loaded.get_layer("lstm"), StreamingLSTMCell)
    rng = np.random.default_rng(3)
    feed = [rng.normal(size=(1, n)).astype(np.float32) for n in (FEATURES, UNITS, UNITS)]
    for got, want in zip(loaded(feed), model(feed), strict=True):
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(got), keras.ops.convert_to_numpy(want))
