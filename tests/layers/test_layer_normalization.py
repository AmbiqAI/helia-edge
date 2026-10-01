"""helia_edge.layers.LayerNormalization normalizes any axes on every backend, with the Keras weights and config."""

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from helia_edge.layers import LayerNormalization  # noqa: E402

CASES = [
    ((2, 1, 16, 4), (1, 2)),  # TCN and UNet: spatial axes, not trailing
    ((2, 1, 16, 4), 2),  # UNext on (1, time, channels) rows
    ((2, 8, 1, 4), 1),
    ((2, 6, 5, 4), (1, 2)),
    ((2, 1, 16, 4), -1),  # trailing: the Keras path on every backend
    ((2, 16, 4), (1, 2)),
]


def reference(x, axes, gamma, beta, epsilon):
    axes = tuple(a % x.ndim for a in np.atleast_1d(axes))
    mean = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    shape = [x.shape[a] if a in axes else 1 for a in range(x.ndim)]
    return (x - mean) / np.sqrt(var + epsilon) * gamma.reshape(shape) + beta.reshape(shape)


@pytest.mark.parametrize("shape,axis", CASES)
def test_matches_the_numpy_reference(shape, axis):
    rng = np.random.default_rng(0)
    x = rng.normal(size=shape).astype(np.float32)
    layer = LayerNormalization(axis=axis)
    layer.build(shape)
    gamma = rng.normal(size=layer.gamma.shape).astype(np.float32)
    beta = rng.normal(size=layer.beta.shape).astype(np.float32)
    layer.set_weights([gamma, beta])
    out = keras.ops.convert_to_numpy(layer(x))
    np.testing.assert_allclose(out, reference(x, axis, gamma, beta, layer.epsilon), rtol=1e-5, atol=1e-5)


def test_weights_and_config_match_the_keras_layer(tmp_path):
    ours, theirs = LayerNormalization(axis=(1, 2)), keras.layers.LayerNormalization(axis=(1, 2))
    ours.build((1, 1, 16, 4))
    theirs.build((1, 1, 16, 4))
    assert [w.shape for w in ours.weights] == [w.shape for w in theirs.weights]
    assert {k: v for k, v in ours.get_config().items() if k != "name"} == {
        k: v for k, v in theirs.get_config().items() if k != "name"
    }
    inputs = keras.Input((1, 16, 4), batch_size=1)
    model = keras.Model(inputs, LayerNormalization(axis=(1, 2))(inputs))
    model.save(tmp_path / "m.keras")
    reloaded = keras.models.load_model(tmp_path / "m.keras")
    assert type(reloaded.layers[-1]) is LayerNormalization


@pytest.mark.parametrize("axis,expected", [(-1, keras.layers.LayerNormalization), ((1, 2), LayerNormalization)])
def test_the_helper_keeps_the_keras_class_for_the_last_axis(axis, expected):
    from helia_edge.layers import layer_normalization

    inputs = keras.Input((1, 16, 4), batch_size=1)
    layer = keras.Model(inputs, layer_normalization(name="n", axis=axis)(inputs)).layers[-1]
    assert type(layer) is expected
