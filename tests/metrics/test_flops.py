import pytest
import keras

from helia_edge.metrics.flops import get_flops


class _DummyModel:
    input_shape = (None, 4)

    def call(self, x):
        return x


def test_get_flops_rejects_non_tensorflow_backend(monkeypatch):
    monkeypatch.setattr("keras.backend.backend", lambda: "torch")

    with pytest.raises(ValueError, match="Only tensorflow backend is supported"):
        get_flops(_DummyModel())


def test_get_flops_positive_for_small_model():
    if keras.backend.backend() != "tensorflow":
        pytest.skip("get_flops currently supports TensorFlow backend only.")

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(2),
        ]
    )

    flops = get_flops(model, batch_size=1)
    assert flops > 0.0
