"""export_model on a backend without an exporter."""

import keras
import pytest

from helia_edge.export import BackendUnavailable, ExportSpec, export_model


def test_non_tensorflow_backend_is_refused_with_guidance():
    if keras.backend.backend() == "tensorflow":
        pytest.skip("Checks the refusal on other backends")
    inputs = keras.Input((4,), batch_size=1)
    model = keras.Model(inputs, keras.layers.Dense(2)(inputs))
    spec = ExportSpec(precision="fp32", io_dtype="float32", mode="concrete")
    with pytest.raises(BackendUnavailable, match="KERAS_BACKEND=tensorflow"):
        export_model(model, spec)
