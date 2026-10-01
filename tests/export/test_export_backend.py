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


def test_a_plugin_exporter_for_this_backend_is_used(monkeypatch):
    import types

    from helia_edge import registry

    backend = keras.backend.backend()
    if f"litert:{backend}" in registry.exporters:
        pytest.skip("The built-in exporter covers this backend")

    def register(module):
        module.exporters.add(f"litert:{backend}", lambda model, spec, calibration: "plugin export")

    monkeypatch.setattr(registry, "_plugins_loaded", False)
    monkeypatch.setattr(
        registry.importlib.metadata,
        "entry_points",
        lambda group: [types.SimpleNamespace(name="p", load=lambda: register)],
    )
    inputs = keras.Input((4,), batch_size=1)
    model = keras.Model(inputs, keras.layers.Dense(2)(inputs))
    try:
        assert export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="concrete")) == "plugin export"
    finally:
        registry.exporters._values.pop(f"litert:{backend}", None)
