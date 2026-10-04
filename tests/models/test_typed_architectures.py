"""Public typed configs preserve model behavior and caller-owned initialization."""

import json

import keras
import numpy as np
import pytest
from pydantic import ValidationError

from helia_edge.models import (
    MlperfTinyParams,
    ModelSpec,
    TcnParams,
    build,
    compact_tcn_params,
)
from helia_edge.models.mlperf_tiny import build as mlperf_build
from helia_edge.models.tcn import build as tcn_build

ARCHITECTURES = ("kws", "vww", "resnet", "ad")


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_mlperf_spec_round_trips_builds_and_serializes(architecture, tmp_path):
    spec = ModelSpec(params=MlperfTinyParams(architecture=architecture))
    assert ModelSpec.model_validate_json(spec.model_dump_json()) == spec
    keras.backend.clear_session()
    keras.utils.set_random_seed(123)
    original = mlperf_build(spec.params)
    keras.backend.clear_session()
    keras.utils.set_random_seed(123)
    adapted = build(spec)
    assert original.to_json() == adapted.to_json() and adapted.name == "mlperf_tiny"
    for left, right in zip(original.get_weights(), adapted.get_weights(), strict=True):
        np.testing.assert_array_equal(left, right)
    x = np.ones((1, *adapted.input_shape[1:]), np.float32)
    expected = keras.ops.convert_to_numpy(adapted(x, training=False))
    path = tmp_path / "model.keras"
    adapted.save(path)
    restored = keras.models.load_model(path)
    np.testing.assert_array_equal(expected, keras.ops.convert_to_numpy(restored(x, training=False)))


@pytest.mark.parametrize(
    "config", [{"architecture": "resnet18"}, {"architecture": "kws", "seed": 1}, {"architecture": "vww", "scale": 0.5}]
)
def test_fixed_family_rejects_unsupported_fields(config):
    with pytest.raises(ValidationError):
        MlperfTinyParams.model_validate(config)


def test_mlperf_input_shape_is_fixed_and_name_is_an_option():
    params = MlperfTinyParams(architecture="ad")
    with pytest.raises(ValidationError):
        params.architecture = "kws"
    assert mlperf_build(params, name="custom_ad").name == "custom_ad"
    assert build(ModelSpec(params=params, input_shape=(640,))).input_shape == (None, 640)
    with pytest.raises(ValueError, match="takes input shape"):
        build(ModelSpec(params=params, input_shape=(320,)))


@pytest.mark.parametrize("filters", [8, 16])
def test_tcn_preset_roundtrips_and_builds_the_same_model_both_ways(filters, tmp_path):
    params = compact_tcn_params(filters=filters, num_classes=2)
    spec = ModelSpec(params=params, input_shape=(240, 14))
    assert ModelSpec.model_validate_json(spec.model_dump_json()) == spec
    assert [b.dilation for b in params.blocks] == [(1, 1), (1, 2), (1, 4), (1, 8)]
    assert all(b.filters == filters and b.se_ratio == 4 for b in params.blocks)
    keras.backend.clear_session()
    keras.utils.set_random_seed(20260925)
    typed = build(spec, batch_size=1)
    keras.backend.clear_session()
    keras.utils.set_random_seed(20260925)
    direct = tcn_build(params, (240, 14), batch_size=1)
    assert typed.name == direct.name == "tcn"
    assert typed.to_json() == direct.to_json()
    for left, right in zip(typed.get_weights(), direct.get_weights(), strict=True):
        np.testing.assert_array_equal(left, right)
    x = np.ones((1, 240, 14), np.float32)
    expected = keras.ops.convert_to_numpy(typed(x, training=False))
    path = tmp_path / "tcn.keras"
    typed.save(path)
    restored = keras.models.load_model(path)
    np.testing.assert_array_equal(expected, keras.ops.convert_to_numpy(restored(x, training=False)))


@pytest.mark.parametrize("nested", [False, True])
def test_tcn_params_reject_unknown_fields(nested):
    config = compact_tcn_params().model_dump(mode="json")
    target = config["blocks"][0] if nested else config
    target["typo"] = 1
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TcnParams.model_validate(config)


@pytest.mark.parametrize("filters", [True, 7, 8.5])
def test_preset_rejects_invalid_width(filters):
    with pytest.raises(ValueError):
        compact_tcn_params(filters=filters)


def test_public_factories_do_not_reset_seed_or_session(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("factory reset caller-owned global state")

    monkeypatch.setattr(keras.utils, "set_random_seed", forbidden)
    monkeypatch.setattr(keras.backend, "clear_session", forbidden)
    for architecture in ARCHITECTURES:
        mlperf_build(MlperfTinyParams(architecture=architecture))
    spec = ModelSpec(params=compact_tcn_params(num_classes=4), input_shape=(32, 3))
    first, second = build(spec), build(spec)
    assert first.output_shape == second.output_shape == (None, 32, 4)
    assert any(not np.array_equal(a, b) for a, b in zip(first.get_weights(), second.get_weights(), strict=True))


def test_tcn_spec_json_and_explicit_weights(tmp_path, monkeypatch):
    spec = ModelSpec(params=compact_tcn_params(num_classes=3), input_shape=(16, 2))
    decoded = ModelSpec.model_validate(json.loads(spec.model_dump_json()))
    assert isinstance(decoded.params, TcnParams) and decoded == spec

    def forbidden(*args, **kwargs):
        raise AssertionError("construction changed caller-owned state")

    monkeypatch.setattr(keras.backend, "clear_session", forbidden)
    monkeypatch.setattr(keras.utils, "set_random_seed", forbidden)
    model = build(decoded)
    path = tmp_path / "source.weights.h5"
    model.save_weights(path)
    restored = build(decoded)
    restored.load_weights(path)
    x = np.arange(32, dtype="float32").reshape(1, 16, 2) / 32
    expected = keras.ops.convert_to_numpy(model(x, training=False))
    np.testing.assert_array_equal(expected, keras.ops.convert_to_numpy(restored(x, training=False)))
    restored.save(tmp_path / "restored.keras")
    loaded = keras.models.load_model(tmp_path / "restored.keras", compile=False)
    np.testing.assert_array_equal(expected, keras.ops.convert_to_numpy(loaded(x, training=False)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("filters", 0),
        ("depth", 0),
        ("branch", 0),
        ("kernel", [1, 0]),
        ("dilation", -1),
        ("ex_ratio", 0),
        ("se_ratio", -1),
        ("dropout", 1.0),
    ],
)
def test_tcn_config_rejects_invalid_block_geometry(field, value):
    config = compact_tcn_params().model_dump(mode="json")
    config["blocks"][0][field] = value
    with pytest.raises(ValidationError):
        TcnParams.model_validate(config)


@pytest.mark.parametrize("field,value", [("input_kernel", [0, 3]), ("output_kernel", -1)])
def test_tcn_config_rejects_invalid_outer_geometry(field, value):
    config = compact_tcn_params().model_dump(mode="json")
    config[field] = value
    with pytest.raises(ValidationError):
        TcnParams.model_validate(config)


@pytest.mark.parametrize("explicit_zero", [False, True])
def test_small_tcn_zero_se_disables_attention(explicit_zero, tmp_path):
    block = {"filters": 8, "kernel": [1, 3], "norm": "batch"}
    if explicit_zero:
        block["se_ratio"] = 0
    params = TcnParams.model_validate(
        {"block_type": "sm", "blocks": [block], "output_kernel": [1, 1], "num_classes": 3}
    )
    model = tcn_build(params, (16, 2))
    assert model.output_shape == (None, 16, 3)
    assert not any(
        isinstance(layer, (keras.layers.GlobalAveragePooling2D, keras.layers.Multiply)) for layer in model.layers
    )
    assert sum(isinstance(layer, keras.layers.DepthwiseConv2D) for layer in model.layers) == 1
    assert sum(isinstance(layer, keras.layers.Conv2D) for layer in model.layers) == 2
    x = np.arange(32, dtype="float32").reshape(1, 16, 2) / 32
    expected = keras.ops.convert_to_numpy(model(x, training=False))
    assert np.isfinite(expected).all()
    model.save(tmp_path / "zero.keras")
    restored = keras.models.load_model(tmp_path / "zero.keras", compile=False)
    np.testing.assert_array_equal(expected, keras.ops.convert_to_numpy(restored(x, training=False)))
