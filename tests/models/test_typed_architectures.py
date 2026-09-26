"""Public typed configs preserve model behavior and caller-owned initialization."""

import json

import keras
import numpy as np
import pytest
from pydantic import ValidationError

from helia_edge.models import (
    MlperfTinyModel, MlperfTinyParams, TcnModel, TcnParams, compact_tcn_params,
    mlperf_tiny_ad, mlperf_tiny_kws, mlperf_tiny_resnet, mlperf_tiny_vww,
)

BUILDERS = {"kws": mlperf_tiny_kws, "vww": mlperf_tiny_vww,
            "resnet": mlperf_tiny_resnet, "ad": mlperf_tiny_ad}


@pytest.mark.parametrize("architecture", BUILDERS)
def test_family_adapter_preserves_wrapper_config_weights_and_serialization(architecture, tmp_path):
    params = MlperfTinyParams(architecture=architecture)
    assert MlperfTinyParams.model_validate_json(params.model_dump_json()) == params
    keras.backend.clear_session()
    keras.utils.set_random_seed(123)
    original = BUILDERS[architecture]()
    keras.backend.clear_session()
    keras.utils.set_random_seed(123)
    adapted = MlperfTinyModel.model_from_params(params)
    assert original.to_json() == adapted.to_json()
    for left, right in zip(original.get_weights(), adapted.get_weights(), strict=True):
        np.testing.assert_array_equal(left, right)
    x = np.ones((1, *adapted.input_shape[1:]), np.float32)
    expected = keras.ops.convert_to_numpy(adapted(x, training=False))
    path = tmp_path / "model.keras"
    adapted.save(path)
    restored = keras.models.load_model(path)
    np.testing.assert_array_equal(expected, keras.ops.convert_to_numpy(restored(x, training=False)))


@pytest.mark.parametrize("config", [{"architecture": "resnet18"}, {"architecture": "kws", "seed": 1},
                                     {"architecture": "vww", "scale": 0.5}])
def test_fixed_family_rejects_unsupported_fields(config):
    with pytest.raises(ValidationError):
        MlperfTinyModel.model_from_params(config)


def test_config_is_immutable_and_name_is_instantiation_option():
    params = MlperfTinyParams(architecture="ad")
    with pytest.raises(ValidationError):
        params.architecture = "kws"
    assert MlperfTinyModel.model_from_params({"architecture": "ad"}, name="custom_ad").name == "custom_ad"


@pytest.mark.parametrize("filters", [8, 16])
def test_tcn_preset_strict_roundtrip_legacy_factory_and_serialization(filters, tmp_path):
    params = compact_tcn_params(filters=filters)
    config = json.loads(params.model_dump_json())
    assert TcnParams.from_config(config) == params
    assert [b.dilation for b in params.blocks] == [(1, 1), (1, 2), (1, 4), (1, 8)]
    assert all(b.filters == filters and b.se_ratio == 4 for b in params.blocks)
    keras.backend.clear_session()
    keras.utils.set_random_seed(20260925)
    typed = TcnModel.model_from_params(keras.Input(shape=(240, 14), batch_size=1), params, 2)
    keras.backend.clear_session()
    keras.utils.set_random_seed(20260925)
    legacy = TcnModel.model_from_params(keras.Input(shape=(240, 14), batch_size=1), config, 2)
    assert typed.to_json() == legacy.to_json()
    for left, right in zip(typed.get_weights(), legacy.get_weights(), strict=True):
        np.testing.assert_array_equal(left, right)
    x = np.ones((1, 240, 14), np.float32)
    expected = keras.ops.convert_to_numpy(typed(x, training=False))
    path = tmp_path / "tcn.keras"
    typed.save(path)
    restored = keras.models.load_model(path)
    np.testing.assert_array_equal(expected, keras.ops.convert_to_numpy(restored(x, training=False)))


@pytest.mark.parametrize("nested", [False, True])
def test_strict_ingress_rejects_unknown_fields_without_changing_legacy(nested):
    config = compact_tcn_params().model_dump(mode="json")
    target = config["blocks"][0] if nested else config
    target["typo"] = 1
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TcnParams.from_config(config)
    assert TcnParams.model_validate(config) == compact_tcn_params()


@pytest.mark.parametrize("filters", [True, 7, 8.5])
def test_preset_rejects_invalid_width(filters):
    with pytest.raises(ValueError):
        compact_tcn_params(filters=filters)


def test_public_factories_do_not_reset_seed_or_session(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("factory reset caller-owned global state")

    monkeypatch.setattr(keras.utils, "set_random_seed", forbidden)
    monkeypatch.setattr(keras.backend, "clear_session", forbidden)
    for architecture in BUILDERS:
        MlperfTinyModel.model_from_params(MlperfTinyParams(architecture=architecture))
    params = compact_tcn_params()
    first = TcnModel.model_from_params(keras.Input(shape=(32, 3)), params, 4)
    second = TcnModel.model_from_params(keras.Input(shape=(32, 3)), params, 4)
    assert first.output_shape == second.output_shape == (None, 32, 4)
    assert any(not np.array_equal(a, b) for a, b in zip(first.get_weights(), second.get_weights(), strict=True))
