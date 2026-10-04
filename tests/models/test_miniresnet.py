"""MiniResNet config, geometry and explicit Keras hydration contracts."""

import json

import keras
import numpy as np
import pytest
from pydantic import ValidationError

from helia_edge.models import MiniResNetV1Params, ModelSpec
from helia_edge.models import build as build_spec


def build(params=MiniResNetV1Params(), input_shape=(64, 50, 1), num_classes=10):
    params = params.model_copy(update={"num_classes": num_classes})
    return build_spec(ModelSpec(params=params, input_shape=input_shape))


def test_spec_round_trips_and_params_are_frozen():
    spec = ModelSpec(params=MiniResNetV1Params(num_classes=10), input_shape=(64, 50, 1))
    assert ModelSpec.model_validate(json.loads(spec.model_dump_json())) == spec
    with pytest.raises(ValidationError):
        spec.params.stacks = 2


@pytest.mark.parametrize(
    "config",
    [
        {"stacks": 0},
        {"stacks": 4},
        {"base_filters": 0},
        {"dropout": 1.0},
        {"dropout": float("nan")},
        {"dropout": float("inf")},
        {"pooling": "None"},
        {"num_classes": 0},
        {"seed": 7},
        {"weights": "asset.keras"},
        {"name": "miniresnet_v1"},
    ],
)
def test_invalid_config(config):
    with pytest.raises(ValidationError):
        MiniResNetV1Params.model_validate(config)


def test_default_checkpoint_geometry():
    model = build()
    assert model.name == "miniresnet"
    assert model.count_params() == 126922
    assert len(model.layers) == 27
    assert model.output_shape == (None, 10)
    assert model.get_layer("flatten").output.shape == (None, 3584)
    convs = [layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D)]
    assert len(convs) == 6
    assert all(layer.use_bias and layer.filters == 64 for layer in convs)
    assert model.get_layer("conv2_block1_0_conv").strides == (2, 2)
    assert model.get_layer("conv2_block2_1_conv").strides == (1, 1)
    assert all(
        layer.epsilon == 1.001e-5 for layer in model.layers if isinstance(layer, keras.layers.BatchNormalization)
    )


@pytest.mark.parametrize("pooling", ["flatten", "avg", "max"])
def test_smaller_variant_hydrates_and_serializes(pooling, tmp_path):
    params = MiniResNetV1Params(base_filters=8, stacks=2, pooling=pooling, dropout=0.1)
    model = build(params, (33, 25, 1), 3)
    x = np.random.default_rng(9).normal(size=(2, 33, 25, 1)).astype("float32")
    expected = keras.ops.convert_to_numpy(model(x, training=False))
    np.testing.assert_allclose(expected.sum(-1), 1.0, atol=1e-6)
    path = tmp_path / "fixture.keras"
    model.save(path)
    hydrated = build(params, (33, 25, 1), 3)
    hydrated.load_weights(path)
    restored = keras.models.load_model(path, compile=False, safe_mode=True)
    for other in (hydrated, restored):
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(other(x, training=False)), expected)
    mismatch = build(params, (33, 25, 1), 4)
    with pytest.raises(ValueError):
        mismatch.load_weights(path)


@pytest.mark.parametrize("shape,classes", [((50, 1), 10), ((64, 50, 1), 0), ((64, 50, 1), None), ((None, 50, 1), 10)])
def test_invalid_input_contract(shape, classes):
    with pytest.raises(ValueError):
        build(MiniResNetV1Params(), shape, classes)


def test_dynamic_global_pooling_and_explicit_layout(monkeypatch):
    # A caller's global image layout must not silently change this NHWC family.
    previous = keras.config.image_data_format()
    try:
        keras.config.set_image_data_format("channels_first")

        def forbidden(*args, **kwargs):
            raise AssertionError("constructor altered caller state")

        monkeypatch.setattr(keras.backend, "clear_session", forbidden)
        monkeypatch.setattr(keras.utils, "set_random_seed", forbidden)
        model = build(MiniResNetV1Params(base_filters=8, pooling="avg"), (None, None, 1), 2)
        assert model(np.ones((1, 17, 13, 1), dtype="float32"), training=False).shape == (1, 2)
        assert keras.config.image_data_format() == "channels_first"
    finally:
        keras.config.set_image_data_format(previous)
