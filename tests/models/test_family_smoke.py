"""Smoke tests for model families without dedicated tests: build, forward, save/load, LiteRT FP32."""

import keras
import numpy as np
import pytest

from helia_edge.models import (
    ComposerModel,
    ComposerParams,
    ConformerModel,
    ConformerParams,
    ConvMixerModel,
    ConvMixerParams,
    EfficientNetParams,
    EfficientNetV2Model,
    MetaFormerModel,
    MetaFormerParams,
    MobileNetV1Model,
    MobileNetV1Params,
    MobileOneModel,
    MobileOneParams,
    RegNetModel,
    RegNetParams,
    ResNetModel,
    ResNetParams,
    TsMixerModel,
    TsMixerParams,
    UNetModel,
    UNetParams,
    UNextModel,
    UNextParams,
)

SERIES = (64, 4)  # (time, channels)
ROW = (1, 64, 4)  # (1, time, channels) for families built from 2D layers
NUM_CLASSES = 3


def known_bug(issue, raises, condition=True):
    return pytest.mark.xfail(condition, strict=True, raises=raises, reason=f"AmbiqAI/helia-edge#{issue}")


# Dotted layer names from the shared layer helpers are rejected by Torch (#73).
DOTTED_NAMES_ON_TORCH = known_bug(73, KeyError, condition=keras.backend.backend() == "torch")


FAMILIES = [
    pytest.param(
        ComposerModel,
        ComposerParams(layers=[{"name": "dense", "params": {"units": 8}}, {"name": "dense", "params": {"units": 8}}]),
        ROW,
        id="composer-dense",
    ),
    pytest.param(
        ComposerModel,
        ComposerParams(
            layers=[
                {"name": "conv2d", "params": {"filters": 8, "kernel_size": (1, 3)}},
                {"name": "batch_norm", "params": {}},
                {"name": "relu6", "params": {}},
                {"name": "se_block", "params": {"ratio": 2}},
            ]
        ),
        ROW,
        id="composer-conv",
        marks=known_bug(71, TypeError),
    ),
    pytest.param(
        ConformerModel,
        ConformerParams(subsamples=[{"depth": 16}], blocks=[{"depth": 16, "num_heads": 2, "kernel_size": 3}]),
        SERIES,
        id="conformer",
        marks=known_bug(66, AttributeError),
    ),
    pytest.param(ConvMixerModel, ConvMixerParams(filters=8, depth=2, kernel_size=3, patch_size=2), ROW, id="convmixer"),
    pytest.param(
        EfficientNetV2Model,
        EfficientNetParams(
            input_filters=8,
            input_kernel_size=(1, 3),
            input_strides=(1, 2),
            blocks=[
                {"filters": 8, "depth": 1, "kernel_size": (1, 3), "strides": (1, 2), "ex_ratio": 1, "se_ratio": 2},
                {"filters": 16, "depth": 2, "kernel_size": (1, 3), "strides": (1, 2), "ex_ratio": 2, "se_ratio": 2},
            ],
            output_filters=16,
        ),
        ROW,
        id="efficientnetv2",
        marks=DOTTED_NAMES_ON_TORCH,
    ),
    pytest.param(
        MetaFormerModel,
        MetaFormerParams(
            blocks=[
                {
                    "layers": 1,
                    "patch_embed": {"embed_dim": 8, "patch_shape": (1, 4), "stride_shape": (1, 4)},
                    "token_mixer": {"name": "pool", "args": {"pool_size": (1, 3)}},
                    "channel_mixer": {"name": "mlp", "args": {"embed_dim": 8, "ratio": 2}},
                }
            ]
        ),
        ROW,
        id="metaformer",
        marks=known_bug(67, ValueError),
    ),
    pytest.param(
        MobileNetV1Model, MobileNetV1Params(input_filters=8), ROW, id="mobilenetv1", marks=known_bug(68, ValueError)
    ),
    pytest.param(
        MobileOneModel,
        MobileOneParams(
            input_filters=8,
            input_kernel_size=(1, 3),
            input_strides=(1, 2),
            input_padding=(0, 1),
            blocks=[
                {"filters": 8, "depth": 1, "kernel_size": (1, 3), "strides": (1, 2), "padding": (0, 1), "se_ratio": 2}
            ],
        ),
        ROW,
        id="mobileone",
    ),
    pytest.param(
        RegNetModel,
        RegNetParams(
            input_filters=8,
            blocks=[
                {"filters": 16, "depth": 1, "group_width": 4, "kernel_size": (1, 3), "strides": (1, 2), "se_ratio": 2},
                {"filters": 16, "depth": 2, "group_width": 4, "kernel_size": (1, 3), "strides": (1, 1), "se_ratio": 2},
            ],
        ),
        ROW,
        id="regnet",
        marks=DOTTED_NAMES_ON_TORCH,
    ),
    pytest.param(
        RegNetModel,
        RegNetParams(
            input_filters=8,
            blocks=[{"filters": 8, "depth": 1, "group_width": 4, "kernel_size": (1, 3), "strides": (1, 2)}],
        ),
        ROW,
        id="regnet-strided-same-width",
        marks=known_bug(72, ValueError) if keras.backend.backend() != "torch" else DOTTED_NAMES_ON_TORCH,
    ),
    pytest.param(
        ResNetModel,
        ResNetParams(
            input_filters=8,
            input_kernel_size=(1, 3),
            input_strides=(1, 2),
            blocks=[
                {"filters": 8, "depth": 1, "kernel_size": (1, 3), "strides": (1, 2)},
                {"filters": 16, "depth": 1, "kernel_size": (1, 3), "strides": (1, 2), "bottleneck": True},
            ],
        ),
        ROW,
        id="resnet",
    ),
    pytest.param(
        TsMixerModel,
        TsMixerParams(blocks=[{"ff_dim": 8}, {"ff_dim": 8}]),
        SERIES,
        id="tsmixer",
        marks=known_bug(69, TypeError),
    ),
    pytest.param(
        UNetModel,
        UNetParams(
            blocks=[
                {"filters": 8, "depth": 1, "kernel": (1, 3), "pool": (1, 2), "strides": (1, 2)},
                {"filters": 16, "depth": 1, "kernel": (1, 3), "pool": (1, 2), "strides": (1, 2)},
            ]
        ),
        ROW,
        id="unet",
    ),
    pytest.param(
        UNextModel,
        UNextParams(blocks=[{"filters": 8, "depth": 1, "kernel": (1, 3), "pool": (1, 2), "strides": (1, 2)}]),
        ROW,
        id="unext",
        marks=known_bug(70, TypeError),
    ),
]


def build(model_cls, params, shape):
    keras.backend.clear_session()
    keras.utils.set_random_seed(0)
    inputs = keras.Input(shape=shape, batch_size=1)
    return model_cls.model_from_params(inputs=inputs, params=params, num_classes=NUM_CLASSES)


def sample(shape):
    return np.random.default_rng(1).normal(size=(1, *shape)).astype(np.float32)


@pytest.mark.parametrize("model_cls,params,shape", FAMILIES)
def test_builds_runs_and_reloads(model_cls, params, shape, tmp_path):
    model = build(model_cls, params, shape)
    assert model.output_shape[-1] == NUM_CLASSES
    assert len(model.layers) > 3, "the family's blocks were not built"
    x = sample(shape)
    y = keras.ops.convert_to_numpy(model(x, training=False))
    assert y.shape == tuple(model.output_shape)
    assert np.isfinite(y).all()
    model.save(tmp_path / "model.keras")
    reloaded = keras.models.load_model(tmp_path / "model.keras")
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(reloaded(x, training=False)), y)


@pytest.mark.parametrize("model_cls,params,shape", FAMILIES)
def test_exports_to_litert_fp32(model_cls, params, shape):
    if keras.backend.backend() != "tensorflow":
        pytest.skip("LiteRT export needs the TensorFlow backend")
    pytest.importorskip("ai_edge_litert")
    from helia_edge.export import ExportSpec, LiteRTRunner, export_model

    model = build(model_cls, params, shape)
    x = sample(shape)
    result = export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="concrete"))
    expected = keras.ops.convert_to_numpy(model(x, training=False))
    np.testing.assert_allclose(LiteRTRunner(result.content).predict(x), expected, rtol=1e-5, atol=1e-5)
