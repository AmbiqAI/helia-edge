"""Smoke tests for model families without dedicated tests: build, forward, save/load, LiteRT FP32."""

from dataclasses import dataclass
from typing import Any

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
TORCH = keras.backend.backend() == "torch"


@dataclass(frozen=True)
class Case:
    model_cls: Any
    params: Any
    shape: tuple[int, ...]
    reduced: Any = None  # the same params with one block (or one level of depth) fewer
    class_axis: int = -1  # TsMixer forecasts num_classes steps along axis 1


def known_bug(issues, raises, condition=True):
    reason = ", ".join(f"AmbiqAI/helia-edge#{issue}" for issue in issues)
    return pytest.mark.xfail(condition, strict=True, raises=raises, reason=reason)


def drop_last(params, field="blocks"):
    return params.model_copy(update={field: getattr(params, field)[:-1]})


# Dotted layer names from the shared layer helpers are rejected by Torch (#73).
DOTTED_NAMES_ON_TORCH = known_bug([73], KeyError, condition=TORCH)

COMPOSER = ComposerParams(layers=[{"name": "dense", "params": {"units": 8}}, {"name": "dense", "params": {"units": 8}}])
COMPOSER_CONV = ComposerParams(
    layers=[
        {"name": "conv2d", "params": {"filters": 8, "kernel_size": (1, 3)}},
        {"name": "batch_norm", "params": {}},
        {"name": "relu6", "params": {}},
        {"name": "se_block", "params": {"ratio": 2}},
    ]
)
CONFORMER = ConformerParams(
    subsamples=[{"depth": 16}],
    blocks=[{"depth": 16, "num_heads": 2, "kernel_size": 3}, {"depth": 16, "num_heads": 2, "kernel_size": 3}],
)
CONVMIXER = ConvMixerParams(filters=8, depth=2, kernel_size=3, patch_size=2)
EFFICIENTNET = EfficientNetParams(
    input_filters=8,
    input_kernel_size=(1, 3),
    input_strides=(1, 2),
    blocks=[
        {"filters": 8, "depth": 1, "kernel_size": (1, 3), "strides": (1, 2), "ex_ratio": 1, "se_ratio": 2},
        {"filters": 16, "depth": 2, "kernel_size": (1, 3), "strides": (1, 2), "ex_ratio": 2, "se_ratio": 2},
    ],
    output_filters=16,
)
METAFORMER_BLOCK = {
    "layers": 1,
    "patch_embed": {"embed_dim": 8, "patch_shape": (1, 2), "stride_shape": (1, 2)},
    "token_mixer": {"name": "pool", "args": {"pool_size": (1, 3)}},
    "channel_mixer": {"name": "mlp", "args": {"embed_dim": 8, "ratio": 2}},
}
METAFORMER = MetaFormerParams(blocks=[METAFORMER_BLOCK, METAFORMER_BLOCK])
MOBILEONE_BLOCK = {"filters": 8, "depth": 1, "kernel_size": (1, 3), "strides": (1, 2), "padding": (0, 1), "se_ratio": 2}
MOBILEONE = MobileOneParams(
    input_filters=8,
    input_kernel_size=(1, 3),
    input_strides=(1, 2),
    input_padding=(0, 1),
    blocks=[MOBILEONE_BLOCK, {**MOBILEONE_BLOCK, "filters": 16}],
)
REGNET = RegNetParams(
    input_filters=8,
    blocks=[
        {"filters": 16, "depth": 1, "group_width": 4, "kernel_size": (1, 3), "strides": (1, 2), "se_ratio": 2},
        {"filters": 16, "depth": 2, "group_width": 4, "kernel_size": (1, 3), "strides": (1, 1), "se_ratio": 2},
    ],
)
REGNET_STRIDED_SAME_WIDTH = RegNetParams(
    input_filters=8, blocks=[{"filters": 8, "depth": 1, "group_width": 4, "kernel_size": (1, 3), "strides": (1, 2)}]
)
RESNET = ResNetParams(
    input_filters=8,
    input_kernel_size=(1, 3),
    input_strides=(1, 2),
    blocks=[
        {"filters": 8, "depth": 1, "kernel_size": (1, 3), "strides": (1, 2)},
        {"filters": 16, "depth": 1, "kernel_size": (1, 3), "strides": (1, 2), "bottleneck": True},
    ],
)
TSMIXER = TsMixerParams(blocks=[{"ff_dim": 8}, {"ff_dim": 8}])
UNET_BLOCK = {"filters": 8, "depth": 1, "kernel": (1, 3), "pool": (1, 2), "strides": (1, 2)}
UNET = UNetParams(blocks=[UNET_BLOCK, {**UNET_BLOCK, "filters": 16}])
UNEXT = UNextParams(blocks=[UNET_BLOCK, {**UNET_BLOCK, "filters": 16}])

FAMILIES = [
    pytest.param(Case(ComposerModel, COMPOSER, ROW, drop_last(COMPOSER, "layers")), id="composer-dense"),
    pytest.param(Case(ComposerModel, COMPOSER_CONV, ROW), id="composer-conv", marks=known_bug([71], TypeError)),
    pytest.param(
        Case(ConformerModel, CONFORMER, ROW, drop_last(CONFORMER)),
        id="conformer",
        # On Torch, the #73 dotted-name KeyError follows once #66 is fixed.
        marks=known_bug([66, 73], (AttributeError, KeyError)) if TORCH else known_bug([66], AttributeError),
    ),
    pytest.param(Case(ConvMixerModel, CONVMIXER, ROW, CONVMIXER.model_copy(update={"depth": 1})), id="convmixer"),
    pytest.param(
        Case(EfficientNetV2Model, EFFICIENTNET, ROW, drop_last(EFFICIENTNET)),
        id="efficientnetv2",
        marks=DOTTED_NAMES_ON_TORCH,
    ),
    pytest.param(
        Case(MetaFormerModel, METAFORMER, ROW, drop_last(METAFORMER)),
        id="metaformer",
        marks=known_bug([67], ValueError),
    ),
    pytest.param(
        Case(MobileNetV1Model, MobileNetV1Params(input_filters=8), ROW),
        id="mobilenetv1",
        marks=known_bug([68], ValueError),
    ),
    pytest.param(Case(MobileOneModel, MOBILEONE, ROW, drop_last(MOBILEONE)), id="mobileone"),
    pytest.param(Case(RegNetModel, REGNET, ROW, drop_last(REGNET)), id="regnet", marks=DOTTED_NAMES_ON_TORCH),
    pytest.param(
        Case(RegNetModel, REGNET_STRIDED_SAME_WIDTH, ROW),
        id="regnet-strided-same-width",
        # On Torch the #73 KeyError comes first; with #73 fixed, the #72 ValueError remains.
        marks=known_bug([72, 73], (ValueError, KeyError)) if TORCH else known_bug([72], ValueError),
    ),
    pytest.param(Case(ResNetModel, RESNET, ROW, drop_last(RESNET)), id="resnet"),
    pytest.param(
        Case(TsMixerModel, TSMIXER, SERIES, drop_last(TSMIXER), class_axis=1),
        id="tsmixer",
        marks=known_bug([69], TypeError),
    ),
    pytest.param(Case(UNetModel, UNET, ROW, drop_last(UNET)), id="unet"),
    pytest.param(
        Case(UNextModel, UNEXT, ROW, drop_last(UNEXT)),
        id="unext",
        # On Torch, layer normalization over a non-last axis fails once #70 is fixed (#74).
        marks=known_bug([70, 74], (TypeError, RuntimeError)) if TORCH else known_bug([70], TypeError),
    ),
]


def build(case, params=None):
    keras.backend.clear_session()
    keras.utils.set_random_seed(0)
    inputs = keras.Input(shape=case.shape, batch_size=1)
    return case.model_cls.model_from_params(inputs=inputs, params=params or case.params, num_classes=NUM_CLASSES)


def sample(shape, seed=1):
    return np.random.default_rng(seed).normal(size=(1, *shape)).astype(np.float32)


def predict(model, x):
    return keras.ops.convert_to_numpy(model(x, training=False))


@pytest.mark.parametrize("case", FAMILIES)
def test_builds_runs_and_reloads(case, tmp_path):
    if case.reduced is not None:
        reduced = build(case, case.reduced).count_params()
        assert build(case).count_params() > reduced, "the last block (or depth level) was not built"
    model = build(case)
    assert model.output_shape[0] == 1 and model.output_shape[case.class_axis] == NUM_CLASSES
    x = sample(case.shape)
    y = predict(model, x)
    assert y.shape == tuple(model.output_shape)
    assert np.isfinite(y).all()
    spread = np.abs(predict(model, sample(case.shape, seed=2)) - y).max()
    assert spread > 0 and spread > 1e-3 * np.abs(y).max(), "the output does not depend on the input"
    model.save(tmp_path / "model.keras")
    np.testing.assert_array_equal(predict(keras.models.load_model(tmp_path / "model.keras"), x), y)


@pytest.mark.parametrize("case", FAMILIES)
def test_exports_to_litert_fp32(case):
    if TORCH:
        pytest.skip("LiteRT export needs the TensorFlow backend")
    pytest.importorskip("ai_edge_litert")
    from helia_edge.export import ExportSpec, LiteRTRunner, export_model

    model = build(case)
    x = sample(case.shape)
    result = export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="concrete"))
    np.testing.assert_allclose(LiteRTRunner(result.content).predict(x), predict(model, x), rtol=1e-5, atol=1e-5)
