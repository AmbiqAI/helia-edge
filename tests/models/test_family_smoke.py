"""Smoke tests for model families without dedicated tests: build, forward, save/load, LiteRT FP32."""

from collections import Counter
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
    TcnModel,
    TcnParams,
    TsMixerModel,
    TsMixerParams,
    UNetModel,
    UNetParams,
    UNextModel,
    UNextParams,
)

SERIES = (64, 4)  # (time, channels)
ROW = (1, 64, 4)  # (1, time, channels) for families built from 2D layers
SQUARE = (16, 16, 4)  # 2D input, to stride the first spatial axis
NUM_CLASSES = 3
TORCH = keras.backend.backend() == "torch"


@dataclass(frozen=True)
class Case:
    model_cls: Any
    params: Any
    shape: tuple[int, ...]
    reduced: Any = None  # the same params with one block (or one level of depth) fewer
    class_axis: int = -1  # TsMixer forecasts num_classes steps along axis 1
    layer_counts: Any = None  # exact layer-type counts, for families without a block list


def known_bug(issues, raises, condition=True):
    reason = ", ".join(f"AmbiqAI/helia-edge#{issue}" for issue in issues)
    return pytest.mark.xfail(condition, strict=True, raises=raises, reason=reason)


def drop_last(params, field="blocks"):
    return params.model_copy(update={field: getattr(params, field)[:-1]})


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
    subsamples=[{"depth": 16}, {"depth": 16}],
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
RESNET_STRIDED_SAME_WIDTH = ResNetParams(
    input_filters=8,
    input_kernel_size=(1, 3),
    input_strides=(1, 2),
    blocks=[{"filters": 8, "kernel_size": (1, 3)}, {"filters": 8, "kernel_size": (1, 3), "strides": (1, 2)}],
)
RESNET_BOTTLENECK_STRIDED_SAME_WIDTH = ResNetParams(
    **{
        **RESNET_STRIDED_SAME_WIDTH.model_dump(),
        "blocks": [{**b.model_dump(), "bottleneck": True} for b in RESNET_STRIDED_SAME_WIDTH.blocks],
    }
)
REGNET_Z_STRIDED_SAME_WIDTH = REGNET_STRIDED_SAME_WIDTH.model_copy(update={"block_style": "z"})
TSMIXER = TsMixerParams(blocks=[{"ff_dim": 8}, {"ff_dim": 8}])
TSMIXER_DEFAULT_WIDTH = TsMixerParams(blocks=[{}, {"dropout": 0.1}])  # ff_dim None uses the channel count
STRIDE_2_1 = {"kernel_size": 3, "strides": (2, 1)}
RESNET_SQUARE_STRIDED = ResNetParams(input_filters=8, blocks=[{"filters": 8}, {"filters": 8, **STRIDE_2_1}])
REGNET_SQUARE_STRIDED = RegNetParams(input_filters=8, blocks=[{"filters": 8, "group_width": 4, **STRIDE_2_1}])
REGNET_Z_SQUARE_STRIDED = REGNET_SQUARE_STRIDED.model_copy(update={"block_style": "z"})
UNET_BLOCK = {"filters": 8, "depth": 1, "kernel": (1, 3), "pool": (1, 2), "strides": (1, 2)}
UNET = UNetParams(blocks=[UNET_BLOCK, {**UNET_BLOCK, "filters": 16}])
UNET_LAYER_NORM = UNetParams(blocks=[{**UNET_BLOCK, "norm": "layer"}, {**UNET_BLOCK, "filters": 16, "norm": "layer"}])
TCN_BLOCK = {"depth": 1, "branch": 1, "filters": 8, "kernel": (1, 3), "dilation": (1, 1), "dropout": 0, "ex_ratio": 1}
TCN_LAYER_NORM = TcnParams(
    input_kernel=(1, 3),
    input_norm="layer",
    blocks=[
        {**TCN_BLOCK, "se_ratio": 0, "norm": "layer"},
        {**TCN_BLOCK, "dilation": (1, 2), "se_ratio": 0, "norm": "layer"},
    ],
    output_kernel=(1, 3),
    include_top=True,
    use_logits=True,
)
UNEXT = UNextParams(blocks=[UNET_BLOCK, {**UNET_BLOCK, "filters": 16}])

FAMILIES = [
    pytest.param(Case(ComposerModel, COMPOSER, ROW, drop_last(COMPOSER, "layers")), id="composer-dense"),
    pytest.param(
        Case(
            ComposerModel,
            COMPOSER_CONV,
            ROW,
            # conv2d, batch_norm, relu6 (Activation), se_block (pool, 2 Conv2D, 2 Activation, Multiply), head
            layer_counts={
                "Conv2D": 3,
                "BatchNormalization": 1,
                "Activation": 3,
                "GlobalAveragePooling2D": 1,
                "Multiply": 1,
            },
        ),
        id="composer-conv",
    ),
    pytest.param(Case(ConformerModel, CONFORMER, ROW, drop_last(CONFORMER)), id="conformer"),
    pytest.param(Case(ConvMixerModel, CONVMIXER, ROW, CONVMIXER.model_copy(update={"depth": 1})), id="convmixer"),
    pytest.param(Case(EfficientNetV2Model, EFFICIENTNET, ROW, drop_last(EFFICIENTNET)), id="efficientnetv2"),
    pytest.param(
        Case(MetaFormerModel, METAFORMER, ROW, drop_last(METAFORMER)),
        id="metaformer",
    ),
    pytest.param(
        Case(
            MobileNetV1Model,
            MobileNetV1Params(input_filters=8),
            ROW,
            layer_counts={"DepthwiseConv2D": 13, "Conv2D": 14, "BatchNormalization": 27},
        ),
        id="mobilenetv1",
    ),
    pytest.param(Case(MobileOneModel, MOBILEONE, ROW, drop_last(MOBILEONE)), id="mobileone"),
    pytest.param(Case(RegNetModel, REGNET, ROW, drop_last(REGNET)), id="regnet"),
    pytest.param(Case(RegNetModel, REGNET_STRIDED_SAME_WIDTH, ROW), id="regnet-strided-same-width"),
    pytest.param(Case(ResNetModel, RESNET, ROW, drop_last(RESNET)), id="resnet"),
    pytest.param(Case(ResNetModel, RESNET_STRIDED_SAME_WIDTH, ROW), id="resnet-strided-same-width"),
    pytest.param(
        Case(ResNetModel, RESNET_BOTTLENECK_STRIDED_SAME_WIDTH, ROW), id="resnet-bottleneck-strided-same-width"
    ),
    pytest.param(Case(RegNetModel, REGNET_Z_STRIDED_SAME_WIDTH, ROW), id="regnet-z-strided-same-width"),
    pytest.param(
        Case(TsMixerModel, TSMIXER, SERIES, drop_last(TSMIXER), class_axis=1),
        id="tsmixer",
    ),
    pytest.param(
        Case(TsMixerModel, TSMIXER_DEFAULT_WIDTH, SERIES, drop_last(TSMIXER_DEFAULT_WIDTH), class_axis=1),
        id="tsmixer-default-width",
    ),
    pytest.param(Case(ResNetModel, RESNET_SQUARE_STRIDED, SQUARE), id="resnet-square-strided"),
    pytest.param(Case(RegNetModel, REGNET_SQUARE_STRIDED, SQUARE), id="regnet-square-strided"),
    pytest.param(Case(RegNetModel, REGNET_Z_SQUARE_STRIDED, SQUARE), id="regnet-z-square-strided"),
    pytest.param(Case(UNetModel, UNET, ROW, drop_last(UNET)), id="unet"),
    pytest.param(Case(UNetModel, UNET_LAYER_NORM, ROW, drop_last(UNET_LAYER_NORM)), id="unet-layer-norm"),
    pytest.param(Case(TcnModel, TCN_LAYER_NORM, ROW, drop_last(TCN_LAYER_NORM)), id="tcn-layer-norm"),
    pytest.param(
        Case(UNextModel, UNEXT, ROW, drop_last(UNEXT)),
        id="unext",
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
    if case.layer_counts is not None:
        counts = Counter(type(layer).__name__ for layer in model.layers)
        assert {name: counts[name] for name in case.layer_counts} == case.layer_counts
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
    expected = predict(model, x)
    scale = np.abs(expected).max()
    # Relative to the output scale, so outputs that are only numerical noise fail (observed at most 1.2e-6).
    assert scale > 0 and np.abs(LiteRTRunner(result.content).predict(x) - expected).max() <= 1e-5 * scale


def test_mobilenetv1_depthwise_layers_keep_he_normal_and_l2():
    model = build(Case(MobileNetV1Model, MobileNetV1Params(input_filters=8), ROW))
    for layer in model.layers:
        if isinstance(layer, keras.layers.DepthwiseConv2D):
            config = layer.get_config()
            assert config["depthwise_initializer"]["class_name"] == "HeNormal"
            assert config["depthwise_regularizer"]["class_name"] == "L2"


def test_tsmixer_default_feed_forward_width_is_the_channel_count():
    model = build(Case(TsMixerModel, TSMIXER_DEFAULT_WIDTH, SERIES))
    widths = [layer.units for layer in model.layers if layer.name.endswith("_FL_DENSE")]
    assert widths == [SERIES[-1]] * len(TSMIXER_DEFAULT_WIDTH.blocks)


def test_conformer_layer_norms_normalize_features():
    model = build(Case(ConformerModel, CONFORMER, ROW))
    axes = [layer.axis for layer in model.layers if isinstance(layer, keras.layers.LayerNormalization)]
    # Five layer norms per block (two feed-forward, attention, convolution, output), each over features.
    assert axes == [[-1]] * 5 * len(CONFORMER.blocks), axes
