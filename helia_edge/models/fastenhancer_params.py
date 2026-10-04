"""Validated FastEnhancer architecture config, official presets and weight mappings; no backend imports."""

import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..importers.mapping import Reshape, SourcePin, Transpose, WeightMapping, WeightRow


class FastEnhancerRNNFormerParams(BaseModel):
    """RNNFormer stage: per-band GRU over time, then attention across bands."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_blocks: int = Field(default=2, ge=1)
    channels: int = Field(default=20, gt=0)
    freq: int = Field(default=16, ge=2)
    num_heads: int = Field(default=4, ge=1)
    positional_embedding: bool = True
    attn_bias: bool = False
    post_act: bool = False

    @model_validator(mode="after")
    def _heads_divide_channels(self) -> "FastEnhancerRNNFormerParams":
        if self.channels % self.num_heads:
            raise ValueError("rnnformer channels must be divisible by num_heads")
        return self


class FastEnhancerParams(BaseModel):
    """Folded-inference FastEnhancer config for one spectral frame per call.

    BatchNorm and weight normalization are folded into biased layers, matching
    the released inference graphs; this form is not the trainable architecture.
    STFT/iSTFT framing belongs to the caller. Changing any field defines a new,
    untrained architecture unless matching weights exist.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["fastenhancer"] = "fastenhancer"
    form: Literal["folded_inference"] = "folded_inference"
    n_fft: int = Field(default=512, ge=4)
    channels: int = Field(default=24, gt=0)
    kernel_size: tuple[int, ...] = (8, 3, 3)
    stride: int = Field(default=4, ge=1)
    rnnformer: FastEnhancerRNNFormerParams = FastEnhancerRNNFormerParams()
    activation: Literal["silu", "relu"] = "silu"
    mask: Literal["none", "sigmoid", "tanh"] = "none"
    input_compression: float = Field(default=0.3, gt=0, le=1, allow_inf_nan=False)
    resnet: bool = False

    @model_validator(mode="after")
    def _consistent_geometry(self) -> "FastEnhancerParams":
        if self.n_fft % 2:
            raise ValueError("n_fft must be even")
        if len(self.kernel_size) < 1 or any(k < 1 for k in self.kernel_size):
            raise ValueError("kernel_size must be nonempty positive integers")
        k0 = self.kernel_size[0]
        if k0 % self.stride or (k0 - self.stride) % 2:
            raise ValueError("kernel_size[0] must be a multiple of stride with even (kernel - stride)")
        if any(k % 2 == 0 for k in self.kernel_size[1:]):
            raise ValueError("encoder and decoder kernels after the first must be odd")
        if (self.n_fft // 2) % self.stride:
            raise ValueError("n_fft // 2 must be divisible by stride")
        if self.rnnformer.freq > self.encoder_bins:
            raise ValueError("rnnformer freq must not exceed n_fft // 2 // stride; filters would cover no bin")
        return self

    @property
    def spectral_bins(self) -> int:
        """Spectral input/output bins, including the zero-filled Nyquist bin."""
        return self.n_fft // 2 + 1

    @property
    def input_shape(self) -> tuple[int, ...]:
        """The fixed ``spec_in`` shape ``(spectral_bins, 1, 2)``, without the batch axis."""
        return (self.spectral_bins, 1, 2)

    @property
    def encoder_bins(self) -> int:
        """Frequency positions after the strided input convolution."""
        return self.n_fft // 2 // self.stride


FASTENHANCER_PRESET_SOURCE = "aask1357/fastenhancer@e74cab157662dd0d28f381959f9b4870654d5d1f"

FASTENHANCER_PRESETS: Mapping[str, FastEnhancerParams] = {
    # configs/fastenhancer/t.yaml at FASTENHANCER_PRESET_SOURCE (onnx-vd-v1.0.0 era).
    "fastenhancer_t": FastEnhancerParams(),
}


class FastEnhancerResolvedConfig(BaseModel):
    """Serializable record of a preset, its overrides and the concrete config.

    Reload from ``params``; re-resolving a preset later may differ if the
    preset table changes. ``official`` is true only when the resolved config
    equals the preset exactly.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    preset: str
    source: str
    overrides: dict[str, Any]
    official: bool
    params: FastEnhancerParams


def resolve_fastenhancer(preset: str, overrides: Mapping[str, Any] | None = None) -> FastEnhancerResolvedConfig:
    """Resolve an official preset with validated overrides into a full record."""
    if preset not in FASTENHANCER_PRESETS:
        raise ValueError(f"unknown FastEnhancer preset {preset!r}; choose from {sorted(FASTENHANCER_PRESETS)}")
    try:
        overrides = json.loads(json.dumps(dict(overrides or {})))
    except TypeError as exc:
        raise ValueError(f"FastEnhancer overrides must be JSON-compatible: {exc}") from exc
    base = FASTENHANCER_PRESETS[preset].model_dump(mode="json")
    for key, value in overrides.items():
        if key not in base:
            raise ValueError(f"unknown FastEnhancer override {key!r}")
        if isinstance(base[key], dict) and isinstance(value, Mapping):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    params = FastEnhancerParams.model_validate_json(json.dumps(base))
    official = params == FASTENHANCER_PRESETS[preset]
    return FastEnhancerResolvedConfig(
        preset=preset,
        source=FASTENHANCER_PRESET_SOURCE,
        overrides=overrides,
        official=official,
        params=params,
    )


def fastenhancer_mapping(
    params: FastEnhancerParams,
    name: str,
    source: SourcePin,
    *,
    tensor_names: Mapping[str, str] | None = None,
    unused: tuple[str, ...] = (),
) -> WeightMapping:
    """Mapping of every weight of ``build(params)`` from folded tensors in ONNX export layout.

    Source tensors are keyed by module path, as ``fastenhancer.fastenhancer_weight_shapes(params)`` lists
    them. GRU tensors must use ONNX gate order z, r, h; PyTorch ``nn.GRU`` stores r, z, n, which shapes
    alone cannot detect.

    Args:
        params: The params the tensors were trained with.
        name: Mapping name.
        source: The pinned source file.
        tensor_names: Source tensor names for module paths stored under another name.
        unused: Source tensors deliberately not imported.

    Returns:
        WeightMapping: The mapping, for ``helia_edge.importers.import_weights``.
    """
    names = tensor_names or {}
    rf, c, rc, k0, s = (
        params.rnnformer,
        params.channels,
        params.rnnformer.channels,
        params.kernel_size[0],
        params.stride,
    )
    conv = Transpose(perm=(2, 1, 0))  # [out, in, k] (transposed: [in, out, k]) to Keras [k, in, out] ([k, out, in])
    rows = []

    def row(path: str, layer: str, weight: str, *transforms) -> None:
        rows.append(WeightRow(sources=(names.get(path, path),), transforms=transforms, layer=layer, weight=weight))

    def conv_rows(module: str, layer: str) -> None:
        row(f"{module}.weight", layer, "kernel", conv)
        row(f"{module}.bias", layer, "bias")

    # The source's strided input convolution stores [out, phase * 2 + ri, j]; Keras needs [out, ri, j * stride + phase]
    enc = (Reshape(shape=(c, s, 2, k0 // s)), Transpose(perm=(0, 2, 3, 1)), Reshape(shape=(c, 2, k0)), conv)
    row("enc_pre.0.weight", "enc_pre_conv", "kernel", *enc)
    row("enc_pre.0.bias", "enc_pre_conv", "bias")
    for i in range(len(params.kernel_size) - 1):
        conv_rows(f"encoder.{i}.0", f"encoder_{i}_conv")
        conv_rows(f"decoder.{i}.0", f"decoder_{i}_conv0")
        conv_rows(f"decoder.{i}.2", f"decoder_{i}_conv1")
    row("rf_pre.0.kernel", "rf_pre_proj", "kernel")
    conv_rows("rf_pre.1", "rf_pre_conv")
    for i in range(rf.num_blocks):
        b, layer = f"rf_block.{i}", f"rf_block_{i}"
        gate = (Reshape(shape=(3 * rc, rc)), Transpose(perm=(1, 0)))  # [1, 3H, in] to [in, 3H]
        row(f"{b}.rnn.W", f"{layer}_gru", "kernel", *gate)
        row(f"{b}.rnn.R", f"{layer}_gru", "recurrent_kernel", *gate)
        row(f"{b}.rnn.B", f"{layer}_gru", "bias", Reshape(shape=(2, 3 * rc)))
        row(f"{b}.rnn_fc.kernel", f"{layer}_rnn_fc", "kernel")
        row(f"{b}.rnn_fc.bias", f"{layer}_rnn_fc", "bias")
        row(f"{b}.attn.qkv.kernel", f"{layer}_attn", "kernel")
        if rf.attn_bias:
            row(f"{b}.attn.qkv.bias", f"{layer}_attn", "bias")
        row(f"{b}.attn_fc.kernel", f"{layer}_attn_fc", "kernel")
        row(f"{b}.attn_fc.bias", f"{layer}_attn_fc", "bias")
    if rf.positional_embedding:
        row("rf_block.0.pe", "rf_block_0_pe", "embedding")
    row("rf_post.0.kernel", "rf_post_proj", "kernel")
    conv_rows("rf_post.1", "rf_post_conv")
    conv_rows("dec_post.0", "dec_post_conv")
    conv_rows("dec_post.2", "dec_post_upsample")
    return WeightMapping(name=name, source=source, rows=tuple(rows), unused=unused)


FASTENHANCER_T_ONNX = fastenhancer_mapping(
    FASTENHANCER_PRESETS["fastenhancer_t"],
    "fastenhancer_t_onnx",
    SourcePin(
        uri="https://github.com/aask1357/fastenhancer/releases/download/onnx-vd-v1.0.0/fastenhancer_t.spec.onnx",
        sha256="915a451f3b1ea8e98c20517c63b50943aa1d540624189f3106cc2e03d09634eb",
        format="onnx",
        note="onnx-vd-v1.0.0 FP32 spectral graph (VoiceBank-DEMAND, 16 kHz); code MIT.",
    ),
    # The release stores these tensors under anonymous initializer names
    tensor_names={
        "rf_pre.0.kernel": "onnx::MatMul_638",
        "rf_block.0.rnn.W": "onnx::GRU_662",
        "rf_block.0.rnn.R": "onnx::GRU_663",
        "rf_block.0.rnn.B": "onnx::GRU_664",
        "rf_block.0.rnn_fc.kernel": "onnx::MatMul_675",
        "rf_block.0.attn.qkv.kernel": "onnx::MatMul_680",
        "rf_block.0.attn_fc.kernel": "onnx::MatMul_702",
        "rf_block.1.rnn.W": "onnx::GRU_724",
        "rf_block.1.rnn.R": "onnx::GRU_725",
        "rf_block.1.rnn.B": "onnx::GRU_726",
        "rf_block.1.rnn_fc.kernel": "onnx::MatMul_737",
        "rf_block.1.attn.qkv.kernel": "onnx::MatMul_742",
        "rf_block.1.attn_fc.kernel": "onnx::MatMul_764",
        "rf_post.0.kernel": "onnx::MatMul_766",
    },
    # Graph constants (shapes, axes and scalars), not weights
    unused=(
        "/Constant_output_0",
        "/Constant_1_output_0",
        "/Constant_2_output_0",
        "/Constant_3_output_0",
        "/Constant_4_output_0",
        "/Constant_6_output_0",
        "/Constant_7_output_0",
        "/Constant_8_output_0",
        "/rf_block.0/Constant_output_0",
        "/rf_block.0/Constant_1_output_0",
        "/rf_block.0/attn/Constant_output_0",
        "/rf_block.0/attn/Constant_1_output_0",
        "/rf_block.0/attn/Constant_3_output_0",
        "/rf_block.0/attn/Constant_7_output_0",
        "/rf_block.0/attn/Constant_11_output_0",
        "/Constant_14_output_0",
        "/Constant_17_output_0",
        "/enc_pre/enc_pre.0/Reshape_1_output_0",
        "/Concat_output_0",
        "/rf_block.0/attn/Sqrt_1_output_0",
        "/Concat_1_output_0",
        "/Reshape_5_output_0",
        "/enc_pre/enc_pre.0/Concat_1_output_0",
        "/enc_pre/enc_pre.0/Concat_2_output_0",
    ),
)
"""Mapping of every weight of ``build(FastEnhancerParams())`` from the pinned FastEnhancer-T ONNX release."""

MAPPINGS: dict[str, WeightMapping] = {FASTENHANCER_T_ONNX.name: FASTENHANCER_T_ONNX}
"""Weight mappings for this family, by name."""
