"""Validated FastEnhancer architecture config and official presets; no backend imports."""

import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
