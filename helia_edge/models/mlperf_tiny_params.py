"""Typed parameters of the MLPerf Tiny family; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class MlperfTinyParams(BaseModel):
    """Select a faithful fixed architecture; initialization and export belong to callers.

    Attributes:
        family (Literal["mlperf_tiny"]): Model family
        architecture (Literal["kws", "vww", "resnet", "ad"]): DS-CNN keyword spotting, MobileNetV1 visual wake
            words, ResNet-8 image classification or the dense anomaly-detection autoencoder
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["mlperf_tiny"] = "mlperf_tiny"
    architecture: Literal["kws", "vww", "resnet", "ad"]
