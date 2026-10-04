"""The model spec: one serializable identity per architecture, and the single builder; importable without Keras.

A ``ModelSpec`` is the typed Params of one family (selected by its ``family`` field) and the input shape
without the batch axis. ``build(spec)`` builds it. Families are a closed set: each has a Keras-free
``<family>_params.py`` and a ``build(params, input_shape, *, batch_size=None, name=None)`` in its model module.
"""

from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .composer_params import ComposerParams
from .conformer_params import ConformerParams
from .convmixer_params import ConvMixerParams
from .cornet_params import CorNetParams
from .efficientnet_params import EfficientNetParams
from .fastenhancer_params import FastEnhancerParams
from .metaformer_params import MetaFormerParams
from .miniresnet_params import MiniResNetV1Params
from .mlperf_tiny_params import MlperfTinyParams
from .mobilenet_params import MobileNetV1Params
from .mobileone_params import MobileOneParams
from .regnet_params import RegNetParams
from .resnet_params import ResNetParams
from .silero_vad_params import SileroVadParams
from .tcn_params import TcnParams
from .timeppg_params import TimePPGParams
from .tsmixer_params import TsMixerParams
from .unet_params import UNetParams
from .unext_params import UNextParams

if TYPE_CHECKING:
    import keras

ModelParams = Annotated[
    ComposerParams
    | ConformerParams
    | ConvMixerParams
    | CorNetParams
    | EfficientNetParams
    | FastEnhancerParams
    | MetaFormerParams
    | MiniResNetV1Params
    | MlperfTinyParams
    | MobileNetV1Params
    | MobileOneParams
    | RegNetParams
    | ResNetParams
    | SileroVadParams
    | TcnParams
    | TimePPGParams
    | TsMixerParams
    | UNetParams
    | UNextParams,
    Field(discriminator="family"),
]
"""The Params of any family, discriminated by ``family``."""

FIXED_INPUT = (FastEnhancerParams, MlperfTinyParams, SileroVadParams)
"""Families whose input shape follows from their Params (``params.input_shape``); their ``input_shape`` may be None."""


class ModelSpec(BaseModel):
    """The serializable identity of an architecture.

    Attributes:
        params (ModelParams): The family's typed Params; ``params.family`` selects the family.
        input_shape (tuple[int | None, ...] | None): Input shape without the batch axis; None for a variable
            axis. None for the families in ``FIXED_INPUT``, whose shape follows from their Params.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    params: ModelParams
    input_shape: Annotated[tuple[Annotated[int, Field(gt=0)] | None, ...], Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def _shape_unless_fixed(self) -> "ModelSpec":
        if isinstance(self.params, FIXED_INPUT):
            if self.input_shape is not None and self.input_shape != self.params.input_shape:
                raise ValueError(
                    f"{self.params.family} takes input shape {self.params.input_shape}, not {self.input_shape}"
                )
        elif self.input_shape is None:
            raise ValueError(f"{self.params.family} needs an input_shape")
        return self


def build(spec: ModelSpec, *, batch_size: int | None = None, name: str | None = None) -> "keras.Model":
    """Build the model a spec describes.

    Args:
        spec (ModelSpec): Family Params and input shape.
        batch_size (int | None): Static batch size; None for a dynamic batch.
        name (str | None): Model name; the family when None. Give distinct names to combine two models
            of one family in a Keras graph.

    Returns:
        keras.Model: The model.
    """
    params = spec.params
    if spec.input_shape is None and not isinstance(params, FIXED_INPUT):
        raise ValueError(f"{params.family} needs an input_shape")
    match params:
        case ComposerParams():
            from .composer import build as family_build
        case ConformerParams():
            from .conformer import build as family_build
        case ConvMixerParams():
            from .convmixer import build as family_build
        case CorNetParams():
            from .cornet import build as family_build
        case EfficientNetParams():
            from .efficientnet import build as family_build
        case FastEnhancerParams():
            from .fastenhancer import build as family_build
        case MetaFormerParams():
            from .metaformer import build as family_build
        case MiniResNetV1Params():
            from .miniresnet import build as family_build
        case MlperfTinyParams():
            from .mlperf_tiny import build as family_build
        case MobileNetV1Params():
            from .mobilenet import build as family_build
        case MobileOneParams():
            from .mobileone import build as family_build
        case RegNetParams():
            from .regnet import build as family_build
        case ResNetParams():
            from .resnet import build as family_build
        case SileroVadParams():
            from .silero_vad import build as family_build
        case TcnParams():
            from .tcn import build as family_build
        case TimePPGParams():
            from .timeppg import build as family_build
        case TsMixerParams():
            from .tsmixer import build as family_build
        case UNetParams():
            from .unet import build as family_build
        case UNextParams():
            from .unext import build as family_build
        case _:
            raise TypeError(f"Not a family's Params: {type(params).__name__}")
    return family_build(params, spec.input_shape, batch_size=batch_size, name=name)
