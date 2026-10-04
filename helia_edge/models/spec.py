"""The model spec: one serializable identity per architecture, and the single builder; importable without Keras.

A ``ModelSpec`` is the typed Params of one family (selected by its ``family`` field) and the input shape
without the batch axis. ``build(spec)`` builds it. Families are a closed set: each has a Keras-free
``<family>_params.py`` and a ``build(params, input_shape, *, batch_size=None, name=None)`` in its model module.
"""

from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field

from .composer_params import ComposerParams
from .conformer_params import ConformerParams
from .convmixer_params import ConvMixerParams
from .efficientnet_params import EfficientNetParams
from .metaformer_params import MetaFormerParams
from .mobilenet_params import MobileNetV1Params
from .mobileone_params import MobileOneParams
from .regnet_params import RegNetParams
from .resnet_params import ResNetParams
from .tcn_params import TcnParams
from .tsmixer_params import TsMixerParams

if TYPE_CHECKING:
    import keras

ModelParams = Annotated[
    ComposerParams
    | ConformerParams
    | ConvMixerParams
    | EfficientNetParams
    | MetaFormerParams
    | MobileNetV1Params
    | MobileOneParams
    | RegNetParams
    | ResNetParams
    | TcnParams
    | TsMixerParams,
    Field(discriminator="family"),
]
"""The Params of any family, discriminated by ``family``."""


class ModelSpec(BaseModel):
    """The serializable identity of an architecture.

    Attributes:
        params (ModelParams): The family's typed Params; ``params.family`` selects the family.
        input_shape (tuple[int | None, ...]): Input shape without the batch axis; None for a variable axis.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    params: ModelParams
    input_shape: Annotated[tuple[Annotated[int, Field(gt=0)] | None, ...], Field(min_length=1)]


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
    match params:
        case ComposerParams():
            from .composer import build as family_build
        case ConformerParams():
            from .conformer import build as family_build
        case ConvMixerParams():
            from .convmixer import build as family_build
        case EfficientNetParams():
            from .efficientnet import build as family_build
        case MetaFormerParams():
            from .metaformer import build as family_build
        case MobileNetV1Params():
            from .mobilenet import build as family_build
        case MobileOneParams():
            from .mobileone import build as family_build
        case RegNetParams():
            from .regnet import build as family_build
        case ResNetParams():
            from .resnet import build as family_build
        case TcnParams():
            from .tcn import build as family_build
        case TsMixerParams():
            from .tsmixer import build as family_build
        case _:
            raise TypeError(f"Not a family's Params: {type(params).__name__}")
    return family_build(params, spec.input_shape, batch_size=batch_size, name=name)
