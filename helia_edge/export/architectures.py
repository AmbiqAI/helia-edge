"""Architectures an export recipe can build, with one builder signature.

Every builder takes ``(params, input_shape, num_classes)`` and returns an untrained Keras model
with batch size 1. ``BUILTIN_ARCHITECTURES`` holds the built-in entries of
``helia_edge.registry.architectures`` as ``"module:attr"`` strings, so nothing is imported until a
recipe uses one; other packages add architectures through the registry.
"""

from collections.abc import Callable, Mapping
from typing import Any

BUILTIN_ARCHITECTURES: dict[str, str] = {
    "tcn": "helia_edge.export.architectures:build_tcn",
    "mlperf_tiny": "helia_edge.export.architectures:build_mlperf_tiny",
    "miniresnet_v1": "helia_edge.export.architectures:build_miniresnet_v1",
    "timeppg": "helia_edge.export.architectures:build_timeppg",
    "cornet": "helia_edge.export.architectures:build_cornet",
    "vad_silero_v6": "helia_edge.export.architectures:build_vad_silero_v6",
}

Builder = Callable[[Mapping[str, Any], tuple[int, ...] | None, int | None], Any]


def resolve_architecture(name: str) -> Builder:
    """Return the builder registered for ``name`` in ``helia_edge.registry.architectures``.

    Raises:
        NotRegistered: A ``ValueError`` if neither a built-in nor a plugin registers ``name``.
    """
    from ..registry import architectures

    return architectures.get(name)


def _require(name: str, input_shape, num_classes, *, shape: bool, classes: bool | None) -> None:
    if shape and input_shape is None:
        raise ValueError(f"{name} needs input_shape")
    if not shape and input_shape is not None:
        raise ValueError(f"{name} has a fixed input shape; omit input_shape")
    if classes is True and num_classes is None:
        raise ValueError(f"{name} needs num_classes")
    if classes is False and num_classes is not None:
        raise ValueError(f"{name} has no num_classes")


def _input(input_shape):
    import keras

    return keras.Input(tuple(input_shape), batch_size=1)


def build_tcn(params, input_shape, num_classes):
    """Build a TCN from ``TcnParams``; needs ``input_shape``, ``num_classes`` optional."""
    from ..models import TcnModel, TcnParams

    _require("tcn", input_shape, num_classes, shape=True, classes=None)
    return TcnModel.model_from_params(_input(input_shape), TcnParams.from_config(params), num_classes=num_classes)


def build_mlperf_tiny(params, input_shape, num_classes):
    """Build an MLPerf Tiny reference model from ``MlperfTinyParams``; its input shape is fixed."""
    from ..models import MlperfTinyModel, MlperfTinyParams

    _require("mlperf_tiny", input_shape, num_classes, shape=False, classes=False)
    return MlperfTinyModel.model_from_params(MlperfTinyParams.model_validate(params))


def build_miniresnet_v1(params, input_shape, num_classes):
    """Build MiniResNet-v1 from ``MiniResNetV1Params``; needs ``input_shape`` and ``num_classes``."""
    from ..models import MiniResNetV1Model, MiniResNetV1Params

    _require("miniresnet_v1", input_shape, num_classes, shape=True, classes=True)
    return MiniResNetV1Model.model_from_params(_input(input_shape), MiniResNetV1Params.from_config(params), num_classes)


def build_timeppg(params, input_shape, num_classes):
    """Build TimePPG from ``TimePPGParams``; needs ``input_shape``, one regression output."""
    from ..models import TimePPGModel, TimePPGParams

    _require("timeppg", input_shape, num_classes, shape=True, classes=False)
    return TimePPGModel.model_from_params(_input(input_shape), TimePPGParams.from_config(params))


def build_cornet(params, input_shape, num_classes):
    """Build CorNET from ``CorNetParams``; needs ``input_shape``, one regression output."""
    from ..models import CorNetModel, CorNetParams

    _require("cornet", input_shape, num_classes, shape=True, classes=False)
    return CorNetModel.model_from_params(_input(input_shape), CorNetParams.from_config(params))


def build_vad_silero_v6(params, input_shape, num_classes):
    """Build the Silero VAD v6 streaming model from ``SileroVadParams``; its input shape is fixed."""
    from ..models import SileroVadParams, silero_vad_v6

    _require("vad_silero_v6", input_shape, num_classes, shape=False, classes=False)
    return silero_vad_v6(SileroVadParams.model_validate(params))
