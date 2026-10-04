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


def _with_classes(name: str, params, num_classes):
    if num_classes is not None:
        if params.get("num_classes") not in (None, num_classes):
            raise ValueError(
                f"{name}: num_classes {num_classes} differs from params num_classes {params['num_classes']}"
            )
        params = {**params, "num_classes": num_classes}
    return params


def build_tcn(params, input_shape, num_classes):
    """Build a TCN from ``TcnParams``; needs ``input_shape``, ``num_classes`` optional."""
    from ..models import TcnParams
    from ..models.tcn import build

    _require("tcn", input_shape, num_classes, shape=True, classes=None)
    params = _with_classes("tcn", params, num_classes)
    return build(TcnParams.model_validate(params), tuple(input_shape), batch_size=1)


def build_mlperf_tiny(params, input_shape, num_classes):
    """Build an MLPerf Tiny reference model from ``MlperfTinyParams``; its input shape is fixed."""
    from ..models import MlperfTinyParams
    from ..models.mlperf_tiny import build

    _require("mlperf_tiny", input_shape, num_classes, shape=False, classes=False)
    return build(MlperfTinyParams.model_validate(params))


def build_miniresnet_v1(params, input_shape, num_classes):
    """Build MiniResNet-v1 from ``MiniResNetV1Params``; needs ``input_shape``, and ``num_classes`` here or in params."""
    from ..models import MiniResNetV1Params
    from ..models.miniresnet import build

    _require("miniresnet_v1", input_shape, num_classes, shape=True, classes=None)
    params = _with_classes("miniresnet_v1", params, num_classes)
    return build(MiniResNetV1Params.model_validate(params), tuple(input_shape), batch_size=1)


def build_timeppg(params, input_shape, num_classes):
    """Build TimePPG from ``TimePPGParams``; needs ``input_shape``, one regression output."""
    from ..models import TimePPGParams
    from ..models.timeppg import build

    _require("timeppg", input_shape, num_classes, shape=True, classes=False)
    return build(TimePPGParams.model_validate(params), tuple(input_shape), batch_size=1)


def build_cornet(params, input_shape, num_classes):
    """Build CorNET from ``CorNetParams``; needs ``input_shape``, one regression output."""
    from ..models import CorNetParams
    from ..models.cornet import build

    _require("cornet", input_shape, num_classes, shape=True, classes=False)
    return build(CorNetParams.model_validate(params), tuple(input_shape), batch_size=1)


def build_vad_silero_v6(params, input_shape, num_classes):
    """Build the Silero VAD v6 streaming model from ``SileroVadParams``; its input shape is fixed."""
    from ..models import SileroVadParams
    from ..models.silero_vad import build

    _require("vad_silero_v6", input_shape, num_classes, shape=False, classes=False)
    return build(SileroVadParams.model_validate(params), batch_size=1)
