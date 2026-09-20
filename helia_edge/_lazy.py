"""Load public exports from the same declarations used by type checkers."""

from collections.abc import Callable
import os
from typing import Any

from lazy_loader import attach_stub

from ._backend import Backend


def attach_exports(package: str, filename: str) -> tuple[Callable[[str], Any], Callable[[], list[str]], list[str]]:
    resolve, directory, names = attach_stub(package, filename)

    def resolve_optional(name: str) -> Any:
        try:
            return resolve(name)
        except ModuleNotFoundError as exc:
            dependency = exc.name
            if dependency not in ("keras", *Backend):
                raise
            backend = os.getenv("KERAS_BACKEND", Backend.TENSORFLOW) if dependency == "keras" else dependency
            extra = f"helia-edge[{backend}]" if backend in Backend else "helia-edge[tensorflow] or helia-edge[torch]"
            raise ImportError(
                f"{package}.{name} requires {dependency}. Install {extra} "
                "and select KERAS_BACKEND before importing Keras."
            ) from exc

    return resolve_optional, directory, names
