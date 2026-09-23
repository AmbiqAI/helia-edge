"""Load public exports from the same declarations used by type checkers."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
import os
import sys
from typing import Any

from lazy_loader import attach_stub

from ._backend import Backend


class Extra(StrEnum):
    PLOTTING = "plotting"
    AWS = "aws"


_EXTRA_MODULES = {
    Extra.PLOTTING: {"matplotlib", "pandas", "sklearn", "plotly", "seaborn"},
    Extra.AWS: {"boto3", "botocore"},
}


@contextmanager
def optional_imports(extra: Extra) -> Iterator[None]:
    """Explain an absent capability dependency without masking internal failures."""
    try:
        yield
    except ModuleNotFoundError as exc:
        if exc.name not in _EXTRA_MODULES[extra]:
            raise
        raise ImportError(f"{exc.name} is required. Install helia-edge[{extra}].") from exc


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
            if backend == Backend.TENSORFLOW and sys.version_info >= (3, 14):
                extra = "helia-edge[tensorflow] on Python 3.12–3.13, or helia-edge[torch] with KERAS_BACKEND=torch"
            raise ImportError(
                f"{package}.{name} requires {dependency}. Install {extra} "
                "and select KERAS_BACKEND before importing Keras."
            ) from exc

    return resolve_optional, directory, names
