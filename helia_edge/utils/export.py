"""Keras serialization registration without changing decorated object types."""

from types import FunctionType

import keras


def maybe_register_serializable(symbol: object, package: str) -> None:
    if isinstance(symbol, FunctionType) or hasattr(symbol, "get_config"):
        keras.saving.register_keras_serializable(package=package)(symbol)


class helia_export:
    """Register serializable symbols; path is retained for API compatibility."""

    def __init__(self, path: str | None = None, package: str = "helia_edge") -> None:
        self.path = path
        self.package = package

    def __call__[T](self, symbol: T) -> T:
        maybe_register_serializable(symbol, self.package)
        return symbol


nse_export = helia_export
