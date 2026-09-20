"""Resolve public exports without initializing unrelated optional features."""

import importlib
import os
import sys


def lazy_exports(package, exports):
    def resolve(name):
        if name not in exports:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        module_name, symbol = exports[name]
        try:
            module = importlib.import_module(module_name, package)
            value = getattr(module, symbol) if symbol else module
        except ModuleNotFoundError as exc:
            extras = {"tensorflow": "tensorflow", "torch": "torch", "keras": os.getenv("KERAS_BACKEND", "tensorflow")}
            if exc.name in extras:
                extra = extras[exc.name]
                raise ImportError(
                    f"{package}.{name} requires {exc.name}. Install 'helia-edge[{extra}]' "
                    "and select KERAS_BACKEND before importing Keras."
                ) from exc
            raise
        setattr(sys.modules[package], name, value)
        return value

    def directory():
        return sorted(set(vars(sys.modules[package])) | set(exports))

    return resolve, directory, list(exports)
