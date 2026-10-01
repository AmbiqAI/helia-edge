"""Registries for extensions; importable without Keras.

Keys are open strings. Built-in entries are ``"module:attr"`` strings, imported on first lookup.
Other packages register through the ``helia_edge.plugins`` entry-point group: each entry point is
a callable that receives this module and registers into its registries, for example::

    # pyproject.toml of another package
    [project.entry-points."helia_edge.plugins"]
    mypkg = "mypkg.edge:register"

    # mypkg/edge.py
    def register(registry):
        registry.exporters.add("litert:torch", "mypkg.export:export_torch_litert")

Plugins load once, on the first lookup of a key that is not registered.
"""

import importlib
import importlib.metadata
import sys
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

from .export.architectures import BUILTIN_ARCHITECTURES

V = TypeVar("V")

PLUGIN_GROUP = "helia_edge.plugins"


class NotRegistered(ValueError):
    """No entry is registered under the requested key."""


class Registry(Generic[V]):
    """Named entries of one kind, resolved lazily.

    Args:
        item: What an entry is, used in error messages (for example ``"architecture"``).
        builtins: Built-in entries as ``key -> "module:attr"``.
        key_parts: Number of non-empty ``:``-separated parts a key must have (1 for plain names,
            2 for ``"<format>:<backend>"``).
    """

    def __init__(self, item: str, builtins: dict[str, str] | None = None, key_parts: int = 1) -> None:
        self.item = item
        self.key_parts = key_parts
        self._targets: dict[str, str] = {}
        self._values: dict[str, V] = {}
        for key, target in (builtins or {}).items():
            self.add(key, target)

    def add(self, key: str, value: V | str) -> None:
        """Register ``value`` under ``key``; a ``"module:attr"`` string is imported on first lookup."""
        parts = key.split(":")
        if len(parts) != self.key_parts or not all(parts):
            shape = "a name without ':'" if self.key_parts == 1 else f"{self.key_parts} ':'-separated parts"
            raise ValueError(f"{self.item} keys are {shape}, got {key!r}")
        if key in self:
            raise ValueError(f"{self.item} {key!r} is already registered")
        if isinstance(value, str):
            target = value.split(":")
            if len(target) != 2 or not all(target):
                raise ValueError(f"lazy {self.item} targets are 'module:attr' strings, got {value!r}")
            self._targets[key] = value
        else:
            self._values[key] = value

    def register(self, key: str) -> Callable[[V], V]:
        """Decorator form of ``add``."""

        def decorate(value: V) -> V:
            self.add(key, value)
            return value

        return decorate

    def get(self, key: str) -> V:
        """Return the entry for ``key``, loading plugins once if it is not registered.

        Raises:
            NotRegistered: If no built-in or plugin registers ``key``.
            PluginError: If a ``helia_edge.plugins`` entry point fails while plugins load.
        """
        if key not in self:
            load_plugins()
        if key in self._values:
            return self._values[key]
        if key not in self._targets:
            raise NotRegistered(f"Unknown {self.item} {key!r}; available: {sorted(self)}")
        module, attr = self._targets[key].split(":")
        value = getattr(importlib.import_module(module), attr)
        self._values[key] = value
        return value

    def __contains__(self, key: object) -> bool:
        return key in self._values or key in self._targets

    def __iter__(self) -> Iterator[str]:
        return iter(sorted({*self._values, *self._targets}))


_plugins_loaded = False


class PluginError(RuntimeError):
    """One or more ``helia_edge.plugins`` entry points failed to load or register."""


def load_plugins() -> None:
    """Call every ``helia_edge.plugins`` entry point once, passing this module.

    Every plugin is attempted even if an earlier one fails; failures are then raised together as
    ``PluginError`` (once; later lookups see only what did register).
    """
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    failures = []
    for entry_point in importlib.metadata.entry_points(group=PLUGIN_GROUP):
        try:
            entry_point.load()(sys.modules[__name__])
        except Exception as exc:  # reported below with the plugin's name
            failures.append(f"{getattr(entry_point, 'name', entry_point)}: {type(exc).__name__}: {exc}")
    if failures:
        raise PluginError("helia_edge plugins failed to register:\n" + "\n".join(failures))


exporters: Registry[Callable[..., Any]] = Registry(
    "exporter", {"litert:tensorflow": "helia_edge.export.litert:export_litert"}, key_parts=2
)
"""``"<format>:<backend>" -> exporter(model, spec, calibration) -> ExportResult``."""

architectures: Registry[Callable[..., Any]] = Registry("architecture", BUILTIN_ARCHITECTURES)
"""``name -> builder(params, input_shape, num_classes) -> keras.Model`` with batch size 1."""
