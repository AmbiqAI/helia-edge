"""Modules removed in the hygiene step stay removed, and the examples avoid the deprecated converters."""

import importlib
from pathlib import Path

import pytest

import helia_edge

REMOVED = [
    "helia_edge.optimizers",
    "helia_edge.quantizers",
    "helia_edge.converters.torch",
    "helia_edge.utils.train",
    "helia_edge.models.cct",
]


@pytest.mark.parametrize("module", REMOVED)
def test_removed_modules_do_not_import(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize("name", ["optimizers", "quantizers"])
def test_removed_namespaces_are_not_exported(name):
    assert name not in dir(helia_edge)
    with pytest.raises(AttributeError):
        getattr(helia_edge, name)


def test_examples_do_not_use_the_deprecated_converters():
    root = Path(__file__).resolve().parents[1]
    offenders = [
        str(path.relative_to(root))
        for path in sorted((root / "examples").rglob("*.py"))
        if any(name in path.read_text() for name in ("helia_edge.converters", "helia_edge.interpreters"))
    ]
    assert offenders == []
