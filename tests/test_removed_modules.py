"""Removed modules stay removed, and the examples avoid the deprecated converters."""

import importlib
import json
import re
from pathlib import Path

import pytest

import helia_edge

REMOVED = [
    "helia_edge.optimizers",
    "helia_edge.quantizers",
    "helia_edge.converters.torch",
    "helia_edge.utils.train",
    "helia_edge.models.cct",
    "helia_edge.registry",
    "helia_edge.export.recipe",
    "helia_edge.export.run",
    "helia_edge.export.manifest",
    "helia_edge.export.architectures",
]


@pytest.mark.parametrize("module", REMOVED)
def test_removed_modules_do_not_import(module):
    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module(module)
    assert excinfo.value.name == module


@pytest.mark.parametrize("name", ["optimizers", "quantizers", "registry"])
def test_removed_namespaces_are_not_exported(name):
    assert name not in dir(helia_edge)
    with pytest.raises(AttributeError):
        getattr(helia_edge, name)


@pytest.mark.parametrize("name", ["ExportRecipe", "ExportManifest", "run_recipe", "verify_manifest", "load_recipe"])
def test_the_recipe_api_is_not_exported(name):
    import helia_edge.export

    assert name not in dir(helia_edge.export)
    with pytest.raises(AttributeError):
        getattr(helia_edge.export, name)


def example_sources(root):
    for path in sorted((root / "examples").rglob("*.py")):
        yield path, path.read_text()
    for path in sorted((root / "docs/guides").glob("*.ipynb")):
        cells = json.loads(path.read_text())["cells"]
        yield path, "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


def test_examples_do_not_use_the_deprecated_converters():
    root = Path(__file__).resolve().parents[1]
    deprecated = re.compile(r"\b(converters|interpreters)\b")
    offenders = [str(path.relative_to(root)) for path, text in example_sources(root) if deprecated.search(text)]
    assert offenders == []
