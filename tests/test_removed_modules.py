"""Removed modules stay removed, and the examples do not use the removed converters."""

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
    "helia_edge.converters",
    "helia_edge.converters.tflite",
    "helia_edge.converters.litert",
    "helia_edge.converters.cpp",
    "helia_edge.interpreters",
    "helia_edge.interpreters.tflite",
    "helia_edge.utils.factory",
]


@pytest.mark.parametrize("module", REMOVED)
def test_removed_modules_do_not_import(module):
    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module(module)
    missing = excinfo.value.name  # the module itself, or a removed package above it
    assert missing is not None and (module == missing or module.startswith(missing + "."))


@pytest.mark.parametrize("name", ["optimizers", "quantizers", "registry", "converters", "interpreters"])
def test_removed_namespaces_are_not_exported(name):
    assert name not in dir(helia_edge)
    with pytest.raises(AttributeError):
        getattr(helia_edge, name)


@pytest.mark.parametrize(
    "name",
    [
        "ExportRecipe",
        "ExportManifest",
        "run_recipe",
        "verify_manifest",
        "load_recipe",
        "LEGACY_PRECISION",
        "LEGACY_MODE",
    ],
)
def test_the_recipe_api_is_not_exported(name):
    import helia_edge.export

    assert name not in dir(helia_edge.export)
    with pytest.raises(AttributeError):
        getattr(helia_edge.export, name)


@pytest.mark.parametrize("name", ["ItemFactory", "create_factory"])
def test_the_item_factory_is_not_exported(name):
    import helia_edge.utils

    assert name not in dir(helia_edge.utils)
    with pytest.raises(AttributeError):
        getattr(helia_edge.utils, name)


def test_fp32_w16_and_the_legacy_tables_are_gone():
    from helia_edge.export import spec

    assert "fp32-w16" not in {p.value for p in spec.Precision}
    assert not {"LEGACY_PRECISION", "LEGACY_MODE"} & set(vars(spec))


def example_sources(root):
    for path in sorted((root / "examples").rglob("*.py")):
        yield path, path.read_text()
    for path in sorted((root / "docs/guides").glob("*.ipynb")):
        cells = json.loads(path.read_text())["cells"]
        yield path, "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


def test_examples_do_not_use_the_removed_converters():
    root = Path(__file__).resolve().parents[1]
    deprecated = re.compile(r"\b(converters|interpreters)\b")
    offenders = [str(path.relative_to(root)) for path, text in example_sources(root) if deprecated.search(text)]
    assert offenders == []
