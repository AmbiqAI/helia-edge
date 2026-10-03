"""Registries resolve lazily, load plugins once and name what is registered; no Keras needed."""

import subprocess
import sys
import types

import pytest

from helia_edge import registry
from helia_edge.registry import NotRegistered, Registry


def test_builtins_are_strings_until_used():
    source = """
import sys
from helia_edge import registry
assert "litert:tensorflow" in registry.exporters and "tcn" in registry.architectures
assert "helia_edge.export.litert" not in sys.modules and "helia_edge.models.tcn" not in sys.modules
assert "vad_silero_v6:npu" in registry.lowerings and "helia_edge.models.silero_vad_npu" not in sys.modules
assert not {"keras", "tensorflow", "torch"} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", source], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_architecture_builtins_match_the_recipe_table():
    from helia_edge.export.architectures import BUILTIN_ARCHITECTURES

    assert list(registry.architectures) == sorted(BUILTIN_ARCHITECTURES)
    assert list(registry.exporters) == ["litert:tensorflow"]
    assert list(registry.lowerings) == ["vad_silero_v6:npu"]
    assert {key.split(":")[0] for key in registry.lowerings} <= set(BUILTIN_ARCHITECTURES)


def test_add_register_get_and_duplicates():
    things = Registry("thing", {"lazy": "math:sqrt"})
    import math

    assert things.get("lazy") is math.sqrt

    @things.register("fn")
    def fn():
        return 1

    things.add("direct", len)
    assert things.get("fn") is fn and things.get("direct") is len
    assert list(things) == ["direct", "fn", "lazy"]
    with pytest.raises(ValueError, match="already registered"):
        things.add("fn", fn)
    with pytest.raises(ValueError, match="module:attr"):
        things.add("bad", "math.sqrt")


def test_unknown_keys_list_what_is_registered(monkeypatch):
    monkeypatch.setattr(registry, "_plugins_loaded", True)
    things = Registry("thing", {"a": "math:sqrt"})
    with pytest.raises(NotRegistered, match=r"Unknown thing 'b'; available: \['a'\]"):
        things.get("b")
    assert issubclass(NotRegistered, ValueError)


def test_plugins_load_once_on_a_miss(monkeypatch):
    calls = []

    def register(module):
        calls.append(module)
        module.architectures.add("plugin_arch", len)

    entry_point = types.SimpleNamespace(load=lambda: register)
    monkeypatch.setattr(registry, "_plugins_loaded", False)
    monkeypatch.setattr(
        registry.importlib.metadata,
        "entry_points",
        lambda group: [entry_point] if group == registry.PLUGIN_GROUP else [],
    )
    monkeypatch.setattr(registry, "architectures", Registry("architecture", {"tcn": "math:sqrt"}))
    registry.architectures.get("tcn")  # registered: plugins are not loaded
    assert calls == []
    assert registry.architectures.get("plugin_arch") is len
    assert calls == [registry]
    with pytest.raises(NotRegistered):
        registry.architectures.get("missing")
    assert len(calls) == 1


def test_keys_and_targets_are_validated():
    with pytest.raises(ValueError, match="2 ':'-separated parts"):
        registry.exporters.add("custom", len)
    with pytest.raises(ValueError, match="without ':'"):
        registry.architectures.add("a:b", len)
    with pytest.raises(ValueError, match="module:attr"):
        Registry("thing").add("x", "a:b:c")
    with pytest.raises(ValueError, match="module:attr"):
        Registry("thing", {"x": "nocolon"})


def test_a_failing_plugin_does_not_stop_the_others(monkeypatch):
    def broken(module):
        raise RuntimeError("boom")

    def good(module):
        module.architectures.add("good_arch", len)

    entry_points = [
        types.SimpleNamespace(name="broken", load=lambda: broken),
        types.SimpleNamespace(name="good", load=lambda: good),
    ]
    monkeypatch.setattr(registry, "_plugins_loaded", False)
    monkeypatch.setattr(registry.importlib.metadata, "entry_points", lambda group: entry_points)
    monkeypatch.setattr(registry, "architectures", Registry("architecture"))
    with pytest.raises(registry.PluginError, match="broken: RuntimeError: boom"):
        registry.architectures.get("good_arch")
    assert registry.architectures.get("good_arch") is len
