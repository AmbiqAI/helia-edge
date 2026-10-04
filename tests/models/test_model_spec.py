"""ModelSpec: one serializable identity per architecture, built by models.build."""

import json
import subprocess
import sys
import typing

import pydantic
import pytest

from helia_edge.models import MobileNetV1Params, ModelParams, ModelSpec, TcnParams, TsMixerParams
from helia_edge.models.spec import build

FAMILIES = typing.get_args(typing.get_args(ModelParams)[0])


@pytest.mark.parametrize("params_cls", FAMILIES, ids=lambda cls: cls.__name__)
def test_every_family_round_trips_through_json(params_cls):
    spec = ModelSpec(params=params_cls(), input_shape=(1, 32, 4))
    data = json.loads(spec.model_dump_json())
    assert data["params"]["family"] == params_cls.model_fields["family"].default
    restored = ModelSpec.model_validate(data)
    assert type(restored.params) is params_cls and restored == spec


@pytest.mark.parametrize("params_cls", FAMILIES, ids=lambda cls: cls.__name__)
def test_each_family_has_its_params_module_and_builder(params_cls):
    import importlib.util

    family = params_cls.model_fields["family"].default
    assert params_cls.__module__ == f"helia_edge.models.{family}_params"
    assert importlib.util.find_spec(f"helia_edge.models.{family}") is not None
    assert "num_classes" in params_cls.model_fields and "name" not in params_cls.model_fields


def test_families_are_unique():
    families = [cls.model_fields["family"].default for cls in FAMILIES]
    assert len(set(families)) == len(families) == 11


@pytest.mark.parametrize(
    "data",
    [
        {"params": {"family": "nope"}, "input_shape": [4]},
        {"params": {"family": "tcn", "typo": 1}, "input_shape": [4]},
        {"params": {"family": "tcn", "name": "tcn"}, "input_shape": [4]},
        {"params": {"family": "tcn"}, "input_shape": [4], "extra": 1},
        {"params": {"blocks": []}, "input_shape": [4]},
        {"params": {"family": "tcn", "num_classes": 0}, "input_shape": [4]},
    ],
)
def test_specs_are_strict(data):
    with pytest.raises(pydantic.ValidationError):
        ModelSpec.model_validate(data)


def test_params_and_specs_are_frozen():
    spec = ModelSpec(params=TcnParams(), input_shape=(4,))
    with pytest.raises(pydantic.ValidationError):
        spec.params.num_classes = 2
    with pytest.raises(pydantic.ValidationError):
        spec.input_shape = (8,)


def test_specs_and_params_import_without_keras():
    source = """
import sys
import helia_edge.models.spec
from helia_edge.models import ModelSpec, TcnParams, compact_tcn_params
ModelSpec(params=compact_tcn_params(num_classes=2), input_shape=(240, 14))
assert not {"keras", "tensorflow", "torch"} & sys.modules.keys(), sorted({"keras", "tensorflow", "torch"} & sys.modules.keys())
"""
    result = subprocess.run([sys.executable, "-c", source], text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_build_names_the_model_after_its_family_and_sets_the_batch():
    pytest.importorskip("keras")
    spec = ModelSpec(params=TcnParams(blocks=[{"filters": 8}], num_classes=3), input_shape=(1, 16, 2))
    dynamic, static = build(spec), build(spec, batch_size=1)
    assert dynamic.name == static.name == "tcn"
    assert dynamic.input_shape == (None, 1, 16, 2) and static.input_shape == (1, 1, 16, 2)
    assert dynamic.output_shape[-1] == 3


def test_build_refuses_a_non_family():
    with pytest.raises(TypeError, match="Not a family"):
        build(ModelSpec.model_construct(params=object(), input_shape=(4,)))


def _params_classes():
    import importlib
    import inspect

    modules = [f"helia_edge.models.{cls.model_fields['family'].default}_params" for cls in FAMILIES]
    modules.append("helia_edge.layers.mbconv_params")
    for name in modules:
        module = importlib.import_module(name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, pydantic.BaseModel) and cls.__module__ == name:
                yield cls


@pytest.mark.parametrize("cls", list(_params_classes()), ids=lambda cls: cls.__name__)
def test_every_params_class_is_frozen_and_strict(cls):
    assert cls.model_config.get("frozen") is True and cls.model_config.get("extra") == "forbid"


@pytest.mark.parametrize("params_cls", FAMILIES, ids=lambda cls: cls.__name__)
def test_num_classes_is_positive(params_cls):
    with pytest.raises(pydantic.ValidationError):
        params_cls(num_classes=0)


@pytest.mark.parametrize("shape", [(), (0, 4), (-1, 4)])
def test_input_shapes_are_positive_and_not_empty(shape):
    with pytest.raises(pydantic.ValidationError):
        ModelSpec(params=TcnParams(), input_shape=shape)


def test_a_variable_axis_is_none():
    assert ModelSpec(params=TcnParams(), input_shape=(1, None, 4)).input_shape == (1, None, 4)


@pytest.mark.parametrize("params_cls", [TcnParams, MobileNetV1Params, TsMixerParams], ids=lambda cls: cls.__name__)
def test_families_without_an_optional_head_say_they_need_num_classes(params_cls):
    pytest.importorskip("keras")
    with pytest.raises(ValueError, match="needs num_classes"):
        build(ModelSpec(params=params_cls(), input_shape=(1, 32, 4)))


def test_two_models_of_one_family_combine_with_names():
    keras = pytest.importorskip("keras")
    spec = ModelSpec(params=TcnParams(blocks=[{"filters": 8}], num_classes=2), input_shape=(1, 16, 2))
    teacher, student = build(spec, name="teacher"), build(spec, name="student")
    assert (teacher.name, student.name) == ("teacher", "student")
    inputs = keras.Input((1, 16, 2))
    combined = keras.Model(inputs, [teacher(inputs), student(inputs)])
    assert len(combined.outputs) == 2
