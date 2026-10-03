"""Export specification contract; runs without Keras or a training backend."""

import pydantic
import pytest

from helia_edge.export import (
    LEGACY_MODE,
    LEGACY_PRECISION,
    VALID_IO,
    ConversionMode,
    ExportSpec,
    IODType,
    Precision,
)


def test_values_are_the_benchmark_strings():
    assert [p.value for p in Precision] == ["fp32", "fp32-w16", "fp16", "a8w8", "a16w8"]
    assert [d.value for d in IODType] == ["float32", "float16", "int8", "int16"]


@pytest.mark.parametrize("legacy", ["FP16", "fp32_fp16w", "A8W8", "INT8"])
def test_precision_values_are_not_case_folded(legacy):
    with pytest.raises(ValueError):
        Precision(legacy)


def test_legacy_tables_are_explicit():
    assert LEGACY_PRECISION == {
        "FP32": Precision.FP32,
        "FP16": Precision.FP32_FP16W,
        "FP16_NATIVE": Precision.FP16,
        "INT8": Precision.A8W8,
        "INT16X8": Precision.A16W8,
    }
    assert LEGACY_MODE == {
        "KERAS": ConversionMode.KERAS,
        "SAVED_MODEL": ConversionMode.SAVED_MODEL,
        "CONCRETE": ConversionMode.CONCRETE,
    }


@pytest.mark.parametrize("precision", list(Precision))
@pytest.mark.parametrize("io_dtype", list(IODType))
def test_io_dtype_matrix(precision, io_dtype):
    make = lambda: ExportSpec(precision=precision, io_dtype=io_dtype, mode=ConversionMode.CONCRETE)  # noqa: E731
    if io_dtype in VALID_IO[precision]:
        assert make().io_dtype == io_dtype
    else:
        with pytest.raises(pydantic.ValidationError, match="not valid for"):
            make()


def test_matrix_is_the_documented_one():
    assert {p.value: sorted(d.value for d in VALID_IO[p]) for p in Precision} == {
        "fp32": ["float32"],
        "fp32-w16": ["float32"],
        "fp16": ["float16"],
        "a8w8": ["float32", "int8"],
        "a16w8": ["float32", "int16"],
    }


def test_byte_changing_settings_have_no_defaults():
    with pytest.raises(pydantic.ValidationError):
        ExportSpec(precision="fp32", io_dtype="float32")  # mode missing
    with pytest.raises(pydantic.ValidationError):
        ExportSpec(precision="fp32", mode="concrete")  # io_dtype missing


def test_strict_is_a_real_boolean():
    with pytest.raises(pydantic.ValidationError):
        ExportSpec(precision="a8w8", io_dtype="int8", mode="concrete", strict="no")


def test_export_model_without_keras_says_what_to_install():
    import importlib.util

    if importlib.util.find_spec("keras") is not None:
        pytest.skip("Checks the base environment")
    from helia_edge.export import export_model

    with pytest.raises(ImportError, match=r"helia-edge\[litert\]"):
        export_model(object(), ExportSpec(precision="fp32", io_dtype="float32", mode="concrete"))


def test_spec_is_frozen_strict_and_round_trips():
    spec = ExportSpec(precision="a8w8", io_dtype="int8", mode="concrete")
    assert spec.format == "litert" and spec.strict is True
    with pytest.raises(pydantic.ValidationError):
        ExportSpec(precision="a8w8", io_dtype="int8", mode="concrete", calibration=[1])
    with pytest.raises(pydantic.ValidationError):
        spec.strict = False
    data = spec.model_dump(mode="json")
    assert data == {
        "format": "litert",
        "precision": "a8w8",
        "io_dtype": "int8",
        "mode": "concrete",
        "strict": True,
        "state_tie_tolerance": 0.01,
        "lowering": None,
    }
    assert ExportSpec.model_validate(data) == spec


@pytest.mark.parametrize("lowering", ["", "NPU", "npu:int", "a b", 1])
def test_a_lowering_is_a_plain_lowercase_name(lowering):
    with pytest.raises(pydantic.ValidationError):
        ExportSpec(precision="fp32", io_dtype="float32", mode="keras", lowering=lowering)
    assert ExportSpec(precision="fp32", io_dtype="float32", mode="keras", lowering="npu").lowering == "npu"
