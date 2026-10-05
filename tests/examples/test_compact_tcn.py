"""Structural and export checks for the compact TCN example."""

import copy
import importlib.util
import json
from pathlib import Path

import keras
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("compact_tcn", ROOT / "examples/compact_tcn/generate.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


def assert_architecture(model, width):
    assert model.input_shape == (1, 240, 14)
    assert model.output_shape == (1, 240, 2)
    depthwise = [layer for layer in model.layers if isinstance(layer, keras.layers.DepthwiseConv2D)]
    assert len(depthwise) == 4
    assert [layer.dilation_rate for layer in depthwise] == [(1, 1), (1, 2), (1, 4), (1, 8)]
    assert all(layer.kernel_size == (1, 3) and layer.padding == "same" for layer in depthwise)
    assert len([layer for layer in model.layers if isinstance(layer, keras.layers.GlobalAveragePooling2D)]) == 4
    assert len([layer for layer in model.layers if isinstance(layer, keras.layers.Multiply)]) == 4
    assert len([layer for layer in model.layers if isinstance(layer, keras.layers.Add)]) == 3
    for stage in range(1, 5):
        assert model.get_layer(f"B{stage}_D1_PW_B1_CN").filters == width
        assert model.get_layer(f"B{stage}_SE_sq").filters == width // 4
        assert model.get_layer(f"B{stage}_SE_ex").filters == width
    neck = model.get_layer("NECK_conv")
    assert neck.filters == 2 and neck.kernel_size == (1, 1)
    assert neck.activation is keras.activations.linear
    assert not any(isinstance(layer, keras.layers.Softmax) for layer in model.layers)


@pytest.mark.parametrize("width", [8, 16])
def test_architecture_and_seeded_weights(width):
    model, _ = fixture.build_model(width)
    assert_architecture(model, width)
    expected = [fixture.array_hash(w) for w in model.get_weights()]
    rebuilt, _ = fixture.build_model(width)
    assert expected == [fixture.array_hash(w) for w in rebuilt.get_weights()]


def test_structure_check_detects_wrong_dilation():
    tcn = copy.deepcopy(fixture.TCN)
    tcn["blocks"][2]["dilation"] = [1, 1]
    mutant, _ = fixture.build_model(8, tcn)
    with pytest.raises(AssertionError):
        assert_architecture(mutant, 8)


def test_calibration_is_reproducible_and_bounded():
    calibration = fixture.calibration_inputs(32)
    assert calibration.shape == (32, 240, 14) and calibration.dtype == np.float32
    assert calibration.min() >= -1 and calibration.max() <= 1
    np.testing.assert_array_equal(calibration, fixture.calibration_inputs(32))


@pytest.mark.parametrize("samples", [0, 257, 1.5])
def test_invalid_calibration_samples_create_nothing(tmp_path, samples):
    with pytest.raises(ValueError, match="calibration samples"):
        fixture.generate(tmp_path / "exports", samples)
    assert not (tmp_path / "exports").exists()


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    output = tmp_path_factory.mktemp("compact-tcn") / "exports"
    fixture.generate(output)
    return output


def test_four_export_records(exported):
    from helia_edge.export import ExportRecord

    records = {path.parent.name: ExportRecord.read(path) for path in exported.glob("*/record.json")}
    assert set(records) == {f"tcn-w{w}-{p}" for w in (8, 16) for p in ("fp32", "a8w8")}
    assert not [p for p in exported.iterdir() if p.name in ("manifest.json", "recipe.json")]
    for name, record in records.items():
        width = int(name.split("-")[1][1:])
        assert record.model.params.blocks[0].filters == width and record.model.input_shape == (240, 14)
        assert record.export.batch_size == 1 and record.export.options.mode == "concrete"
        assert fixture.sha256((exported / name / "model.tflite").read_bytes()) == record.artifact.sha256
        assert record.io.outputs[0].shape == (1, 240, 2)
        assert record.weights.digest == records[f"tcn-w{width}-fp32"].weights.digest  # one weight lineage per width
        if record.export.precision == "a8w8":
            assert record.export.calibration.sha256 and record.io.outputs[0].dtype == "int8"
            graph = json.loads((exported / name / "graph.json").read_text())
            assert all(t["dtype"] in {"INT8", "INT32"} for t in graph["tensors"])
        else:
            assert record.export.calibration is None and record.io.outputs[0].dtype == "float32"


@pytest.mark.parametrize("name", ["tcn-w8-fp32", "tcn-w16-a8w8"])
def test_records_reproduce(exported, name):
    from helia_edge.export.reproduce import reproduce

    calibration = exported / "calibration.npy" if name.endswith("a8w8") else None
    report = reproduce(exported / name / "record.json", exported / name / "model.weights.h5", calibration)
    assert report.status == "same", report


def test_int8_validator_rejects_float_graph(exported):
    with pytest.raises(ValueError, match="floating-point"):
        fixture.graph_info((exported / "tcn-w8-fp32" / "model.tflite").read_bytes(), "a8w8")


def test_emitted_graph_detects_builder_dilation_regression(tmp_path, monkeypatch):
    build = fixture.build_model

    def wrong_builder(width):
        changed = copy.deepcopy(fixture.TCN)
        changed["blocks"][3]["dilation"] = [1, 1]
        return build(width, changed)

    monkeypatch.setattr(fixture, "build_model", wrong_builder)
    with pytest.raises(ValueError, match="violates compact TCN preset"):
        fixture.generate(tmp_path / "wrong-builder")


@pytest.mark.parametrize("mutation", ["se_pool", "se_multiply", "pointwise", "logits", "residual"])
def test_generation_rejects_actual_builder_regression_before_conversion(tmp_path, monkeypatch, mutation):
    model, config = fixture.build_model(8)

    def clone(layer):
        options = layer.get_config()
        if mutation == "se_pool" and layer.name == "B1_SE_pool":
            return keras.layers.GlobalMaxPooling2D(keepdims=True, name=layer.name)
        if mutation == "se_multiply" and layer.name == "B1_SE_ex_mul":
            return keras.layers.Lambda(lambda x: x[0], name=layer.name)
        if mutation == "residual" and layer.name == "B2_ADD":
            return keras.layers.Lambda(lambda x: x[0], name=layer.name)
        if mutation == "pointwise" and layer.name == "B1_D1_PW_B1_CN":
            options["kernel_size"] = (1, 3)
        if mutation == "logits" and layer.name == "NECK_conv":
            options["activation"] = "softmax"
        return type(layer).from_config(options)

    wrong = keras.models.clone_model(model, clone_function=clone)
    monkeypatch.setattr(fixture, "build_model", lambda width: (wrong, config))

    def must_not_convert(*args, **kwargs):
        raise RuntimeError("invalid model reached conversion")

    monkeypatch.setattr(fixture, "export", must_not_convert)
    with pytest.raises(ValueError, match="preset"):
        fixture.generate(tmp_path / mutation)


def test_int8_validator_rejects_non_int8_operand_without_export(exported):
    import flatbuffers

    schema = fixture.schema
    retained = exported / "tcn-w8-a8w8" / "model.tflite"
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(retained.read_bytes(), 0))
    model.subgraphs[0].tensors[model.subgraphs[0].inputs[0]].type = schema.TensorType.UINT8
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    with pytest.raises(ValueError, match="operand dtype"):
        fixture.graph_info(bytes(builder.Output()), "a8w8")


@pytest.mark.parametrize("width", [8, 16])
def test_production_guard_accepts_preset_and_template_filters_are_overridden(width):
    original, _ = fixture.build_model(width)
    fixture.validate_model(original, width)
    tcn = copy.deepcopy(fixture.TCN)
    for block in tcn["blocks"]:
        block["filters"] = 123
    changed, spec = fixture.build_model(width, tcn)
    fixture.validate_model(changed, width)
    assert all(block.filters == width for block in spec.params.blocks)
    assert [fixture.array_hash(w) for w in original.get_weights()] == [
        fixture.array_hash(w) for w in changed.get_weights()
    ]


def test_retained_license_metadata_matches_copied_source(tmp_path):
    metadata = fixture.retain_license(tmp_path)
    assert metadata["source_spdx"] == "BSD-3-Clause"
    assert metadata["sha256"] == fixture.sha256((ROOT / "LICENSE").read_bytes())
    assert (tmp_path / metadata["file"]).read_bytes() == (ROOT / "LICENSE").read_bytes()
