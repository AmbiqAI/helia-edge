"""Structural and retained-artifact checks for the compact TCN example."""

import importlib.util
import json
import os
from pathlib import Path
import shutil

import keras
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("compact_tcn", ROOT / "examples/compact_tcn/generate.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)
RECIPE = ROOT / "examples/compact_tcn/recipe.json"


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
    recipe = fixture.read_recipe(RECIPE)
    model, _ = fixture.build_model(recipe, width)
    assert_architecture(model, width)
    expected = [fixture.array_hash(w) for w in model.get_weights()]
    rebuilt, _ = fixture.build_model(recipe, width)
    assert expected == [fixture.array_hash(w) for w in rebuilt.get_weights()]


def test_structure_check_detects_wrong_dilation():
    recipe = fixture.read_recipe(RECIPE)
    recipe["tcn"]["blocks"][2]["dilation"] = [1, 1]
    mutant, _ = fixture.build_model(recipe, 8)
    with pytest.raises(AssertionError):
        assert_architecture(mutant, 8)


@pytest.mark.parametrize("field,value", [("input_shape", [120, 14]), ("widths", [8]),
                                         ("calibration_samples", 0), ("seed", -1)])
def test_invalid_recipe(tmp_path, field, value):
    recipe = json.loads(RECIPE.read_text())
    recipe[field] = value
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError):
        fixture.read_recipe(path)


def test_calibration_is_separate_and_reproducible():
    recipe = fixture.read_recipe(RECIPE)
    calibration, held_out = fixture.fixture_inputs(recipe)
    assert not set(map(fixture.array_hash, calibration)) & set(map(fixture.array_hash, held_out))
    again, _ = fixture.fixture_inputs(recipe)
    np.testing.assert_array_equal(calibration, again)
    np.testing.assert_array_equal(held_out[0], np.zeros((240, 14), np.float32))
    assert held_out[1].min() == -1 and held_out[2].max() == 1
    assert np.ptp(held_out[3]) > 1


def test_quantization_rounds_and_saturates():
    detail = {"dtype": np.int8, "quantization": (0.25, -3)}
    actual = fixture.quantize(np.array([-100, -0.3, 0, 0.3, 100], np.float32), detail)
    np.testing.assert_array_equal(actual, np.array([-128, -4, -3, -2, 127], np.int8))


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    retained = os.environ.get("HELIA_TCN_FIXTURE")
    if retained:
        output = Path(retained)
        fixture.verify(output)
    else:
        output = tmp_path_factory.mktemp("compact-tcn") / "exports"
        fixture.generate(RECIPE, output)
    return output


def test_four_exports_and_full_golden_replay(exported):
    manifest = fixture.verify(exported)
    assert {(e["width"], e["precision"]) for e in manifest["exports"]} == {
        (8, "FP32"), (8, "INT8"), (16, "FP32"), (16, "INT8")}
    for export in manifest["exports"]:
        assert export["output"]["shape"] == [1, 240, 2]
        assert export["output_bytes"] == (480 if export["precision"] == "INT8" else 1920)
        with np.load(exported / export["goldens"], allow_pickle=False) as data:
            assert data["outputs"].shape == (4, 240, 2)
        peers = [e for e in manifest["exports"] if e["width"] == export["width"]]
        assert peers[0]["weight_array_hashes"] == peers[1]["weight_array_hashes"]
        if export["precision"] == "INT8":
            graph = json.loads((exported / export["graph"]).read_text())
            assert all(t["dtype"] in {"INT8", "INT32"} for t in graph["tensors"])


def test_verify_rejects_corrupted_model(exported, tmp_path):
    copy_path = tmp_path / "corrupt"
    shutil.copytree(exported, copy_path)
    model = copy_path / "tcn-w8-int8.tflite"
    model.write_bytes(model.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        fixture.verify(copy_path)


def test_int8_validator_rejects_float_graph(exported):
    with pytest.raises(ValueError, match="floating-point"):
        fixture.graph_info((exported / "tcn-w8-fp32.tflite").read_bytes(), "INT8")


def test_replay_detects_changed_golden_even_with_updated_file_hash(exported, tmp_path):
    copy_path = tmp_path / "wrong-golden"
    shutil.copytree(exported, copy_path)
    manifest_path = copy_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    name = "tcn-w8-int8-goldens.npz"
    with np.load(copy_path / name) as data:
        inputs, outputs = data["inputs"], data["outputs"].copy()
    outputs.flat[0] = 0 if outputs.flat[0] != 0 else 1
    np.savez(copy_path / name, inputs=inputs, outputs=outputs)
    manifest["files"][name] = fixture.sha256((copy_path / name).read_bytes())
    fixture.write_json(manifest_path, manifest)
    with pytest.raises(AssertionError):
        fixture.verify(copy_path)


@pytest.mark.parametrize("change", ["dilation", "kernel", "output_kernel", "input_kernel", "input_norm", "seed"])
def test_generation_rejects_changed_named_preset(tmp_path, change):
    recipe = json.loads(RECIPE.read_text())
    if change == "dilation":
        recipe["tcn"]["blocks"][3]["dilation"] = [1, 1]
    elif change == "kernel":
        recipe["tcn"]["blocks"][0]["kernel"] = [1, 5]
    elif change == "output_kernel":
        recipe["tcn"]["output_kernel"] = [1, 3]
    elif change == "input_kernel":
        recipe["tcn"]["input_kernel"] = [1, 3]
    elif change == "input_norm":
        recipe["tcn"]["input_norm"] = "layer"
    else:
        recipe["seed"] = 42
    path = tmp_path / "wrong-recipe.json"
    path.write_text(json.dumps(recipe))
    output = tmp_path / "exports"
    with pytest.raises(ValueError, match="preset"):
        fixture.generate(path, output)
    assert not output.exists()


def test_emitted_graph_detects_builder_dilation_regression(tmp_path, monkeypatch):
    build = fixture.build_model

    def wrong_builder(recipe, width):
        changed = json.loads(json.dumps(recipe))
        changed["tcn"]["blocks"][3]["dilation"] = [1, 1]
        return build(changed, width)

    monkeypatch.setattr(fixture, "build_model", wrong_builder)
    with pytest.raises(ValueError, match="violates compact TCN preset"):
        fixture.generate(RECIPE, tmp_path / "wrong-builder")


@pytest.mark.parametrize("mutation", ["se_pool", "se_multiply", "pointwise", "logits", "residual"])
def test_generation_rejects_actual_builder_regression_before_conversion(tmp_path, monkeypatch, mutation):
    model, config = fixture.build_model(fixture.read_recipe(RECIPE), 8)

    def clone(layer):
        options = layer.get_config()
        if mutation == "se_pool" and layer.name == "B1_SE_pool":
            return keras.layers.GlobalMaxPooling2D(keepdims=True, name=layer.name)
        if mutation == "se_multiply" and layer.name == "B1_SE_ex.mul":
            return keras.layers.Lambda(lambda x: x[0], name=layer.name)
        if mutation == "residual" and layer.name == "B2_ADD":
            return keras.layers.Lambda(lambda x: x[0], name=layer.name)
        if mutation == "pointwise" and layer.name == "B1_D1_PW_B1_CN":
            options["kernel_size"] = (1, 3)
        if mutation == "logits" and layer.name == "NECK_conv":
            options["activation"] = "softmax"
        return type(layer).from_config(options)

    wrong = keras.models.clone_model(model, clone_function=clone)
    monkeypatch.setattr(fixture, "build_model", lambda recipe, width: (wrong, config))

    def must_not_convert(*args, **kwargs):
        raise RuntimeError("invalid model reached conversion")

    monkeypatch.setattr(fixture, "LiteRTKerasConverter", must_not_convert)
    with pytest.raises(ValueError, match="preset"):
        fixture.generate(RECIPE, tmp_path / mutation)


def test_int8_validator_rejects_non_int8_operand_without_export(exported):
    import flatbuffers
    schema = fixture.schema
    retained = exported / "tcn-w8-int8.tflite"
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(retained.read_bytes(), 0))
    model.subgraphs[0].tensors[model.subgraphs[0].inputs[0]].type = schema.TensorType.UINT8
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    with pytest.raises(ValueError, match="operand dtype"):
        fixture.graph_info(bytes(builder.Output()), "INT8")


@pytest.mark.parametrize("width", [8, 16])
def test_production_guard_accepts_preset_and_template_filters_are_overridden(width):
    recipe = fixture.read_recipe(RECIPE)
    original, _ = fixture.build_model(recipe, width)
    fixture.validate_model(original, width)
    for block in recipe["tcn"]["blocks"]:
        block["filters"] = 123
    changed, config = fixture.build_model(recipe, width)
    fixture.validate_model(changed, width)
    assert all(block["filters"] == width for block in config["blocks"])
    assert [fixture.array_hash(w) for w in original.get_weights()] == [fixture.array_hash(w) for w in changed.get_weights()]


def test_retained_license_metadata_matches_copied_source(tmp_path):
    metadata = fixture.retain_license(tmp_path)
    assert metadata["source_spdx"] == "BSD-3-Clause"
    assert metadata["sha256"] == fixture.sha256((ROOT / "LICENSE").read_bytes())
    assert (tmp_path / metadata["file"]).read_bytes() == (ROOT / "LICENSE").read_bytes()
