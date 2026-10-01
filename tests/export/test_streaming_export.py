"""Export of streaming models: state roles and pairs, name-keyed calibration and tied INT state scales."""

import tempfile

import keras
import numpy as np
import pytest

if keras.backend.backend() != "tensorflow":
    pytest.skip("LiteRT export runs on the TensorFlow backend", allow_module_level=True)
pytest.importorskip("ai_edge_litert.interpreter")

from helia_edge.export import (  # noqa: E402
    ExportSpec,
    LiteRTStreamRunner,
    TensorRole,
    export_model,
    stream_calibration,
)
from helia_edge.export.litert import convert_litert, operator_names, tensor_records, tie_state_scales  # noqa: E402
from helia_edge.export.result import state_scales_tied  # noqa: E402
from helia_edge.layers import StreamingLSTMCell, state_input, state_output  # noqa: E402

UNITS, FEATURES = 12, 6
INT16 = ExportSpec(precision="a16w8", io_dtype="int16", mode="keras")


def build(concat_state=False):
    """A one-step LSTM detector; with ``concat_state`` the hidden state also feeds a concatenation."""
    keras.utils.set_random_seed(7)
    x = keras.Input((FEATURES,), batch_size=1, name="signal")
    h, c = state_input(0, (UNITS,), 1), state_input(1, (UNITS,), 1)
    h_next, c_next = StreamingLSTMCell(UNITS, name="lstm")([x, h, c])
    features = keras.layers.Concatenate()([h_next, h]) if concat_state else h_next
    prob = keras.layers.Dense(1, activation="sigmoid", name="prob")(features)
    model = keras.Model([x, h, c], [prob, state_output(0, h_next), state_output(1, c_next)])
    lstm = model.get_layer("lstm")
    lstm.bias.assign(np.random.default_rng(0).normal(size=lstm.bias.shape).astype(np.float32))
    return model


@pytest.fixture(scope="module")
def model():
    return build()


@pytest.fixture(scope="module")
def signal():
    return np.random.default_rng(1).normal(size=(160, FEATURES)).astype(np.float32)


@pytest.fixture(scope="module")
def calibration(model, signal):
    return stream_calibration(model, {"signal": signal[:96]}, resets=[48])


def keras_stream(model, signal, resets=()):
    """Keras outputs per step, with the states carried as stream_calibration carries them."""
    fed = stream_calibration(model, {"signal": signal}, resets)
    names = [t.name for t in model.inputs]
    outputs = [model([fed[n][t : t + 1] for n in names], training=False) for t in range(len(signal))]
    return {
        name: np.concatenate([keras.ops.convert_to_numpy(o[i]) for o in outputs])
        for i, name in enumerate(model.output_names)
    }


def scaled_state(calibration, factor):
    """Calibration whose state_in_0 samples are scaled, so its calibrated scale differs from state_out_0's."""
    return calibration | {"state_in_0": calibration["state_in_0"] * np.float32(factor)}


def pair_params(records, k):
    return [(r.scale, r.zero_point) for r in records if r.pair == k]


def test_stream_calibration_feeds_back_the_state_and_resets_it(model, signal):
    fed = stream_calibration(model, {"signal": signal[:10]}, resets=[5])
    assert fed["state_in_0"].shape == (10, UNITS) and fed["signal"].shape == (10, FEATURES)
    for k in (0, 1):
        assert not fed[f"state_in_{k}"][[0, 5]].any()
    out = model([fed["signal"][:1], fed["state_in_0"][:1], fed["state_in_1"][:1]], training=False)
    np.testing.assert_array_equal(fed["state_in_0"][1:2], keras.ops.convert_to_numpy(out[1]))
    np.testing.assert_array_equal(fed["state_in_1"][1:2], keras.ops.convert_to_numpy(out[2]))


@pytest.mark.parametrize("mode", ["keras", "saved_model"])
def test_state_tensors_are_recorded_by_pair(model, mode):
    result = export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode=mode))
    assert not {"VAR_HANDLE", "READ_VARIABLE"} & set(operator_names(result.content))
    roles = {(r.role, r.pair) for r in result.inputs} | {(r.role, r.pair) for r in result.outputs}
    assert roles == {(TensorRole.SIGNAL, None), (TensorRole.STATE, 0), (TensorRole.STATE, 1)}
    assert sum(r.role == TensorRole.STATE for r in result.inputs) == 2
    assert sum(r.role == TensorRole.STATE for r in result.outputs) == 2
    assert state_scales_tied(result.inputs, result.outputs) is None
    runner = LiteRTStreamRunner(result.content)
    assert runner.pairs == (0, 1) and runner.signals == ("signal",)


def test_float_export_streams_like_keras(model, signal):
    runner = LiteRTStreamRunner(
        export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="keras")).content
    )
    _, outputs = runner.run({"signal": signal[:40, None]}, resets=[20])
    want = keras_stream(model, signal[:40], resets=[20])
    for name in ("prob", "state_out_0", "state_out_1"):
        np.testing.assert_allclose(outputs[name][:, 0], want[name], rtol=1e-5, atol=1e-5)


def test_int16_export_carries_the_raw_state_and_tracks_keras(model, signal, calibration):
    result = export_model(model, INT16, calibration)
    assert state_scales_tied(result.inputs, result.outputs) is True
    runner = LiteRTStreamRunner(result.content, reference_kernels=True)
    fed, outputs = runner.run({"signal": runner.encode("signal", signal[96:, None])}, resets=[32])
    for k in (0, 1):
        carried, produced = fed[f"state_in_{k}"], outputs[f"state_out_{k}"]
        assert carried.dtype == np.int16
        np.testing.assert_array_equal(carried[1:32], produced[:31])
        np.testing.assert_array_equal(carried[33:], produced[32:-1])
        assert not carried[[0, 32]].any()
    want = keras_stream(model, signal[96:], resets=[32])
    assert np.abs(runner.decode("prob", outputs["prob"])[:, 0] - want["prob"]).max() < 0.01


def converted(model, spec, calibration):
    """The converter's model before the tie."""
    with tempfile.TemporaryDirectory() as workdir:
        return convert_litert(
            model,
            precision=spec.precision,
            io_type=spec.io_dtype.value,
            mode=spec.mode,
            strict=spec.strict,
            calibration=calibration,
            workdir=workdir,
        ).content


@pytest.mark.parametrize("factor", [1.004, 1 / 1.004])
def test_int16_pair_scales_that_differ_slightly_are_tied_to_the_larger_range(model, calibration, factor):
    data = scaled_state(calibration, factor)
    raw_inputs, raw_outputs = tensor_records(converted(model, INT16, data))
    (scale_in, _), (scale_out, _) = pair_params(raw_inputs, 0)[0], pair_params(raw_outputs, 0)[0]
    assert abs(scale_in / scale_out - factor) < 2e-3
    result = export_model(model, INT16, data)
    assert result.content == tie_state_scales(converted(model, INT16, data), INT16.state_tie_tolerance)
    for k in (0, 1):
        larger = max(pair_params(raw_inputs, k) + pair_params(raw_outputs, k))
        assert pair_params(result.inputs, k) == pair_params(result.outputs, k) == [larger]
    assert state_scales_tied(result.inputs, result.outputs) is True


def test_only_a_tied_pair_keeps_the_carried_state_value(model, signal, calibration):
    data = scaled_state(calibration, 1.004)
    drift = {}
    for label, content in (("raw", converted(model, INT16, data)), ("tied", export_model(model, INT16, data).content)):
        runner = LiteRTStreamRunner(content, reference_kernels=True)
        fed, outputs = runner.run({"signal": runner.encode("signal", signal[96:120, None])})
        carried = runner.inputs["state_in_0"]["quantization"][0] * fed["state_in_0"][1:]
        produced = runner.outputs["state_out_0"]["quantization"][0] * outputs["state_out_0"][:-1]
        drift[label] = np.abs(carried - produced).max()
    assert drift["tied"] == 0
    assert drift["raw"] > 0


def test_a_scale_difference_over_the_tolerance_is_refused(model, calibration):
    with pytest.raises(ValueError, match=r"State pair 0 scales .* more than state_tie_tolerance"):
        export_model(model, INT16, scaled_state(calibration, 1.5))
    result = export_model(model, INT16.model_copy(update={"state_tie_tolerance": 0.5}), scaled_state(calibration, 1.5))
    assert state_scales_tied(result.inputs, result.outputs) is True


def test_a_state_read_by_a_scale_preserving_operator_is_refused(signal):
    model = build(concat_state=True)
    calibration = stream_calibration(model, {"signal": signal[:96]})
    with pytest.raises(ValueError, match="CONCATENATION"):
        export_model(model, INT16, scaled_state(calibration, 1.004))


def test_float_io_has_no_integer_state_to_tie(model, calibration):
    result = export_model(
        model, ExportSpec(precision="a16w8", io_dtype="float32", mode="keras"), scaled_state(calibration, 1.5)
    )
    assert state_scales_tied(result.inputs, result.outputs) is None


def test_int8_pairs_are_tied_with_their_zero_points(model, calibration):
    result = export_model(
        model, ExportSpec(precision="a8w8", io_dtype="int8", mode="keras"), scaled_state(calibration, 1.004)
    )
    assert state_scales_tied(result.inputs, result.outputs) is True
    assert pair_params(result.inputs, 0) == pair_params(result.outputs, 0)


def test_multi_input_calibration_must_name_every_input(model, calibration):
    with pytest.raises(ValueError, match="mapping of input name"):
        export_model(model, INT16, calibration["signal"])
    with pytest.raises(ValueError, match="do not match the model inputs"):
        export_model(model, INT16, {k: v for k, v in calibration.items() if k != "state_in_1"})
    with pytest.raises(ValueError, match="different numbers of samples"):
        export_model(model, INT16, calibration | {"signal": calibration["signal"][:-1]})
    with pytest.raises(ValueError, match="not calibrated"):
        export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="keras"), calibration)


def test_the_stream_runner_checks_its_inputs(model):
    runner = LiteRTStreamRunner(
        export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="keras")).content
    )
    with pytest.raises(ValueError, match="do not match the inputs that are not states"):
        runner.run({"state_in_0": np.zeros((2, 1, UNITS), np.float32)})
    feed = runner.initial_state() | {"signal": np.zeros((1, FEATURES), np.float64)}
    with pytest.raises(ValueError, match="'signal' is float64"):
        runner.step(feed)


def test_the_manifest_entry_records_pairs_and_the_tie(model, calibration, tmp_path):
    from helia_edge.export.run import _entry

    entry = _entry("detector", INT16, export_model(model, INT16, calibration).content, None, tmp_path)
    assert entry.state_scales_tied is True
    assert sorted(t.pair for t in entry.inputs if t.role == TensorRole.STATE) == [0, 1]
    assert sorted(t.pair for t in entry.outputs if t.role == TensorRole.STATE) == [0, 1]
    float_entry = _entry(
        "float",
        None,
        export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="keras")).content,
        None,
        tmp_path,
    )
    assert float_entry.state_scales_tied is None
