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
from helia_edge.export.litert import (  # noqa: E402
    _covering_quantization,
    convert_litert,
    operator_names,
    tensor_records,
    tie_state_scales,
)
from helia_edge.export.result import state_scales_tied  # noqa: E402
from helia_edge.layers import StreamingLSTMCell, state_input, state_output  # noqa: E402

UNITS, FEATURES = 12, 6
INT16 = ExportSpec(precision="a16w8", io_dtype="int16", mode="keras")


def build(concat_state=False, tanh_state=False):
    """A one-step LSTM detector whose head reads the hidden state through a biased Dense layer.

    With ``concat_state`` the hidden state also feeds a concatenation; with ``tanh_state`` the cell state
    output is written by a tanh.
    """
    keras.utils.set_random_seed(7)
    x = keras.Input((FEATURES,), batch_size=1, name="signal")
    h, c = state_input(0, (UNITS,), 1), state_input(1, (UNITS,), 1)
    h_next, c_next = StreamingLSTMCell(UNITS, name="lstm")([x, h, c])
    features = keras.layers.Concatenate()([h_next, h]) if concat_state else h_next
    prob = keras.layers.Dense(1, activation="sigmoid", name="prob")(features)
    c_out = keras.layers.Activation("tanh")(c_next) if tanh_state else c_next
    model = keras.Model([x, h, c], [prob, state_output(0, h_next), state_output(1, c_out)])
    lstm = model.get_layer("lstm")
    lstm.bias.assign(np.random.default_rng(0).normal(size=lstm.bias.shape).astype(np.float32))
    model.get_layer("prob").bias.assign(np.array([2.0], np.float32))
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
    interpreter = litert_interpreter(result.content)
    assert [r.name for r in result.inputs] == [d["name"] for d in interpreter.get_input_details()]
    assert [r.name for r in result.outputs] == [d["name"] for d in interpreter.get_output_details()]
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


@pytest.mark.parametrize("factor", [1.5, 1 / 1.5])
def test_a_scale_difference_over_the_tolerance_is_refused(model, calibration, factor):
    with pytest.raises(ValueError, match=r"State pair 0 scales .* more than state_tie_tolerance"):
        export_model(model, INT16, scaled_state(calibration, factor))
    accepted = 1.3 if factor > 1 else 1 / 1.3
    result = export_model(
        model, INT16.model_copy(update={"state_tie_tolerance": 0.5}), scaled_state(calibration, accepted)
    )
    assert state_scales_tied(result.inputs, result.outputs) is True


def test_a_state_read_by_a_scale_preserving_operator_is_refused(signal):
    model = build(concat_state=True)
    calibration = stream_calibration(model, {"signal": signal[:96]})
    with pytest.raises(ValueError, match="CONCATENATION"):
        export_model(model, INT16, scaled_state(calibration, 1.004))


def test_a_state_written_by_a_fixed_scale_operator_is_refused(signal):
    # LiteRT gives a tanh output the fixed scale 1/32768, about 10% from the state input's.
    model = build(tanh_state=True)
    calibration = stream_calibration(model, {"signal": signal[:96]})
    with pytest.raises(ValueError, match="TANH"):
        export_model(model, INT16.model_copy(update={"state_tie_tolerance": 0.2}), calibration)


@pytest.mark.parametrize(("precision", "io_dtype"), [("a16w8", "int16"), ("a8w8", "int8")])
def test_a_tie_keeps_the_bias_of_the_layers_reading_the_state(model, signal, calibration, precision, io_dtype):
    spec = ExportSpec(precision=precision, io_dtype=io_dtype, mode="keras", state_tie_tolerance=0.3)
    want = keras_stream(model, signal[96:], resets=[32])["prob"]
    errors = {}
    for factor in (1.0, 1.25):
        content = export_model(model, spec, scaled_state(calibration, factor)).content
        runner = LiteRTStreamRunner(content, reference_kernels=True)
        _, outputs = runner.run({"signal": runner.encode("signal", signal[96:, None])}, resets=[32])
        errors[factor] = np.abs(runner.decode("prob", outputs["prob"])[:, 0] - want).max()
    assert errors[1.25] < 2 * errors[1.0] + 1e-3


def test_unpaired_or_mismatched_state_is_refused_at_export():
    x = keras.Input((FEATURES,), batch_size=1, name="signal")
    h = state_input(0, (UNITS,), 1)
    unpaired = keras.Model([x, h], keras.layers.Dense(1)(keras.layers.Concatenate()([x, h])))
    with pytest.raises(ValueError, match="do not pair up"):
        export_model(unpaired, ExportSpec(precision="fp32", io_dtype="float32", mode="keras"))
    mismatched = keras.Model([x, h], [keras.layers.Dense(1)(x), state_output(0, keras.layers.Dense(UNITS + 1)(h))])
    with pytest.raises(ValueError, match="different shapes or types"):
        export_model(mismatched, ExportSpec(precision="fp32", io_dtype="float32", mode="keras"))
    content = converted(unpaired, ExportSpec(precision="fp32", io_dtype="float32", mode="keras"), None)
    assert {r.role for r in tensor_records(content)[0]} == {TensorRole.SIGNAL}


def test_float_io_has_no_integer_state_to_tie(model, calibration):
    result = export_model(
        model, ExportSpec(precision="a16w8", io_dtype="float32", mode="keras"), scaled_state(calibration, 1.5)
    )
    assert state_scales_tied(result.inputs, result.outputs) is None


def test_int8_pairs_are_tied_with_their_zero_points(model, calibration):
    int8 = ExportSpec(precision="a8w8", io_dtype="int8", mode="keras", state_tie_tolerance=0.2)
    # Widening only the negative side of state_in_0 moves its zero point and keeps its range covering state_out_0's.
    state = calibration["state_in_0"]
    data = calibration | {"state_in_0": np.where(state < 0, state * np.float32(1.15), state)}
    raw_inputs, raw_outputs = tensor_records(converted(model, int8, data))
    assert pair_params(raw_inputs, 0)[0][1] != pair_params(raw_outputs, 0)[0][1]
    result = export_model(model, int8, data)
    assert state_scales_tied(result.inputs, result.outputs) is True
    assert pair_params(result.inputs, 0) == pair_params(result.outputs, 0) == pair_params(raw_inputs, 0)
    zero_point = pair_params(result.inputs, 0)[0][1]
    assert zero_point != 0
    np.testing.assert_array_equal(LiteRTStreamRunner(result.content).initial_state()["state_in_0"], zero_point)


def test_multi_input_calibration_must_name_every_input(model, calibration):
    with pytest.raises(ValueError, match="mapping of input name"):
        export_model(model, INT16, calibration["signal"])
    with pytest.raises(ValueError, match="do not match the model inputs"):
        export_model(model, INT16, {k: v for k, v in calibration.items() if k != "state_in_1"})
    with pytest.raises(ValueError, match="different numbers of samples"):
        export_model(model, INT16, calibration | {"signal": calibration["signal"][:-1]})
    with pytest.raises(ValueError, match="not calibrated"):
        export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="keras"), calibration)
    with pytest.raises(ValueError, match="NaN"):
        export_model(model, INT16, calibration | {"state_in_0": np.full_like(calibration["state_in_0"], np.nan)})
    with pytest.raises(ValueError, match="does not match model input"):
        export_model(model, INT16, calibration | {"state_in_0": calibration["state_in_0"][:, :-1]})


@pytest.mark.parametrize(("precision", "io_dtype"), [("a8w8", "int8"), ("a16w8", "int16"), ("fp32", "float32")])
def test_a_stateless_export_is_the_converter_output(precision, io_dtype):
    keras.utils.set_random_seed(3)
    a, b = keras.Input((4,), batch_size=1, name="a"), keras.Input((4,), batch_size=1, name="b")
    stateless = keras.Model([a, b], keras.layers.Add()([keras.layers.Dense(3)(a), keras.layers.Dense(3)(b)]))
    spec = ExportSpec(precision=precision, io_dtype=io_dtype, mode="keras")
    rng = np.random.default_rng(4)
    data = {n: rng.normal(size=(8, 4)).astype(np.float32) for n in "ab"} if precision != "fp32" else None
    assert export_model(stateless, spec, data).content == converted(stateless, spec, data)


def test_stream_calibration_needs_a_signal_and_fixed_state_shapes():
    h = state_input(0, (UNITS,), 1)
    with pytest.raises(ValueError, match="at least one input that is not a state"):
        stream_calibration(keras.Model(h, state_output(0, h)), {})
    x, free = keras.Input((FEATURES,), name="signal"), keras.Input((None,), name="state_in_0")
    model = keras.Model([x, free], [keras.layers.Dense(1)(x), state_output(0, free)])
    with pytest.raises(ValueError, match="need fixed shapes"):
        stream_calibration(model, {"signal": np.zeros((2, FEATURES), np.float32)})


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


def litert_interpreter(content):
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_content=content)
    interpreter.allocate_tensors()
    return interpreter


def representable(q, info):
    return (info.min - q[1]) * q[0], (info.max - q[1]) * q[0]


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ((0.0139787, 54), (0.0140062, 55)),
        ((0.01, -3), (0.0101, 4)),
        ((0.02, 127), (0.0199, -128)),
        ((0.5, 0), (0.25, 0)),
    ],
)
def test_the_tied_int8_range_covers_both_ranges(a, b):
    info = np.iinfo(np.int8)
    tied = _covering_quantization(a, b, info)
    low, high = representable(tied, info)
    for q in (a, b):
        q_low, q_high = representable(q, info)
        assert low <= q_low and high >= q_high
    assert info.min <= tied[1] <= info.max


def test_a_symmetric_int16_pair_ties_to_the_larger_scale():
    info = np.iinfo(np.int16)
    assert _covering_quantization((2.5e-5, 0), (2.6e-5, 0), info) == (2.6e-5, 0)


def test_int8_pairs_whose_ranges_overlap_are_tied_to_a_range_covering_both(model, calibration):
    int8 = ExportSpec(precision="a8w8", io_dtype="int8", mode="keras")
    # Shifting state_in_1 by about one step moves both ends of its range, so neither tensor's range covers the other's.
    data = calibration | {"state_in_1": calibration["state_in_1"] + np.float32(0.02)}
    info = np.iinfo(np.int8)
    raw_inputs, raw_outputs = tensor_records(converted(model, int8, data))
    raw = pair_params(raw_inputs, 1) + pair_params(raw_outputs, 1)
    (in_low, in_high), (out_low, out_high) = (representable(q, info) for q in raw)
    assert (in_low - out_low) * (in_high - out_high) > 0
    result = export_model(model, int8, data)
    assert state_scales_tied(result.inputs, result.outputs) is True
    low, high = representable(pair_params(result.inputs, 1)[0], info)
    assert low <= min(in_low, out_low) and high >= max(in_high, out_high)
