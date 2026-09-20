"""Behavioral checks run in independent TF-only and Torch-only environments."""

import json
import subprocess
import sys

import keras
import numpy as np
import pytest

from helia_edge.layers import MaskedPatchEncoder2D, PatchLayer2D
from helia_edge.trainers import MaskedAutoencoder


def model_fixture(stateful=False, build=True):
    patch = PatchLayer2D(2, 2, 1, 1, 1)
    patch_encoder = MaskedPatchEncoder2D(1, 1, 1, 2, 0.5, seed=17)
    encoder = keras.Sequential([keras.Input((2, 2)), keras.layers.Dense(2)])
    if stateful:
        encoder.add(keras.layers.BatchNormalization())
        encoder.add(keras.layers.Dropout(0.5, seed=19))
    decoder = keras.Sequential(
        [keras.Input((4, 2)), keras.layers.Flatten(), keras.layers.Dense(4), keras.layers.Reshape((2, 2, 1))]
    )
    model = MaskedAutoencoder(patch, patch_encoder, encoder, decoder)
    x = np.arange(12, dtype="float32").reshape(3, 2, 2, 1) / 12
    if build:
        model(x)
    return model, x


def fix_mask(model):
    def indices(batch_size):
        return (
            keras.ops.tile(keras.ops.convert_to_tensor([[0, 2]], dtype="int32"), (batch_size, 1)),
            keras.ops.tile(keras.ops.convert_to_tensor([[1, 3]], dtype="int32"), (batch_size, 1)),
        )

    model.patch_encoder.get_random_indices = indices


def test_forward_without_compile_and_exact_target_selection():
    model, x = model_fixture()
    fix_mask(model)
    targets, predictions = model.reconstruction_targets(x)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(targets), x.reshape(3, 4, 1)[:, [0, 2]])
    assert tuple(predictions.shape) == (3, 2, 1)
    assert len(model.trainable_weights) > 0


def test_optimizer_updates_and_evaluation_preserves_weights():
    model, x = model_fixture(stateful=True)
    model.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse", metrics=["mae"], jit_compile=False)
    fix_mask(model)
    before = [keras.ops.convert_to_numpy(v).copy() for v in model.trainable_weights]
    logs = model.train_step(keras.ops.convert_to_tensor(x))
    assert {"loss", "mae"} <= logs.keys()
    assert all(np.isfinite(keras.ops.convert_to_numpy(v)).all() for v in logs.values())
    assert any(not np.array_equal(a, keras.ops.convert_to_numpy(b)) for a, b in zip(before, model.trainable_weights))
    weights = model.get_weights()
    model.test_step(keras.ops.convert_to_tensor(x))
    for before, after in zip(weights, model.get_weights()):
        np.testing.assert_array_equal(before, after)
    assert int(keras.ops.convert_to_numpy(model.optimizer.iterations)) == 1


def test_fit_and_evaluate():
    model, x = model_fixture(build=False)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"], jit_compile=False)
    history = model.fit(x, batch_size=2, epochs=2, verbose=0)
    assert np.isfinite(history.history["loss"]).all()
    assert int(keras.ops.convert_to_numpy(model.optimizer.iterations)) == 4
    logs = model.evaluate(x, batch_size=2, return_dict=True, verbose=0)
    assert {"loss", "mae"} <= logs.keys()


def test_native_loop_uses_native_optimizer_without_compile():
    model, x = model_fixture()
    fix_mask(model)
    before = model.get_weights()
    if keras.backend.backend() == "tensorflow":
        import tensorflow as tf

        optimizer = tf.keras.optimizers.SGD(0.01)
        with tf.GradientTape() as tape:
            targets, predicted = model.reconstruction_targets(tf.constant(x), training=True)
            loss = tf.reduce_mean(tf.square(targets - predicted))
        gradients = tape.gradient(loss, model.trainable_weights)
        assert all(g is not None for g in gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    else:
        import torch

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        targets, predicted = model.reconstruction_targets(torch.tensor(x), training=True)
        loss = torch.mean((targets - predicted) ** 2)
        loss.backward()
        assert all(v.value.grad is not None for v in model.trainable_weights)
        optimizer.step()
    assert any(not np.array_equal(a, b) for a, b in zip(before, model.get_weights()))


def test_gradients_are_cleared_between_steps():
    model, x = model_fixture()
    fix_mask(model)
    model.compile(optimizer=keras.optimizers.SGD(0.0), loss="mse", jit_compile=False)
    model.train_step(keras.ops.convert_to_tensor(x))
    if keras.backend.backend() == "torch":
        gradients = [v.value.grad.clone() for v in model.trainable_weights]
        model.train_step(keras.ops.convert_to_tensor(x))
        for before, variable in zip(gradients, model.trainable_weights):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(before), keras.ops.convert_to_numpy(variable.value.grad)
            )


def test_safe_save_reload_and_optimizer_state(tmp_path):
    model, x = model_fixture()
    model.compile(optimizer="adam", loss="mse", jit_compile=False)
    model.train_step(keras.ops.convert_to_tensor(x))
    model.save(tmp_path / "mae.keras")
    np.save(tmp_path / "weights.npy", np.concatenate([v.reshape(-1) for v in model.get_weights()]))
    np.save(
        tmp_path / "optimizer.npy",
        np.concatenate([keras.ops.convert_to_numpy(v).reshape(-1) for v in model.optimizer.variables]),
    )
    model.patch_encoder.seed_generator.state.assign([17, 0])
    np.save(tmp_path / "predictions.npy", keras.ops.convert_to_numpy(model.reconstruction_targets(x)[1]))
    source = """
import sys
from pathlib import Path
import keras
import numpy as np
from helia_edge.models import load_model
path = Path(sys.argv[1])
model = load_model(path / 'mae.keras')
np.testing.assert_allclose(np.concatenate([v.reshape(-1) for v in model.get_weights()]), np.load(path / 'weights.npy'))
assert int(keras.ops.convert_to_numpy(model.optimizer.iterations)) == 1
np.testing.assert_allclose(
    np.concatenate([keras.ops.convert_to_numpy(v).reshape(-1) for v in model.optimizer.variables]),
    np.load(path / 'optimizer.npy'))
x = np.arange(12, dtype='float32').reshape(3, 2, 2, 1) / 12
model.patch_encoder.seed_generator.state.assign([17, 0])
np.testing.assert_allclose(keras.ops.convert_to_numpy(model.reconstruction_targets(x)[1]),
                          np.load(path / 'predictions.npy'), atol=1e-6)
result = model.train_step(keras.ops.convert_to_tensor(x))
assert int(keras.ops.convert_to_numpy(model.optimizer.iterations)) == 2
assert np.isfinite(keras.ops.convert_to_numpy(result['loss']))
"""
    result = subprocess.run([sys.executable, "-c", source, str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_mask_rng_advances_and_seed_reproduces_sequence():
    left = MaskedPatchEncoder2D(1, 1, 1, 2, 0.5, seed=29)
    right = MaskedPatchEncoder2D(1, 1, 1, 2, 0.5, seed=29)
    left.build((None, 20, 1))
    right.build((None, 20, 1))
    first = keras.ops.convert_to_numpy(left.get_random_indices(4)[0])
    second = keras.ops.convert_to_numpy(left.get_random_indices(4)[0])
    np.testing.assert_array_equal(first, keras.ops.convert_to_numpy(right.get_random_indices(4)[0]))
    np.testing.assert_array_equal(second, keras.ops.convert_to_numpy(right.get_random_indices(4)[0]))
    assert not np.array_equal(first, second)
    json.dumps(left.get_config())


@pytest.mark.parametrize("proportion", [-0.1, 1.1, 0, 1, 0.01])
def test_invalid_mask_sizes(proportion):
    with pytest.raises(ValueError, match="mask_proportion"):
        layer = MaskedPatchEncoder2D(1, 1, 1, 2, proportion)
        layer.build((None, 4, 1))


def test_plain_callable_compatibility():
    model, x = model_fixture()
    patch = model.patch_layer
    model = MaskedAutoencoder(lambda inputs: patch(inputs), model.patch_encoder, model.encoder, model.decoder)
    targets, predictions = model.reconstruction_targets(x)
    assert targets.shape == predictions.shape


def test_explicit_target_and_weight_policy():
    model, x = model_fixture()
    model.compile(optimizer="sgd", loss="mse")
    with pytest.raises(ValueError, match="without y"):
        model.train_step((x, x))
    with pytest.raises(ValueError, match="sample_weight"):
        model.train_step((x, None, np.ones(3)))


def test_training_flag_controls_nested_batchnorm_and_dropout():
    model, x = model_fixture(stateful=True)
    fix_mask(model)
    left = keras.ops.convert_to_numpy(model.reconstruction_targets(x, training=False)[1])
    right = keras.ops.convert_to_numpy(model.reconstruction_targets(x, training=False)[1])
    np.testing.assert_array_equal(left, right)
    bn = model.encoder.layers[1]
    before = keras.ops.convert_to_numpy(bn.moving_mean).copy()
    model.reconstruction_targets(x, training=True)
    assert not np.array_equal(before, keras.ops.convert_to_numpy(bn.moving_mean))
    after = keras.ops.convert_to_numpy(bn.moving_mean).copy()
    model.reconstruction_targets(x, training=False)
    np.testing.assert_array_equal(after, keras.ops.convert_to_numpy(bn.moving_mean))


def test_loss_metric_reports_objective_not_mean_target():
    model, x = model_fixture()
    fix_mask(model)
    model.compile(optimizer=keras.optimizers.SGD(0.0), loss="mse", metrics=["mae"], jit_compile=False)
    targets, predictions = model.reconstruction_targets(x)
    expected = keras.ops.convert_to_numpy(keras.ops.mean(keras.ops.square(targets - predictions)))
    logs = model.train_step(keras.ops.convert_to_tensor(x))
    np.testing.assert_allclose(keras.ops.convert_to_numpy(logs["loss"]), expected, rtol=1e-6)
