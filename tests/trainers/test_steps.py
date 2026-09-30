"""gradient_step applies one optimizer update with the active backend's autodiff."""

import keras
import numpy as np
import pytest

from helia_edge.trainers import NotSupported, gradient_step, no_grad, require_backend


def test_gradient_step_is_one_sgd_update_of_a_linear_model():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 3)).astype(np.float32)
    y = rng.standard_normal((8, 1)).astype(np.float32)
    w0 = rng.standard_normal((3, 1)).astype(np.float32)
    inputs = keras.Input((3,))
    model = keras.Model(inputs, keras.layers.Dense(1, use_bias=False)(inputs))
    model.layers[1].set_weights([w0])
    model.compile(optimizer=keras.optimizers.SGD(0.1), loss="mse")

    target = keras.ops.convert_to_tensor(y)

    def loss_fn():
        prediction = model(x, training=True)
        return keras.ops.mean(keras.ops.square(prediction - target)), prediction

    loss, prediction = gradient_step(model, loss_fn)
    expected_w = w0 - 0.1 * (2 / len(x)) * x.T @ (x @ w0 - y)
    np.testing.assert_allclose(model.layers[1].get_weights()[0], expected_w, rtol=1e-5, atol=1e-6)
    loss = float(keras.ops.convert_to_numpy(keras.ops.stop_gradient(loss)))
    np.testing.assert_allclose(loss, float(np.mean((x @ w0 - y) ** 2)), rtol=1e-5)
    assert tuple(keras.ops.shape(prediction)) == (8, 1)


def test_variables_without_gradients_are_skipped():
    inputs = keras.Input((2,))
    used, unused = keras.layers.Dense(1), keras.layers.Dense(1)
    model = keras.Model(inputs, [used(inputs), unused(inputs)])
    model.compile(optimizer=keras.optimizers.SGD(0.5))
    before = [w.copy() for w in unused.get_weights()]
    x = np.ones((2, 2), np.float32)
    gradient_step(model, lambda: (keras.ops.mean(model(x)[0]),))
    for a, b in zip(before, unused.get_weights(), strict=True):
        np.testing.assert_array_equal(a, b)


def test_unsupported_backends_raise_not_supported():
    with pytest.raises(NotSupported, match="does not support the") as info:
        require_backend("feature", ("not-a-backend",))
    assert isinstance(info.value, NotImplementedError)
    assert require_backend("feature", (keras.backend.backend(),)) == keras.backend.backend()


def test_no_grad_is_a_context_manager():
    with no_grad():
        assert keras.ops.convert_to_numpy(keras.ops.ones((2,))).sum() == 2


def test_variables_created_by_the_first_call_are_updated():
    class Lazy(keras.Model):
        def build(self, input_shape):
            self.dense = keras.layers.Dense(1)
            self.dense.build(input_shape)

        def call(self, x):
            return self.dense(x)

    model = Lazy()
    model.compile(optimizer=keras.optimizers.SGD(0.1))
    x = np.ones((4, 2), np.float32)
    assert model.trainable_weights == []
    gradient_step(model, lambda: (keras.ops.mean(model(x)),))
    kernel = keras.ops.convert_to_numpy(model.dense.kernel)
    assert model.trainable_weights and not np.allclose(kernel, 0) and model.optimizer.iterations == 1
