"""Native TF/Torch optimization of an EDGE component, without compile()/fit().

KERAS_BACKEND=torch python examples/masked_autoencoder_native.py
KERAS_BACKEND=tensorflow python examples/masked_autoencoder_native.py
Use --evidence output.npz for a fixed-mask, fixed-weight numerical parity fixture.
"""

import argparse
import os
from typing import cast

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import keras
import numpy as np

from helia_edge.layers import MaskedPatchEncoder2D, PatchLayer2D
from helia_edge.trainers import MaskedAutoencoder


class FixedMaskEncoder(MaskedPatchEncoder2D):
    """Controlled logical masks for cross-backend comparison only."""

    def get_random_indices(self, batch_size):
        return tuple(
            keras.ops.tile(keras.ops.convert_to_tensor([indices], dtype="int32"), (batch_size, 1))
            for indices in ([0, 2], [1, 3])
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence")
    args = parser.parse_args()
    patch_encoder_type = FixedMaskEncoder if args.evidence else MaskedPatchEncoder2D
    model = MaskedAutoencoder(
        PatchLayer2D(2, 2, 1, 1, 1),
        patch_encoder_type(1, 1, 1, 2, 0.5, seed=17),
        keras.Sequential([keras.Input((2, 2)), keras.layers.Dense(2)]),
        keras.Sequential(
            [keras.Input((4, 2)), keras.layers.Flatten(), keras.layers.Dense(4), keras.layers.Reshape((2, 2, 1))]
        ),
    )
    x = np.arange(12, dtype="float32").reshape(3, 2, 2, 1) / 12
    # Build before creating a native optimizer so all parameters are registered.
    model.reconstruction_targets(x)
    for i, variable in enumerate(model.trainable_weights):
        variable.assign(
            np.linspace(-0.1, 0.1, np.prod(variable.shape), dtype="float32").reshape(variable.shape) + i * 0.01
        )
    if keras.backend.backend() == "tensorflow":
        import tensorflow as tf

        optimizer = keras.optimizers.SGD(0.01)
        with tf.GradientTape() as tape:
            targets, predicted = model.reconstruction_targets(cast(tf.Tensor, tf.constant(x)), training=True)
            loss = tf.reduce_mean(tf.math.squared_difference(targets, predicted))
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    elif keras.backend.backend() == "torch":
        import torch

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        targets, predicted = model.reconstruction_targets(torch.tensor(x), training=True)
        loss = torch.mean((cast(torch.Tensor, targets) - cast(torch.Tensor, predicted)) ** 2)
        loss.backward()
        gradients = [v.value.grad for v in model.trainable_weights]
        optimizer.step()
    else:
        raise ValueError("Select KERAS_BACKEND=tensorflow or torch before starting Python")
    assert all(g is not None for g in gradients)
    arrays = {
        "targets": keras.ops.convert_to_numpy(targets),
        "predictions": keras.ops.convert_to_numpy(predicted),
        "loss": keras.ops.convert_to_numpy(loss),
    }
    for i, (gradient, variable) in enumerate(zip(gradients, model.trainable_weights)):
        arrays[f"gradient_{i}"] = keras.ops.convert_to_numpy(keras.ops.convert_to_tensor(gradient))
        arrays[f"updated_weight_{i}"] = keras.ops.convert_to_numpy(variable)
    assert all(np.isfinite(value).all() for value in arrays.values())
    if args.evidence:
        np.savez(args.evidence, **arrays)
    print(f"{keras.backend.backend()} native optimizer step: loss={float(arrays['loss']):.8f}")


if __name__ == "__main__":
    main()
