"""Portable image patch extraction and masked patch embeddings."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import keras
import numpy as np

from ..utils import helia_export

if TYPE_CHECKING:
    from .._typing import Tensor


@helia_export(path="helia_edge.layers.PatchLayer2D")
class PatchLayer2D(keras.layers.Layer):
    def __init__(
        self,
        height: int,
        width: int,
        ch: int,
        patch_height: int,
        patch_width: int,
        **kwargs,
    ):
        """Extract flattened patches from images shaped (batch, height, width, ch)."""
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.ch_size = ch
        self.patch_height = patch_height
        self.patch_width = patch_width

        self.resize = keras.layers.Reshape((-1, patch_height * patch_width * ch))

    def call(self, images: Tensor) -> Tensor:
        patches = keras.ops.image.extract_patches(
            images,
            size=(self.patch_height, self.patch_width),
            strides=(self.patch_height, self.patch_width),
            padding="valid",
        )

        patches = self.resize(patches)
        return patches

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "height": self.height,
            "width": self.width,
            "ch": self.ch_size,
            "patch_height": self.patch_height,
            "patch_width": self.patch_width,
        }

    def show_patched_image(self, images: keras.KerasTensor, patches: keras.KerasTensor) -> int:
        """Plot one image and its reconstructed patches; return its batch index."""

        import matplotlib.pyplot as plt

        idx = np.random.choice(patches.shape[0])

        image = images[idx]
        patch = patches[idx]
        reconstructed_image = self.reconstruct_from_patch(patch)

        plt.figure(figsize=(4, 4))
        plt.imshow(keras.utils.array_to_img(image))
        plt.axis("off")
        plt.show()

        plt.figure(figsize=(4, 4))
        plt.imshow(keras.utils.array_to_img(reconstructed_image))
        plt.axis("off")
        plt.show()

        return idx

    def reconstruct_from_patch(self, patch: keras.KerasTensor) -> keras.KerasTensor:
        """Reconstruct one image from non-overlapping patches in row-major order."""
        num_patches = patch.shape[0]
        n = int(self.height / self.patch_height)

        patch = keras.ops.reshape(patch, (num_patches, self.patch_height, self.patch_width, self.ch_size))
        rows = keras.ops.split(patch, n, axis=0)
        rows = [keras.ops.concatenate(keras.ops.unstack(x), axis=1) for x in rows]
        reconstructed = keras.ops.concatenate(rows, axis=0)
        return reconstructed


@helia_export(path="helia_edge.layers.MaskedPatchEncoder2D")
class MaskedPatchEncoder2D(keras.layers.Layer):
    def __init__(
        self,
        patch_height: int,
        patch_width: int,
        ch_size: int,
        projection_dim: int,
        mask_proportion: float,
        downstream: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        """Project patches with position embeddings and sample masks from a seeded stream."""
        super().__init__(**kwargs)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.ch_size = ch_size
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion
        self.downstream = downstream
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)
        if not 0 <= mask_proportion <= 1:
            raise ValueError("mask_proportion must be between 0 and 1")
        self.projection = keras.layers.Dense(units=self.projection_dim)
        self.position_embedding = None

    def build(self, input_shape):
        (_, self.num_patches, self.patch_area) = input_shape

        self.mask_token = self.add_weight(
            shape=(1, self.patch_height * self.patch_width * self.ch_size),
            initializer="random_normal",
            trainable=True,
        )

        self.projection.build(input_shape)

        self.position_embedding = keras.layers.Embedding(input_dim=self.num_patches, output_dim=self.projection_dim)
        self.position_embedding.build((None, self.num_patches))

        self.num_mask = int(self.mask_proportion * self.num_patches)
        if not self.downstream and not 0 < self.num_mask < self.num_patches:
            raise ValueError("mask_proportion must leave at least one masked and one unmasked patch")
        super().build(input_shape)

    def call(self, patches: Tensor) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = keras.ops.shape(patches)[0]

        positions = keras.ops.arange(start=0, stop=self.num_patches, step=1)
        positions = keras.ops.expand_dims(positions, axis=0)
        pos_embeddings = self.position_embedding(positions)
        pos_embeddings = keras.ops.tile(pos_embeddings, [batch_size, 1, 1])  # (B, num_patches, projection_dim)

        patch_embeddings = self.projection(patches) + pos_embeddings  # (B, num_patches, projection_dim)

        if self.downstream:
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            unmasked_embeddings = keras.ops.take_along_axis(
                patch_embeddings, keras.ops.expand_dims(unmask_indices, -1), axis=1
            )  # (B, unmask_numbers, projection_dim)

            unmasked_positions = keras.ops.take_along_axis(
                pos_embeddings, keras.ops.expand_dims(unmask_indices, -1), axis=1
            )  # (B, unmask_numbers, projection_dim)
            masked_positions = keras.ops.take_along_axis(
                pos_embeddings, keras.ops.expand_dims(mask_indices, -1), axis=1
            )  # (B, mask_numbers, projection_dim)

            mask_tokens = keras.ops.repeat(self.mask_token, repeats=self.num_mask, axis=0)
            mask_tokens = keras.ops.expand_dims(mask_tokens, axis=0)
            mask_tokens = keras.ops.repeat(mask_tokens, repeats=batch_size, axis=0)

            masked_embeddings = self.projection(mask_tokens) + masked_positions
            return (
                unmasked_embeddings,  # Input to the encoder.
                masked_embeddings,  # First part of input to the decoder.
                unmasked_positions,  # Added to the encoder outputs.
                mask_indices,  # The indices that were masked.
                unmask_indices,  # The indices that were unmasked.
            )

    def get_random_indices(self, batch_size: int) -> tuple[Tensor, Tensor]:
        rand_indices = keras.ops.argsort(
            keras.random.uniform(shape=(batch_size, self.num_patches), seed=self.seed_generator), axis=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "patch_height": self.patch_height,
            "patch_width": self.patch_width,
            "ch_size": self.ch_size,
            "projection_dim": self.projection_dim,
            "mask_proportion": self.mask_proportion,
            "downstream": self.downstream,
            "seed": self.seed,
        }

    def generate_masked_image(self, patches: keras.KerasTensor, unmask_indices: keras.KerasTensor):
        idx = np.random.choice(patches.shape[0])
        patch = patches[idx]
        unmask_index = unmask_indices[idx]

        new_patch = np.zeros_like(patch)

        for i in range(unmask_index.shape[0]):
            new_patch[unmask_index[i]] = patch[unmask_index[i]]
        return new_patch, idx
