"""
# Patching Layers API

This module provides classes to extract patches from 2D data and mask a proportion of them.

Classes:
    PatchLayer2D: Extracts patches from 2D data.
    MaskedPatchEncoder2D: Encodes patches and masks a proportion of them.

"""

import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from ..utils import helia_export


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
        """This layer will extract patches from 2D data (e.g. image) and reshape them into flattened vectors.
        Useful as preprocessing technique for patch-based self-supervised learning methods like
        DINO and Masked Autoencoders. For in-model patching, consider using convolutional layers.

        Args:
            height (int): The height of the data.
            width (int): The width of the data.
            ch (int): The number of channels in the data.
            patch_height (int): The height of the patch.
            patch_width (int): The width of the patch.
        """
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.ch_size = ch
        self.patch_height = patch_height
        self.patch_width = patch_width

        # Each patch will be size (patch_height, patch_width, ch).
        self.resize = keras.layers.Reshape((-1, patch_height * patch_width * ch))

    def call(self, images):
        # Create patches from the input images
        patches = keras.ops.image.extract_patches(
            image=images,
            size=(self.patch_height, self.patch_width),
            strides=(self.patch_height, self.patch_width),
            padding="valid",
        )

        # Reshape the patches to (batch, num_patches, patch_area)
        patches = self.resize(patches)
        return patches

    def show_patched_image(self, images: keras.KerasTensor, patches: keras.KerasTensor) -> int:
        """Utility function which accepts a batch of images and its
        corresponding patches and help visualize one image and its patches
        side by side.

        NOTE: Assumes patch size is divisible by the image size.

        Args:
            images (keras.KerasTensor): A batch of images of shape (B, H, W, C).
            patches (keras.KerasTensor): A batch of patches of shape (B, P, A).

        Returns:
            int: The index of the image that was visualized
        """

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
        """Takes a patch from a *single* image and reconstructs it back into the image.

        NOTE: Assumes patch size is divisible by the image size.

        Args:
            patch (keras.KerasTensor): A patch of shape (P, A).

        Returns:
            keras.KerasTensor: The reconstructed image of shape (H, W, C).

        """
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
        **kwargs,
    ):
        """Given a batch of patches, this layer will
        1. Project the patches and apply positional embeddings.
        2. Mask a proportion of patches.
        3. Return the masked and unmasked patches along with

        Args:
            patch_height (int): The height of the patch.
            patch_width (int): The width of the patch.
            ch_size (int): The number of channels in the patch.
            projection_dim (int): The dimension of the projection layer.
            mask_proportion (float): The proportion of patches to mask.
            downstream (bool, optional): Whether to use the layer in the downstream task. Defaults to False
        """
        super().__init__(**kwargs)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.ch_size = ch_size
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion
        self.downstream = downstream

    def build(self, input_shape):
        (_, self.num_patches, self.patch_area) = input_shape

        # A trainable mask token initialized randomly from a normal distribution
        self.mask_token = self.add_weight(
            shape=(1, self.patch_height * self.patch_width * self.ch_size),
            initializer="random_normal",
            trainable=True,
        )

        # Create the projection layer for the patches
        self.projection = keras.layers.Dense(units=self.projection_dim)

        # Create the positional embedding layer
        self.position_embedding = keras.layers.Embedding(input_dim=self.num_patches, output_dim=self.projection_dim)

        # Number of patches that will be masked
        self.num_mask = int(self.mask_proportion * self.num_patches)

    def call(self, patches):
        batch_size = keras.ops.shape(patches)[0]

        # Get the positional embeddings
        positions = keras.ops.arange(start=0, stop=self.num_patches, step=1)
        positions = keras.ops.expand_dims(positions, axis=0)
        pos_embeddings = self.position_embedding(positions)
        pos_embeddings = keras.ops.tile(pos_embeddings, [batch_size, 1, 1])  # (B, num_patches, projection_dim)

        # Embed the patches.
        patch_embeddings = self.projection(patches) + pos_embeddings  # (B, num_patches, projection_dim)

        if self.downstream:
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            # The encoder input is the unmasked patch embeddings
            unmasked_embeddings = tf.gather(
                patch_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)

            # Get the unmasked and masked position embeddings
            unmasked_positions = tf.gather(
                pos_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)
            masked_positions = tf.gather(
                pos_embeddings, mask_indices, axis=1, batch_dims=1
            )  # (B, mask_numbers, projection_dim)

            # Repeat the mask token number of mask times.
            # Mask tokens replace the masks of the image.
            mask_tokens = keras.ops.repeat(self.mask_token, repeats=self.num_mask, axis=0)
            mask_tokens = keras.ops.expand_dims(mask_tokens, axis=0)
            mask_tokens = keras.ops.repeat(mask_tokens, repeats=batch_size, axis=0)

            # Get the masked embeddings for the tokens.
            masked_embeddings = self.projection(mask_tokens) + masked_positions
            return (
                unmasked_embeddings,  # Input to the encoder.
                masked_embeddings,  # First part of input to the decoder.
                unmasked_positions,  # Added to the encoder outputs.
                mask_indices,  # The indices that were masked.
                unmask_indices,  # The indices that were unmaksed.
            )

    def get_random_indices(self, batch_size: int):
        # Create random indices from a uniform distribution and then split
        # it into mask and unmask indices.
        rand_indices = keras.ops.argsort(keras.random.uniform(shape=(batch_size, self.num_patches)), axis=-1)
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices

    def generate_masked_image(self, patches: keras.KerasTensor, unmask_indices: keras.KerasTensor):
        # Choose a random patch and it corresponding unmask index.
        idx = np.random.choice(patches.shape[0])
        patch = patches[idx]
        unmask_index = unmask_indices[idx]

        # Build a numpy array of same shape as patch.
        new_patch = np.zeros_like(patch)

        for i in range(unmask_index.shape[0]):
            new_patch[unmask_index[i]] = patch[unmask_index[i]]
        return new_patch, idx
