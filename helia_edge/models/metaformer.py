"""
# MetaFormer: Meta-Learning with Transformers

## Overview

MetaFormer is a transformer-based model that incorporates both spatial mixing and channel mixing blocks.
The architecture is designed to learn from few examples and generalize to new tasks.

For more info, refer to the original paper [MetaFormer: Meta-Learning with Transformers](https://arxiv.org/abs/2110.11605).

Classes:
    MetaFormerParams: MetaFormer parameters
    MetaFormerModel: Helper class to generate model from parameters

Functions:
    patch_embedding: Patch embedding layer
    pool_token_mixer: Token mixer using average pooling
    conv_token_mixer: Token mixer using separable convolution
    attention_token_mixer: Token mixer using multi-head attention
    mlp_channel_mixer: Channel mixer using MLP via 1x1 convolutions
    metaformer_block: Metaformer block
    metaformer_layer: Metaformer functional layer

"""

import keras
from pydantic import BaseModel, Field


class NameArgs(BaseModel):
    """Name and arguments

    Attributes:
        name (str): Name
        args (dict): Arguments

    """

    name: str = Field(default="conv", description="Name")
    args: dict = Field(default_factory=dict, description="Arguments")


class MetaFormerBlockParams(BaseModel):
    """MetaFormer block parameters

    Attributes:
        layers (int): Number of layers
        patch_embed (dict): Patch embedding
        token_mixer (NameArgs): Token mixer
        channel_mixer (NameArgs): Channel mixer

    """

    layers: int = Field(default=2, description="Number of layers")
    patch_embed: dict = Field(default_factory=dict, description="Patch embedding")
    token_mixer: NameArgs = Field(default_factory=dict, description="Token mixer")
    channel_mixer: NameArgs = Field(default_factory=dict, description="Channel mixer")


class MetaFormerParams(BaseModel):
    """MetaFormer parameters

    Attributes:
        blocks (list[MetaFormerBlockParams]): MetaFormer blocks
        output_filters (int): Output filters
        output_activation (str | None): Output activation
        include_top (bool): Include top
        dropout (float): Dropout rate
        drop_connect_rate (float): Drop connect rate
        name (str): Model name

    """

    blocks: list[MetaFormerBlockParams] = Field(default_factory=list, description="MetaFormer blocks")
    output_filters: int = Field(default=0, description="Output filters")
    output_activation: str | None = Field(default=None, description="Output activation")
    include_top: bool = Field(default=True, description="Include top")
    dropout: float = Field(default=0.2, description="Dropout rate")
    drop_connect_rate: float = Field(default=0.2, description="Drop connect rate")
    name: str = Field(default="MetaFormer", description="Model name")


def patch_embedding(
    embed_dim: int,
    patch_shape: tuple[int, int],
    stride_shape: tuple[int, int] | None = None,
    padding: str = "same",
) -> keras.layers.Layer:
    """Patch embedding layer using 2D convolution

    Args:
        embed_dim (int): Embedding dimension
        patch_shape (tuple[int, int]): Patch shape
        stride_shape (tuple[int, int], optional): Stride shape. Defaults to None.
        padding (str, optional): Padding. Defaults to 'same'.

    """
    # SHAPE (B, F, T, C) -> (B, F//S, T//S, E)
    # SHAPE (B, H, W, C) -> (B, H//S, W//S, E
    return keras.layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_shape,
        strides=stride_shape or patch_shape,
        padding=padding,
        use_bias=False,
    )


def pool_token_mixer(
    pool_size: tuple[int, int] = (2, 2),
) -> keras.layers.Layer:
    """Token mixer using average pooling

    Args:
        pool_size (tuple[int, int], optional): Pool size. Defaults to (2, 2).

    Returns:
        keras.layers.Layer: Token mixer layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        y = (
            keras.layers.AveragePooling2D(
                pool_size=pool_size,
                strides=(1, 1),
                padding="same",
            )(y)
            - y
        )
        return y

    # END DEF
    return layer


def conv_token_mixer(
    embed_dim: int,
    kernel_size: tuple[int, int] = (3, 3),
    strides: tuple[int, int] = (1, 1),
) -> keras.Layer:
    """Token mixer using separable convolution

    Args:
        embed_dim (int): Embedding dimension
        kernel_size (tuple[int, int], optional): Kernel size. Defaults to (3, 3).
        strides (tuple[int, int], optional): Strides. Defaults to (1, 1).

    Returns:
        keras.layers.Layer: Token mixer layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        y = keras.layers.SeparableConv2D(
            filters=embed_dim,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
        )(y)
        return y

    # END DEF
    return layer


def attention_token_mixer(
    embed_dim: int,
    num_heads: int,
    dropout: float = 0.1,
) -> keras.layers.Layer:
    """Token mixer using multi-head attention

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of heads
        dropout (float, optional): Dropout rate. Defaults to 0.1.

    Returns:
        keras.layers.Layer: Token mixer layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        input_shape = keras.ops.shape(x)
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        y = x  # Shape is B, H, W, C
        y = keras.layers.Reshape((height * width, channels))(y)
        y = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout,
        )(y, y)
        # Reshape back to B, H, W, C
        y = keras.layers.Reshape((height, width, channels))(y)
        return y

    # END DEF
    return layer


def mlp_channel_mixer(
    embed_dim: int,
    ratio: int = 4,
    activation: str = "gelu",
    dropout: float = 0,
) -> keras.Layer:
    """Channel mixer using MLP via 1x1 convolutions

    Args:
        embed_dim (int): Embedding dimension
        ratio (int, optional): Expansion ratio. Defaults to 4.
        activation (str, optional): Activation function. Defaults to "gelu".
        dropout (float, optional): Dropout rate. Defaults to 0.

    Returns:
        keras.layers.Layer: Channel mixer layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        y = keras.layers.Conv2D(
            filters=int(embed_dim * ratio),
            kernel_size=(1, 1),
            activation=activation,
        )(y)

        if 0 < dropout < 1:
            y = keras.layers.Dropout(dropout)(y)

        y = keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=(1, 1),
        )(y)

        if 0 < dropout < 1:
            y = keras.layers.Dropout(dropout)(y)

        return y

    # END DEF
    return layer


def metaformer_block(
    token_mixer: keras.layers.Layer | None = None,
    channel_mixer: keras.layers.Layer | None = None,
    name: str = "mf_block",
) -> keras.layers.Layer:
    """Metaformer block

    Args:
        token_mixer (keras.layers.Layer, optional): Token mixer layer. Defaults to None.
        channel_mixer (keras.layers.Layer, optional): Channel mixer layer. Defaults to None.
        name (str, optional): Block name. Defaults to 'mf_block'.

    Returns:
        keras.layers.Layer: Metaformer block
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        # Shape is B, T, F, C
        y = x

        if token_mixer:
            # Apply layer normalization
            y1 = keras.layers.LayerNormalization(name=f"{name}_token_ln", axis=-1)(y)

            # Apply token mixer
            y1 = token_mixer(y1)

            # Add residual connection
            y = y + y1
        # END IF

        if channel_mixer:
            # Apply layer normalization
            y1 = keras.layers.LayerNormalization(name=f"{name}_ch_ln", axis=-1)(y)

            # Apply channel mixer
            y1 = channel_mixer(y1)

            # Add residual connection
            y = y + y1

        return y

    # END DEF

    return layer


def metaformer_layer(
    x: keras.KerasTensor,
    params: MetaFormerParams,
    num_classes: int | None = None,
) -> keras.KerasTensor:
    """MetaFormer functional layer

    Args:
        x (keras.KerasTensor): Input tensor
        params (MetaFormerParams): Model parameters.
        num_classes (int, optional): Number of classes.

    Returns:
        keras.KerasTensor: Output tensor
    """
    y = x

    for b, block in enumerate(params.blocks):
        print("Adding block", b + 1)
        # Apply patch embedding
        y = patch_embedding(**block.patch_embed)(y)
        for lyr in range(block.layers):
            token_mixer = None
            if block.token_mixer.name == "conv":
                token_mixer = conv_token_mixer(**block.token_mixer.args)
            elif block.token_mixer.name == "pool":
                token_mixer = pool_token_mixer(**block.token_mixer.args)
            elif block.token_mixer.name == "attention":
                token_mixer = attention_token_mixer(**block.token_mixer.args)
            # END IF
            channel_mixer = None
            if block.channel_mixer.name == "mlp":
                channel_mixer = mlp_channel_mixer(**block.channel_mixer.args)
            # END IF
            y = metaformer_block(
                token_mixer=token_mixer,
                channel_mixer=channel_mixer,
                name=f"B{b+1}_L{lyr+1}",
            )(y)
        # END FOR
    # END FOR

    if params.include_top:
        name = "top"
        y = keras.layers.GlobalAveragePooling2D(name=f"{name}_gap")(y)
        if 0 < params.dropout < 1:
            y = keras.layers.Dropout(params.dropout, name=f"{name}_dropout")(y)
        if num_classes is not None:
            y = keras.layers.Dense(num_classes, name=name)(y)
        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)
    # END IF


# def ccaa_metaformer(
#     x: keras.KerasTensor,
#     num_classes: int | None = None,
# ) -> keras.Model:
#     """CCAA Metaformer model"""

#     params = MetaFormerParams(
#         blocks=[
#             MetaFormerBlockParams(
#                 layers=2,
#                 patch_embed=dict(
#                     embed_dim=32,
#                     patch_shape=(7, 7),
#                     stride_shape=(4, 4),
#                 ),
#                 token_mixer=dict(
#                     name="conv",
#                     args=dict(
#                         embed_dim=32,
#                         kernel_size=(3, 3),
#                         strides=(1, 1),
#                     ),
#                 ),
#                 channel_mixer=dict(
#                     name="mlp",
#                     args=dict(
#                         embed_dim=32,
#                         ratio=4,
#                         activation="gelu",
#                         dropout=0.1,
#                     ),
#                 ),
#             ),
#             MetaFormerBlockParams(
#                 layers=2,
#                 patch_embed=dict(
#                     embed_dim=64,
#                     patch_shape=(3, 3),
#                     stride_shape=(2, 2),
#                 ),
#                 token_mixer=dict(
#                     name="conv",
#                     args=dict(
#                         embed_dim=64,
#                         kernel_size=(3, 3),
#                         strides=(1, 1),
#                     ),
#                 ),
#                 channel_mixer=dict(
#                     name="mlp",
#                     args=dict(
#                         embed_dim=64,
#                         ratio=4,
#                         activation="gelu",
#                         dropout=0.1,
#                     ),
#                 ),
#             ),
#             MetaFormerBlockParams(
#                 layers=2,
#                 patch_embed=dict(
#                     embed_dim=128,
#                     patch_shape=(3, 3),
#                     stride_shape=(2, 2),
#                 ),
#                 token_mixer=dict(
#                     name="conv",
#                     args=dict(
#                         embed_dim=128,
#                         kernel_size=(3, 3),
#                         strides=(1, 1),
#                     ),
#                 ),
#                 channel_mixer=dict(
#                     name="mlp",
#                     args=dict(
#                         embed_dim=128,
#                         ratio=4,
#                         activation="gelu",
#                         dropout=0.1,
#                     ),
#                 ),
#             ),
#             MetaFormerBlockParams(
#                 layers=2,
#                 patch_embed=dict(
#                     embed_dim=256,
#                     patch_shape=(3, 2),
#                     stride_shape=(3, 2),
#                 ),
#                 token_mixer=dict(
#                     name="conv",
#                     args=dict(
#                         embed_dim=256,
#                         kernel_size=(3, 3),
#                         strides=(1, 1),
#                     ),
#                 ),
#                 channel_mixer=dict(
#                     name="mlp",
#                     args=dict(
#                         embed_dim=256,
#                         ratio=4,
#                         activation="gelu",
#                         dropout=0.1,
#                     ),
#                 ),
#             ),
#         ],
#         include_top=True,
#         output_activation="softmax",
#     )

#     return metaformer_layer(
#         x=x,
#         params=params,
#         num_classes=num_classes,
#     )


class MetaFormerModel:
    """Helper class to generate model from parameters"""

    @staticmethod
    def layer_from_params(inputs: keras.Input, params: MetaFormerParams | dict, num_classes: int | None = None):
        """Create layer from parameters"""
        if isinstance(params, dict):
            params = MetaFormerParams(**params)
        return metaformer_layer(x=inputs, params=params, num_classes=num_classes)

    @staticmethod
    def model_from_params(inputs: keras.Input, params: MetaFormerParams | dict, num_classes: int | None = None):
        """Create model from parameters"""
        outputs = MetaFormerModel.layer_from_params(inputs=inputs, params=params, num_classes=num_classes)
        return keras.Model(inputs=inputs, outputs=outputs)
