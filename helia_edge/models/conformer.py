"""
# Conformer Model

Conformer model implementation in Keras.

Classes:
    SubsampleBlockParams: Subsample block parameters
    ConformerBlockParams: Conformer block parameters
    ConformerParams: Conformer parameters
    ConformerModel: Helper class to generate model from parameters

Functions:
    subsampler: Subsampler block
    fc_block: Fully connected block
    conv_block: Convolutional block
    att_block: Attention block
    conformer_block: Conformer block
    conformer_layer: Conformer functional layer

"""

import keras
from pydantic import BaseModel, Field

from ..layers.normalization import layer_normalization, batch_normalization
from ..layers.activations import swish, glu, relu


class SubsampleBlockParams(BaseModel):
    """Subsample block parameters

    Attributes:
        depth (int): Depth
        kernel_size (int): Kernel size
        strides (int): Stride size
    """

    depth: int = 256
    kernel_size: int = 3
    strides: int = 2


class ConformerBlockParams(BaseModel):
    """Conformer block parameters

    Attributes:
        depth (int): Depth
        fc_ex_factor (float): FC expansion factor
        fc_res_factor (float): FC residual factor
        embedding (str): Embedding type
        num_heads (int): Number of heads
        kernel_size (int): Kernel size
        dropout (float): Dropout rate
        use_bias (bool): Use bias
    """

    depth: int = 256
    fc_ex_factor: float = 4
    fc_res_factor: float = 0.5
    embedding: str = "relative"
    num_heads: int = 4
    kernel_size: int = 9
    dropout: float = 0.1
    use_bias: bool = True


class ConformerParams(BaseModel):
    """Conformer parameters

    Attributes:
        subsamples (list[SubsampleBlockParams]): Subsample blocks
        blocks (list[ConformerBlockParams]): Conformer blocks
        output_activation (str | None): Output activation
        include_top (bool): Include top
        name (str): Model name

    """

    subsamples: list[SubsampleBlockParams] = Field(default_factory=list, description="Subsample blocks")
    blocks: list[ConformerBlockParams] = Field(default_factory=list, description="Conformer blocks")
    output_activation: str | None = Field(default=None, description="Output activation")
    include_top: bool = Field(default=True, description="Include top")
    name: str = Field(default="Fast Conformer", description="Model name")


def subsampler(
    blocks: SubsampleBlockParams,
    kernel_initializer: str = "glorot_uniform",
    bias_initializer: str = "zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    name: str | None = None,
) -> keras.Layer:
    """Subsampler block

    Args:
        blocks (SubsampleBlockParams): Subsample block parameters
        kernel_initializer (str, optional): Kernel initializer. Defaults to "glorot_uniform".
        bias_initializer (str, optional): Bias initializer. Defaults to "zeros".
        kernel_regularizer ([type], optional): Kernel regularizer. Defaults to None.
        bias_regularizer ([type], optional): Bias regularizer. Defaults to None.
        name (str, optional): Name. Defaults to None.

    Returns:
        keras.Layer: Subsampler layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        # Apply subsampling blocks
        for i, block in enumerate(blocks):
            y = keras.layers.SeparableConv2D(
                filters=block.depth,
                kernel_size=block.kernel_size,
                strides=block.strides,
                depthwise_initializer=kernel_initializer,
                pointwise_initializer=kernel_initializer,
                depthwise_regularizer=kernel_regularizer,
                pointwise_regularizer=kernel_regularizer,
                bias_initializer=bias_initializer,
                bias_regularizer=bias_regularizer,
                padding="same",
                name=f"{name}_conv{i+1}",
            )(y)
            y = relu(name=f"{name}_relu{i+1}")(y)
        # END FOR

        # Swap from (b,f,t,c) to (b,t,c,f) and merge (c,f) to get (b,t,c*f)
        y = keras.ops.transpose(y, axes=[0, 2, 3, 1])
        y = keras.layers.Reshape((y.shape[1], -1))(y)

        # Can serve as learnable embedding
        y = keras.layers.Dense(
            units=blocks[-1].depth,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=f"{name}_dense",
        )(y)  # (b,t,c*f) -> (b,t,d)
        return y

    # END DEF
    return layer


def fc_block(
    depth: int,
    ex_factor: int = 4,
    residual_factor: float = 0.5,
    dropout: float = 0,
    use_bias: bool = True,
    kernel_initializer: str = "glorot_uniform",
    bias_initializer: str = "zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    name: str = "fc_block",
) -> keras.Layer:
    """Fully connected block

    Args:
        depth (int): Depth
        ex_factor (int, optional): Expansion factor. Defaults to 4.
        residual_factor (float, optional): Residual factor. Defaults to 0.5.
        dropout (float, optional): Dropout rate. Defaults to 0.
        use_bias (bool, optional): Use bias. Defaults to True.
        kernel_initializer (str, optional): Kernel initializer. Defaults to "glorot_uniform".
        bias_initializer (str, optional): Bias initializer. Defaults to "zeros".
        kernel_regularizer ([type], optional): Kernel regularizer. Defaults to None.
        bias_regularizer ([type], optional): Bias regularizer. Defaults to None.
        name (str, optional): Name. Defaults to "fc_block".

    Returns:
        keras.Layer: Functional layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x

        y = layer_normalization(name=f"{name}_ln")(y)

        y = keras.layers.Dense(
            int(ex_factor * depth),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            use_bias=use_bias,
            name=f"{name}_fc1",
        )(y)
        y = swish(name=f"{name}_swish")(y)
        if 0 < dropout < 1:
            y = keras.layers.Dropout(dropout, name=f"{name}_dropout1")(y)
        # END IF
        y = keras.layers.Dense(
            depth,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            use_bias=use_bias,
            name=f"{name}_fc2",
        )(y)
        if 0 < dropout < 1:
            y = keras.layers.Dropout(dropout, name=f"{name}_dropout2")(y)
        # END IF
        return x + residual_factor * y

    # END DEF
    return layer


def conv_block(
    depth: int,
    kernel_size: int = 9,
    dropout: float = 0.0,
    padding: str = "same",
    scale_factor: int = 2,
    kernel_initializer: str = "glorot_uniform",
    bias_initializer: str = "zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    name: str = "conv_module",
) -> keras.Layer:
    """Convolutional block

    Args:
        depth (int): Depth
        kernel_size (int, optional): Kernel size. Defaults to 9.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        padding (str, optional): Padding. Defaults to "same".
        scale_factor (int, optional): Scale factor. Defaults to 2.
        kernel_initializer (str, optional): Kernel initializer. Defaults to "glorot_uniform".
        bias_initializer (str, optional): Bias initializer. Defaults to "zeros".
        kernel_regularizer ([type], optional): Kernel regularizer. Defaults to None.
        bias_regularizer ([type], optional): Bias regularizer. Defaults to None.
        name (str, optional): Name. Defaults to "conv_module".

    Returns:
        keras.Layer: Functional layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x

        y = layer_normalization(name=f"{name}_ln")(y)

        y = keras.layers.Conv1D(
            filters=scale_factor * depth,
            kernel_size=1,
            strides=1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_pw_conv1",
        )(y)

        y = glu(name=f"{name}_glu")(y)

        y = keras.layers.DepthwiseConv1D(
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            depthwise_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_dw_conv",
        )(y)
        y = batch_normalization(name=f"{name}_bn")(y)

        y = swish(name=f"{name}_swish")(y)

        y = keras.layers.Conv1D(
            filters=depth,
            kernel_size=1,
            strides=1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_pw_conv2",
        )(y)

        if 0 < dropout < 1:
            y = keras.layers.Dropout(rate=dropout, name=f"{name}_dropout")(y)
        # END IF

        # Residual connection
        return x + y

    # END DEF
    return layer


def att_block(
    depth: int,
    embedding: str = "rel",
    num_heads: int = 4,
    dropout: float = 0.1,
    name: str = "att_block",
) -> keras.Layer:
    """Attention block

    Args:
        depth (int): Depth
        embedding (str, optional): Embedding type. Defaults to "rel".
        num_heads (int, optional): Number of heads. Defaults to 4.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        name (str, optional): Name. Defaults to "att_block".

    Returns:
        keras.Layer: Functional layer

    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        y = layer_normalization(name=f"{name}_ln")(y)

        # Att type: rel, abs, none
        y = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=depth // num_heads,
            value_dim=depth // num_heads,
            dropout=dropout,
            name=f"{name}_mha",
        )(y, y)

        if 0 < dropout < 1:
            y = keras.layers.Dropout(dropout, name=f"{name}_dropout")(y)

        return y

    # END DEF
    return layer


def conformer_block(
    depth: int,
    fc_ex_factor: int = 4,
    fc_res_factor: int = 0.5,
    embedding: str = "relative",
    num_heads: int = 4,
    kernel_size: int = 9,
    dropout: float = 0.1,
    use_bias: bool = True,
    name: str = "cf_block",
) -> keras.Layer:
    """Conformer block

    Args:
        depth (int): Depth
        fc_ex_factor (int, optional): FC expansion factor. Defaults to 4.
        fc_res_factor (int, optional): FC residual factor. Defaults to 0.5.
        embedding (str, optional): Embedding type. Defaults to "relative".
        num_heads (int, optional): Number of heads. Defaults to 4.
        kernel_size (int, optional): Kernel size. Defaults to 9.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        use_bias (bool, optional): Use bias. Defaults to True.
        name (str, optional): Name. Defaults to "cf_block".

    Returns:
        keras.Layer: Functional layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x

        # 1st FC Block
        y = fc_block(
            depth=depth,
            ex_factor=fc_ex_factor,
            residual_factor=fc_res_factor,
            dropout=dropout,
            use_bias=use_bias,
            name=f"{name}_fcb1",
        )(y)

        # ATT Block w/ residual
        y = att_block(
            depth=depth,
            embedding=embedding,
            num_heads=num_heads,
            dropout=dropout,
            name=f"{name}_att",
        )(y)

        # Conv Block w/ residual
        y = conv_block(depth=depth, kernel_size=kernel_size, dropout=dropout, name=f"{name}_cvb")(y)

        # 2nd FC Block w/ residual
        y = fc_block(
            depth=depth,
            ex_factor=fc_ex_factor,
            residual_factor=fc_res_factor,
            dropout=dropout,
            use_bias=use_bias,
            name=f"{name}_fcb2",
        )(y)

        # Output Layer Norm
        y = layer_normalization(name=f"{name}_out_ln")(y)

        return y

    # END DEF
    return layer


def conformer_layer(
    x: keras.KerasTensor,
    params: ConformerParams,
    num_classes: int | None = None,
) -> keras.KerasTensor:
    """Conformer functional layer

    Args:
        x (keras.KerasTensor): Input tensor
        params (ConformerParams): Model parameters.
        num_classes (int, optional): Number of classes.

    Returns:
        keras.KerasTensor: Output tensor
    """
    y = x

    y = subsampler(
        blocks=params.subsamples,
        kernel_size=params.subsample.kernel_size,
        strides=params.subsample.strides,
        num_downsamples=params.subsample.downsamples,
        name="subsample",
    )(y)

    for i, block in enumerate(params.blocks):
        y = conformer_block(
            depth=block.depth,
            fc_ex_factor=block.fc_ex_factor,
            fc_res_factor=block.fc_res_factor,
            embedding=block.embedding,
            num_heads=block.num_heads,
            kernel_size=block.kernel_size,
            dropout=block.dropout,
            use_bias=block.use_bias,
            name=f"CB{i+1}",
        )(y)

    if params.include_top:
        name = "top"
        y = keras.layers.GlobalAveragePooling1D(name=f"{name}_gap")(y)
        if num_classes is not None:
            y = keras.layers.Dense(num_classes, name=name)(y)
        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)
    # END IF

    return y


class ConformerModel:
    """Helper class to generate model from parameters"""

    @staticmethod
    def layer_from_params(inputs: keras.Input, params: ConformerParams | dict, num_classes: int | None = None):
        """Create layer from parameters"""
        if isinstance(params, dict):
            params = ConformerParams(**params)
        return conformer_layer(x=inputs, params=params, num_classes=num_classes)

    @staticmethod
    def model_from_params(inputs: keras.Input, params: ConformerParams | dict, num_classes: int | None = None):
        """Create model from parameters"""
        outputs = ConformerModel.layer_from_params(inputs=inputs, params=params, num_classes=num_classes)
        return keras.Model(inputs=inputs, outputs=outputs)
