"""
# :material-layers: Layers API

The `helia_edge.layers` module provides classes to build neural network layers.
For example, you can use the `helia_edge.layers.preprocessing.AmplitudeWarp` layer to apply amplitude warping to audio signals.

## Available Layers

* **[Preprocessing Layers](./preprocessing)**: Provides `tf.data.Dataset` preprocessing layers.
* **[Activations](./activations)**: Provides activation functions.
* **[Convolutional Layers](./convolutional)**: Provides convolutional layers.
* **[EMA Residual Vector Quantizer](./ema_residual_vector_quantizer)**: Provides EMA-based residual vector quantizer layer.
* **[Gumbel Softmax Bottleneck](./gumbel_softmax_bottleneck)**: Provides Gumbel Softmax bottleneck layer.
* **[Normalization Layers](./normalization)**: Provides normalization layers.
* **[Patching Layers](./patching)**: Provides patching layers.
* **[Residual Vector Quantizer](./residual_vector_quantizer)**: Provides residual vector quantizer layer.
* **[Squeeze-and-Excitation Layer](./squeeze_excite)**: Provides squeeze-and-excitation layers.
* **[MBConv Layer](./mbconv)**: Provides popular mbconv block.
* **[Vector Quantizer](./vector_quantizer)**: Provides vector quantizer layer.


"""

from helia_edge._lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "preprocessing": (".preprocessing", None),
        "activations": (".activations", None),
        "convolutional": (".convolutional", None),
        "mbconv": (".mbconv", None),
        "normalization": (".normalization", None),
        "patching": (".patching", None),
        "squeeze_excite": (".squeeze_excite", None),
        "swish": (".activations", "swish"),
        "glu": (".activations", "glu"),
        "relu": (".activations", "relu"),
        "relu6": (".activations", "relu6"),
        "sigmoid": (".activations", "sigmoid"),
        "mish": (".activations", "mish"),
        "gelu": (".activations", "gelu"),
        "conv1d": (".convolutional", "conv1d"),
        "conv2d": (".convolutional", "conv2d"),
        "EmaResidualVectorQuantizer": (".ema_residual_vector_quantizer", "EmaResidualVectorQuantizer"),
        "GumbelSoftmaxBottleneck": (".gumbel_softmax_bottleneck", "GumbelSoftmaxBottleneck"),
        "mbconv_block": (".mbconv", "mbconv_block"),
        "MBConvParams": (".mbconv", "MBConvParams"),
        "batch_normalization": (".normalization", "batch_normalization"),
        "layer_normalization": (".normalization", "layer_normalization"),
        "PatchLayer2D": (".patching", "PatchLayer2D"),
        "MaskedPatchEncoder2D": (".patching", "MaskedPatchEncoder2D"),
        "ResidualVectorQuantizer": (".residual_vector_quantizer", "ResidualVectorQuantizer"),
        "se_layer": (".squeeze_excite", "se_layer"),
        "VectorQuantizer": (".vector_quantizer", "VectorQuantizer"),
        "ema_residual_vector_quantizer": (".ema_residual_vector_quantizer", None),
        "gumbel_softmax_bottleneck": (".gumbel_softmax_bottleneck", None),
        "residual_vector_quantizer": (".residual_vector_quantizer", None),
        "vector_quantizer": (".vector_quantizer", None),
    },
)
