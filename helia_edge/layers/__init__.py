"""
# :material-layers: Layers API

The `helia_edge.layers` module provides classes to build neural network layers.
For example, you can use the `helia_edge.layers.preprocessing.AmplitudeWarp` layer to apply amplitude warping to audio signals.

## Available Layers

* **[Preprocessing Layers](./preprocessing)**: Provides `tf.data.Dataset` preprocessing layers.
* **[Activations](./activations)**: Provides activation functions.
* **[Convolutional Layers](./convolutional)**: Provides convolutional layers.
* **[Gumbel Softmax Bottleneck](./gumbel_softmax_bottleneck)**: Provides Gumbel Softmax bottleneck layer.
* **[Normalization Layers](./normalization)**: Provides normalization layers.
* **[Patching Layers](./patching)**: Provides patching layers.
* **[Residual Vector Quantizer](./residual_vector_quantizer)**: Provides residual vector quantizer layer.
* **[Squeeze-and-Excitation Layer](./squeeze_excite)**: Provides squeeze-and-excitation layers.
* **[MBConv Layer](./mbconv)**: Provides popular mbconv block.
* **[Vector Quantizer](./vector_quantizer)**: Provides vector quantizer layer.


"""

from . import preprocessing
from . import activations
from . import convolutional
from . import mbconv
from . import normalization
from . import patching
from . import squeeze_excite

from .activations import swish, glu, relu, relu6, sigmoid, mish, gelu
from .convolutional import conv1d, conv2d
from .gumbel_softmax_bottleneck import GumbelSoftmaxBottleneck
from .mbconv import mbconv_block, MBConvParams
from .normalization import batch_normalization, layer_normalization
from .patching import PatchLayer2D, MaskedPatchEncoder2D
from .residual_vector_quantizer import ResidualVectorQuantizer
from .squeeze_excite import se_layer
from .vector_quantizer import VectorQuantizer
