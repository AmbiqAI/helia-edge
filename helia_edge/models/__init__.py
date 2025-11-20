"""
# :material-graph: Models API

When targeting edge devices, it is essential to have a model that is both accurate and efficient.
The model should be able to run in real-time while maintaining a high level of accuracy.
To achieve this, the model must be optimized for the target hardware and the specific use case.
This includes optimizing the model architecture, the input data, and the training process.
While there a number of off the shelf, pre-trained models, these are often too big and too slow for edge applications.
Additionally, they may not be optimized for the specific use case.

The `helia_edge.models` module provides highly parameterized model architectures that can be easily customized to meet the specific requirements of the target hardware and use case.
Most of the models are based on state-of-the-art architectures that have been modified to allow for more fine-grain customization.

## Available Models

- **[TCN](./tcn)**: A CNN leveraging dilated convolutions
- **[U-Net](./unet)**: A CNN with encoder-decoder architecture for segmentation tasks
- **[U-NeXt](./unext)**: A U-Net variant leveraging MBConv blocks
- **[EfficientNetV2](./efficientnet)**: A CNN leveraging MBConv blocks
- **[MobileOne](./mobileone)**: A CNN aimed at sub-1ms inference
- **[ResNet](./resnet)**: A popular CNN often used for vision tasks
- **[Conformer](./conformer)**: A transformer composed of both convolutional and self-attention blocks
- **[MetaFormer](./metaformer)**: A transformer composed of both spatial mixing and channel mixing blocks
- **[TSMixer](./tsmixer)**: An All-MLP Architecture for Time Series

## Usage

A model architecture can easily be instantied by providng a custom set of parameters to the model factory. Each model exposes a set of parameters defined using `Pydantic` to ensure type safety and consistency.


!!! Example

    The following example demonstrates how to create a TCN model using the `Tcn` class. The model is defined using a set of parameters defined in the `TcnParams` and `TcnBlockParams` classes.

    ```python
    import keras
    from helia_edge.models import TcnModel, TcnParams, TcnBlockParams

    inputs = keras.Input(shape=(800, 1))
    num_classes = 5

    model = TcnModel.model_from_params(
        x=inputs,
        params=TcnParams(
            input_kernel=(1, 3),
            input_norm="batch",
            blocks=[
                TcnBlockParams(filters=8, kernel=(1, 3), dilation=(1, 1), dropout=0.1, ex_ratio=1, se_ratio=0, norm="batch"),
                TcnBlockParams(filters=16, kernel=(1, 3), dilation=(1, 2), dropout=0.1, ex_ratio=1, se_ratio=0, norm="batch"),
                TcnBlockParams(filters=24, kernel=(1, 3), dilation=(1, 4), dropout=0.1, ex_ratio=1, se_ratio=4, norm="batch"),
                TcnBlockParams(filters=32, kernel=(1, 3), dilation=(1, 8), dropout=0.1, ex_ratio=1, se_ratio=4, norm="batch"),
            ],
            output_kernel=(1, 3),
            include_top=True,
            use_logits=True,
            model_name="tcn",
        ),
        num_classes=num_classes,
    )
    ```
"""

from . import composer
from . import conformer
from . import convmixer
from . import efficientnet
from . import metaformer
from . import mobileone
from . import mobilenet
from . import regnet
from . import resnet
from . import tcn
from . import tsmixer
from . import unet
from . import unext
from . import utils

from .composer import ComposerModel, ComposerParams, ComposerLayerParams, composer_layer
from .conformer import ConformerModel, ConformerParams, ConformerBlockParams, conformer_layer
from .convmixer import ConvMixerModel, ConvMixerParams, conv_mixer_layer
from .efficientnet import EfficientNetParams, EfficientNetV2Model, efficientnetv2_layer
from .metaformer import MetaFormerModel, MetaFormerParams, MetaFormerBlockParams, metaformer_layer
from .mobileone import MobileOneModel, MobileOneParams, MobileOneBlockParams, mobileone_layer
from .mobilenet import MobileNetV1Model, MobileNetV1Params, mobilenetv1_layer
from .regnet import RegNetModel, RegNetParams, RegNetBlockParam, regnet_layer
from .resnet import ResNetModel, ResNetParams, ResNetBlockParams, resnet_layer
from .tcn import TcnModel, TcnParams, TcnBlockParams, tcn_layer
from .tsmixer import TsMixerModel, TsMixerParams, TsMixerBlockParams, tsmixer_layer
from .unet import UNetModel, UNetParams, UNetBlockParams, unet_layer
from .unext import UNextModel, UNextParams, UNextBlockParams, unext_layer
from .utils import make_divisible, load_model, append_layers
