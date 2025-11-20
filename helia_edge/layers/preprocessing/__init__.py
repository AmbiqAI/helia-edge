"""
# :material-link: Preprocessing Layers API

This module provides a variety of preprocessing/augmentation layers to build custom `tf.data.Dataset` pipelines.
HeliaEdge provides layers for both 1D and 2D input data and doesnt assume 2D input data to be images.
In addition, all layers inherit from `BaseAugmentation` and `TFDataLayer`. These two layers provide the following functionalities:

* Dynamically set backend to TensorFlow for pipeline layers
* Coerce input data to have batch dimension and converted to nested dictionary
* Output data will revert to original format (e.g. no batch)
* By supporting nested dictionary, it allows layers to manipulate labels
* The layers map either sequentially or in parallel across the batch dimension


Classes:
    AmplitudeWarp: Amplitude warping layer
    AugmentationPipeline: Augmentation pipeline
    BaseAugmentation: Base augmentation
    BaseAugmentation1D: Base 1D augmentation
    BaseAugmentation2D: Base 2D augmentation
    CascadedBiquadFilter: Cascaded biquad filter
    FirFilter: FIR filter
    FrequencyMixStyle2D: 2D frequency mix style augmentation
    LayerNormalization1D: Layer normalization 1D
    LayerNormalization2D: Layer normalization 2D
    RandomAugmentation1DPipeline: Random augmentation 1D pipeline
    RandomBackgroundNoises1D: Random background noises 1D
    RandomChannel: Random channel
    RandomChoice: Random choice
    RandomCrop1D: Random crop 1D
    RandomCrop2D: Random crop 2D
    RandomCutout1D: Random cutout 1D
    RandomCutout2D: Random cutout 2D
    RandomFlip2D: Random flip 2D
    RandomGaussianNoise1D: Random Gaussian noise 1D
    RandomNoiseDistortion1D: Random noise distortion 1D
    RandomSineWave: Random sine wave
    Resizing1D: Resizing 1D
    Resizing2D: Resizing 2D
    Rescaling1D: Rescaling 1D
    Rescaling2D: Rescaling 2D
    AddSineWave: Add sine wave
    SpecAugment2D: SpecAugment 2D



"""

from .amplitude_warp import AmplitudeWarp
from .augmentation_pipeline import AugmentationPipeline
from .base_augmentation import BaseAugmentation, BaseAugmentation1D, BaseAugmentation2D
from .biquad_filter import CascadedBiquadFilter
from .defines import NestedTensorType, NestedTensorValue
from .fir_filter import FirFilter
from .frequency_mix_style import FrequencyMixStyle2D
from .layer_normalization import LayerNormalization1D, LayerNormalization2D
from .random_augmentation_pipeline import RandomAugmentation1DPipeline, RandomAugmentation2DPipeline
from .random_background_noises import RandomBackgroundNoises1D
from .random_channel import RandomChannel
from .random_choice import RandomChoice
from .random_crop import RandomCrop1D, RandomCrop2D
from .random_cutout import RandomCutout1D, RandomCutout2D
from .random_flip import RandomFlip2D
from .random_gaussian_noise import RandomGaussianNoise1D
from .random_noise_distortion import RandomNoiseDistortion1D
from .random_sine_wave import RandomSineWave
from .resizing import Resizing1D, Resizing2D
from .rescaling import Rescaling1D, Rescaling2D
from .sine_wave import AddSineWave
from .spec_augment import SpecAugment2D
from .tf_data_layer import TFDataLayer
