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
    Normalization1D: Mean/variance normalization 1D
    Normalization2D: Mean/variance normalization 2D
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

from helia_edge._lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "AmplitudeWarp": (".amplitude_warp", "AmplitudeWarp"),
        "AugmentationPipeline": (".augmentation_pipeline", "AugmentationPipeline"),
        "BaseAugmentation": (".base_augmentation", "BaseAugmentation"),
        "BaseAugmentation1D": (".base_augmentation", "BaseAugmentation1D"),
        "BaseAugmentation2D": (".base_augmentation", "BaseAugmentation2D"),
        "CascadedBiquadFilter": (".biquad_filter", "CascadedBiquadFilter"),
        "NestedTensorType": (".defines", "NestedTensorType"),
        "NestedTensorValue": (".defines", "NestedTensorValue"),
        "FirFilter": (".fir_filter", "FirFilter"),
        "FrequencyMixStyle2D": (".frequency_mix_style", "FrequencyMixStyle2D"),
        "LayerNormalization1D": (".layer_normalization", "LayerNormalization1D"),
        "LayerNormalization2D": (".layer_normalization", "LayerNormalization2D"),
        "Normalization1D": (".normalization", "Normalization1D"),
        "Normalization2D": (".normalization", "Normalization2D"),
        "RandomAugmentation1DPipeline": (".random_augmentation_pipeline", "RandomAugmentation1DPipeline"),
        "RandomAugmentation2DPipeline": (".random_augmentation_pipeline", "RandomAugmentation2DPipeline"),
        "RandomBackgroundNoises1D": (".random_background_noises", "RandomBackgroundNoises1D"),
        "RandomChannel": (".random_channel", "RandomChannel"),
        "RandomChoice": (".random_choice", "RandomChoice"),
        "RandomCrop1D": (".random_crop", "RandomCrop1D"),
        "RandomCrop2D": (".random_crop", "RandomCrop2D"),
        "RandomCutout1D": (".random_cutout", "RandomCutout1D"),
        "RandomCutout2D": (".random_cutout", "RandomCutout2D"),
        "RandomFlip2D": (".random_flip", "RandomFlip2D"),
        "RandomGaussianNoise1D": (".random_gaussian_noise", "RandomGaussianNoise1D"),
        "RandomNoiseDistortion1D": (".random_noise_distortion", "RandomNoiseDistortion1D"),
        "RandomSineWave": (".random_sine_wave", "RandomSineWave"),
        "Resizing1D": (".resizing", "Resizing1D"),
        "Resizing2D": (".resizing", "Resizing2D"),
        "Rescaling1D": (".rescaling", "Rescaling1D"),
        "Rescaling2D": (".rescaling", "Rescaling2D"),
        "AddSineWave": (".sine_wave", "AddSineWave"),
        "SpecAugment2D": (".spec_augment", "SpecAugment2D"),
        "TFDataLayer": (".tf_data_layer", "TFDataLayer"),
        "amplitude_warp": (".amplitude_warp", None),
        "augmentation_pipeline": (".augmentation_pipeline", None),
        "base_augmentation": (".base_augmentation", None),
        "biquad_filter": (".biquad_filter", None),
        "defines": (".defines", None),
        "fir_filter": (".fir_filter", None),
        "frequency_mix_style": (".frequency_mix_style", None),
        "layer_normalization": (".layer_normalization", None),
        "normalization": (".normalization", None),
        "random_augmentation_pipeline": (".random_augmentation_pipeline", None),
        "random_background_noises": (".random_background_noises", None),
        "random_channel": (".random_channel", None),
        "random_choice": (".random_choice", None),
        "random_crop": (".random_crop", None),
        "random_cutout": (".random_cutout", None),
        "random_flip": (".random_flip", None),
        "random_gaussian_noise": (".random_gaussian_noise", None),
        "random_noise_distortion": (".random_noise_distortion", None),
        "random_sine_wave": (".random_sine_wave", None),
        "resizing": (".resizing", None),
        "rescaling": (".rescaling", None),
        "sine_wave": (".sine_wave", None),
        "spec_augment": (".spec_augment", None),
        "tf_data_layer": (".tf_data_layer", None),
    },
)
