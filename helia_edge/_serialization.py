"""Explicit registration for loading heliaEDGE custom Keras objects."""

import importlib


def register_keras_serializables():
    """Register supported EDGE objects before calling Keras's safe load_model.

    Imports Keras and the selected backend. TF-only augmentation/trainer objects
    are registered only on TensorFlow. Does not enable unsafe deserialization.
    """
    import keras

    modules = [
        "helia_edge.callbacks.tqdm_progress_bar",
        "helia_edge.layers.ema_residual_vector_quantizer",
        "helia_edge.layers.gumbel_softmax_bottleneck",
        "helia_edge.layers.residual_vector_quantizer",
        "helia_edge.layers.vector_quantizer",
        "helia_edge.layers.patching",
        "helia_edge.losses.simclr",
        "helia_edge.metrics.confusion_matrix",
        "helia_edge.metrics.fscore",
        "helia_edge.metrics.prd",
        "helia_edge.metrics.snr",
        "helia_edge.trainers.distiller",
        "helia_edge.trainers.mask_autoencoder",
    ]
    if keras.backend.backend() == "tensorflow":
        modules += [
            "helia_edge.layers.preprocessing.amplitude_warp",
            "helia_edge.layers.preprocessing.augmentation_pipeline",
            "helia_edge.layers.preprocessing.base_augmentation",
            "helia_edge.layers.preprocessing.biquad_filter",
            "helia_edge.layers.preprocessing.fir_filter",
            "helia_edge.layers.preprocessing.frequency_mix_style",
            "helia_edge.layers.preprocessing.layer_normalization",
            "helia_edge.layers.preprocessing.normalization",
            "helia_edge.layers.preprocessing.random_augmentation_pipeline",
            "helia_edge.layers.preprocessing.random_background_noises",
            "helia_edge.layers.preprocessing.random_channel",
            "helia_edge.layers.preprocessing.random_choice",
            "helia_edge.layers.preprocessing.random_crop",
            "helia_edge.layers.preprocessing.random_cutout",
            "helia_edge.layers.preprocessing.random_flip",
            "helia_edge.layers.preprocessing.random_gaussian_noise",
            "helia_edge.layers.preprocessing.random_noise_distortion",
            "helia_edge.layers.preprocessing.random_sine_wave",
            "helia_edge.layers.preprocessing.rescaling",
            "helia_edge.layers.preprocessing.resizing",
            "helia_edge.layers.preprocessing.sine_wave",
            "helia_edge.layers.preprocessing.spec_augment",
            "helia_edge.layers.preprocessing.tf_data_layer",
            "helia_edge.trainers.contrastive",
            "helia_edge.trainers.simclr",
        ]
    for module in modules:
        importlib.import_module(module)
