"""Names used by heartKIT 64cd51b, sleepKIT 799bf7c and compressionKIT 7fb3663 still resolve.

The list holds every ``helia_edge`` name those kits reference in Python that resolved on main
a342b107, except the names removed in 0.8.0 (the family `XModel` classes).
"""

import importlib

import pytest

keras = pytest.importorskip("keras")
pytest.importorskip("tensorflow")
if keras.backend.backend() != "tensorflow":
    pytest.skip("Kit names include TensorFlow-only converters", allow_module_level=True)

CORE = [
    "helia_edge.callbacks.TQDMProgressBar",
    "helia_edge.converters.tflite.ConversionType",
    "helia_edge.converters.tflite.QuantizationType",
    "helia_edge.converters.tflite.TfLiteKerasConverter",
    "helia_edge.interpreters.tflite.TfLiteKerasInterpreter",
    "helia_edge.layers.EmaResidualVectorQuantizer",
    "helia_edge.layers.ResidualVectorQuantizer",
    "helia_edge.layers.preprocessing.AddSineWave",
    "helia_edge.layers.preprocessing.AmplitudeWarp",
    "helia_edge.layers.preprocessing.AugmentationPipeline",
    "helia_edge.layers.preprocessing.CascadedBiquadFilter",
    "helia_edge.layers.preprocessing.LayerNormalization1D",
    "helia_edge.layers.preprocessing.RandomAugmentation1DPipeline",
    "helia_edge.layers.preprocessing.RandomBackgroundNoises1D",
    "helia_edge.layers.preprocessing.RandomCrop1D",
    "helia_edge.layers.preprocessing.RandomCutout1D",
    "helia_edge.layers.preprocessing.RandomGaussianNoise1D",
    "helia_edge.layers.preprocessing.RandomNoiseDistortion1D",
    "helia_edge.layers.preprocessing.RandomSineWave",
    "helia_edge.layers.preprocessing.Resizing1D",
    "helia_edge.losses.simclr.SimCLRLoss",
    "helia_edge.metrics.MultiF1Score",
    "helia_edge.metrics.Snr",
    "helia_edge.metrics.compute_metrics",
    "helia_edge.metrics.flops.get_flops",
    "helia_edge.metrics.threshold.get_predicted_threshold_indices",
    "helia_edge.models.TcnBlockParams",
    "helia_edge.models.TcnParams",
    "helia_edge.models.UNetModel.model_from_params",
    "helia_edge.models.UNextModel.model_from_params",
    "helia_edge.models.append_layers",
    "helia_edge.models.load_model",
    "helia_edge.trainers.SimCLRTrainer",
    "helia_edge.trainers.SimCLRTrainer.AUG_SAMPLES_0",
    "helia_edge.trainers.SimCLRTrainer.AUG_SAMPLES_1",
    "helia_edge.trainers.SimCLRTrainer.SAMPLES",
    "helia_edge.trainers.VQAutoencoder",
    "helia_edge.utils.ItemFactory",
    "helia_edge.utils.compute_checksum",
    "helia_edge.utils.create_factory",
    "helia_edge.utils.download_file",
    "helia_edge.utils.env_flag",
    "helia_edge.utils.get_output_signature_from_gen",
    "helia_edge.utils.set_random_seed",
    "helia_edge.utils.setup_logger",
    "helia_edge.utils.uniform_id_generator",
]
PLOTTING = [
    "helia_edge.plotting.cm.confusion_matrix_plot",
    "helia_edge.plotting.confusion_matrix_plot",
    "helia_edge.plotting.plot_history_metrics",
    "helia_edge.plotting.px_plot_confusion_matrix",
    "helia_edge.plotting.roc_auc_plot",
]
AWS = ["helia_edge.utils.download_s3_file", "helia_edge.utils.download_s3_objects"]


def resolve(name):
    parts = name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            value = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError as exc:
            if not ".".join(parts[:split]).startswith(exc.name or ""):
                raise
            continue
        for part in parts[split:]:
            value = getattr(value, part)
        return value
    raise ModuleNotFoundError(name)


@pytest.mark.parametrize("name", CORE)
def test_core_names_resolve(name):
    assert resolve(name) is not None


@pytest.mark.parametrize("name", PLOTTING)
def test_plotting_names_resolve(name):
    pytest.importorskip("matplotlib")
    assert resolve(name) is not None


@pytest.mark.parametrize("name", AWS)
def test_aws_names_resolve(name):
    pytest.importorskip("boto3")
    assert resolve(name) is not None
