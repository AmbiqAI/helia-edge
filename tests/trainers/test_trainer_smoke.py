"""Smoke tests for trainers without dedicated tests: compile, fit and evaluate on tiny data."""

import keras
import numpy as np
import pytest

import helia_edge as helia
from helia_edge.layers.gumbel_softmax_bottleneck import GumbelSoftmaxBottleneck
from helia_edge.layers.vector_quantizer import VectorQuantizer

TF = keras.backend.backend() == "tensorflow"
CONTRASTIVE = ["ContrastiveTrainer", "SimCLRTrainer"]


def data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(8, 16, 4)).astype(np.float32), rng.integers(0, 3, size=(8,)).astype(np.int32)


def flat_encoder():
    inputs = keras.Input((16, 4))
    return keras.Model(inputs, keras.layers.Dense(8)(keras.layers.Flatten()(inputs)), name="encoder")


def classifier():
    return keras.Sequential([keras.Input((16, 4)), keras.layers.Flatten(), keras.layers.Dense(3)])


def contrastive(x, y):
    trainer = helia.trainers.ContrastiveTrainer(
        encoder=flat_encoder(), projector=keras.Sequential([keras.layers.Dense(8)])
    )
    trainer.compile(
        encoder_optimizer=keras.optimizers.Adam(1e-3), encoder_loss=helia.losses.simclr.SimCLRLoss(temperature=0.1)
    )
    return trainer, (x,), {"loss"}


def simclr(x, y):
    trainer = helia.trainers.SimCLRTrainer(encoder=flat_encoder())
    trainer.compile(
        encoder_optimizer=keras.optimizers.Adam(1e-3), encoder_loss=helia.losses.simclr.SimCLRLoss(temperature=0.1)
    )
    return trainer, (x,), {"loss"}


def distiller(x, y):
    trainer = helia.trainers.Distiller(student=classifier(), teacher=classifier())
    trainer.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
    )
    return trainer, (x, y), {"loss", "sparse_categorical_accuracy"}


def autoencoder_parts():
    encoder = keras.Sequential([keras.Input((16, 4)), keras.layers.Dense(8)])
    decoder = keras.Sequential([keras.Input((16, 8)), keras.layers.Dense(4)])
    return encoder, decoder


def gs_autoencoder(x, y):
    encoder, decoder = autoencoder_parts()
    gs = GumbelSoftmaxBottleneck(num_embeddings=6, embedding_dim=8)
    trainer = helia.trainers.GSAutoencoder(encoder=encoder, gs=gs, decoder=decoder)
    trainer.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.MeanSquaredError())
    return trainer, (x, x), {"loss", "gs_perplexity", "gs_usage", "gs_bits_per_index", "gs_temperature"}


def vq_autoencoder(x, y):
    encoder, decoder = autoencoder_parts()
    vq = VectorQuantizer(num_embeddings=6, embedding_dim=8)
    trainer = helia.trainers.VQAutoencoder(encoder=encoder, vq=vq, decoder=decoder)
    trainer.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.MeanSquaredError())
    return trainer, (x, x), {"loss", "vq_perplexity", "vq_usage", "vq_bits_per_index"}


TRAINERS = [
    pytest.param(
        contrastive, id="contrastive", marks=pytest.mark.skipif(not TF, reason="TensorFlow only until the Torch port")
    ),
    pytest.param(simclr, id="simclr", marks=pytest.mark.skipif(not TF, reason="TensorFlow only until the Torch port")),
    pytest.param(distiller, id="distiller"),
    pytest.param(gs_autoencoder, id="gs_autoencoder"),
    pytest.param(vq_autoencoder, id="vq_autoencoder"),
]


@pytest.mark.parametrize("make", TRAINERS)
def test_fits_and_evaluates(make):
    keras.backend.clear_session()
    keras.utils.set_random_seed(0)
    trainer, inputs, metrics = make(*data())
    history = trainer.fit(*inputs, batch_size=4, epochs=1, verbose=0)
    assert set(history.history) == metrics
    assert all(np.isfinite(values[-1]) for values in history.history.values())
    result = trainer.evaluate(*inputs, batch_size=4, verbose=0, return_dict=True)
    assert set(result) == metrics
    assert all(np.isfinite(value) for value in result.values())


@pytest.mark.parametrize("name", CONTRASTIVE)
def test_contrastive_trainers_refuse_other_backends(name):
    """Current behaviour, pinned until the Torch port replaces it."""
    if TF:
        pytest.skip("Runs on the TensorFlow backend")
    with pytest.raises(ImportError, match=f"helia_edge.trainers.{name} requires tensorflow"):
        getattr(helia.trainers, name)
