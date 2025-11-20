"""
# SimCLR Trainer API

This module contains the implementation of a SimCLR trainer that can be used to train a model using the SimCLR approach.

Classes:
    SimCLRTrainer: A trainer for SimCLR

"""

import keras

from .contrastive import ContrastiveTrainer
from ..utils import helia_export


@helia_export(path="helia_edge.trainers.SimCLRTrainer")
class SimCLRTrainer(ContrastiveTrainer):
    def __init__(
        self,
        encoder: keras.Model,
        projector: keras.Model | None = None,
        **kwargs,
    ):
        """Creates a SimCLRTrainer.

        If no projector is provided, a default one will be created based on paper.

        References:
            - [SimCLR paper](https://arxiv.org/pdf/2002.05709)

        Args:
            encoder (keras.Model): The encoder model.
            projector (keras.Model, optional): The projector model. Defaults to None.
        """
        if projector is None:
            projection_width = encoder.output_shape[-1]
            projector = keras.Sequential(
                [
                    keras.layers.Dense(projection_width, activation="relu"),
                    keras.layers.Dense(projection_width),
                    keras.layers.BatchNormalization(),
                ],
                name="projector",
            )

        super().__init__(
            encoder=encoder,
            projector=projector,
            **kwargs,
        )

    def compile(
        self,
        encoder_optimizer: keras.Optimizer,
        encoder_loss: keras.Loss | None = None,
        encoder_metrics: list[keras.Metric] | None = None,
        probe_optimizer: keras.Optimizer | None = None,
        probe_loss: keras.Loss | None = None,
        probe_metrics: list[keras.Metric] | None = None,
        **kwargs,
    ):
        super().compile(
            encoder_loss=encoder_loss,
            encoder_optimizer=encoder_optimizer,
            encoder_metrics=encoder_metrics,
            probe_optimizer=probe_optimizer,
            probe_loss=probe_loss,
            probe_metrics=probe_metrics,
            **kwargs,
        )
