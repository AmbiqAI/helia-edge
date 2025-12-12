"""
# Contrastive Trainer API

This module provides a trainer for contrastive learning.

Classes:
    ContrastiveTrainer: Trainer for contrastive learning

"""

import keras
import tensorflow as tf
from ..utils import convert_inputs_to_tf_dataset, helia_export


@helia_export(path="helia_edge.trainers.ContrastiveTrainer")
class ContrastiveTrainer(keras.Model):
    SAMPLES = "data"
    LABELS = "labels"
    AUG_SAMPLES_0 = "augmented_data_0"
    AUG_SAMPLES_1 = "augmented_data_1"

    encoder: keras.Model
    augmenters: tuple[keras.Layer, keras.Layer]

    def __init__(
        self,
        encoder: keras.Model,
        projector: keras.Model | tuple[keras.Model, keras.Model],
        augmenter: keras.Layer | tuple[keras.Layer, keras.Layer] | None = None,
        probe: keras.Layer | keras.Model | None = None,
    ):
        """Creates a self-supervised contrastive trainer for a model.

        Args:
            encoder (keras.Model): The encoder model to be trained.
            projector (keras.Model|tuple[keras.Model, keras.Model]): The projector model to be trained.
            augmenter (keras.Layer|tuple[keras.Layer, keras.Layer]|None): The augmenter to be used for data augmentation.
            probe (keras.Layer|keras.Model|None): The probe model to be trained. If None, no probe is used.

        """
        super().__init__()

        if len(encoder.output.shape) != 2:
            raise ValueError(
                f"`encoder` must have a flattened output. Expected "
                f"rank(encoder.output.shape)=2, got "
                f"encoder.output.shape={encoder.output.shape}"
            )

        if isinstance(augmenter, tuple) and len(augmenter) != 2:
            raise ValueError("`augmenter` must be either a single augmenter or a tuple of exactly 2 augmenters.")

        if isinstance(projector, tuple) and len(projector) != 2:
            raise ValueError("`projector` must be either a single augmenter or a tuple of exactly 2 augmenters.")

        if augmenter is None:
            self.augmenters = (keras.layers.Lambda(lambda x: x), keras.layers.Lambda(lambda x: x))
        elif isinstance(augmenter, tuple):
            self.augmenters = augmenter
        else:
            self.augmenters = (augmenter, augmenter)

        self.encoder = encoder

        # Check to see if the projector is being shared or are distinct.
        self._is_shared_projector = True if not isinstance(projector, tuple) else False
        self.projectors = projector if type(projector) is tuple else (projector, projector)
        self.probe = probe

        self.loss_metric = keras.metrics.Mean(name="loss")
        self.encoder_metrics = []
        if probe is not None:
            self.probe_loss_metric = keras.metrics.Mean(name="probe_loss")
            self.probe_metrics = []

    def compile(
        self,
        encoder_optimizer: keras.Optimizer,
        encoder_loss: keras.Loss,
        encoder_metrics: list[keras.Metric] | None = None,
        probe_optimizer: keras.Optimizer | None = None,
        probe_loss: keras.Loss | None = None,
        probe_metrics: list[keras.Metric] | None = None,
        **kwargs,
    ):
        super().compile(
            loss=encoder_loss,
            optimizer=encoder_optimizer,
            metrics=None,
            **kwargs,
        )

        if self.probe and not probe_optimizer:
            raise ValueError("`probe_optimizer` must be specified when a probe is included.")

        if self.probe and not probe_loss:
            raise ValueError("`probe_loss` must be specified when a probe is included.")

        if "loss" in kwargs:
            raise ValueError(
                "`loss` parameter in ContrastiveTrainer.compile is ambiguous. "
                "Please specify `encoder_loss` or `probe_loss`."
            )

        if "optimizer" in kwargs:
            raise ValueError(
                "`optimizer` parameter in ContrastiveTrainer.compile is "
                "ambiguous. Please specify `encoder_optimizer` or "
                "`probe_optimizer`."
            )

        if "metrics" in kwargs:
            raise ValueError(
                "`metrics` parameter in ContrastiveTrainer.compile is "
                "ambiguous. Please specify `encoder_metrics` or "
                "`probe_metrics`."
            )
        self.encoder_metrics = encoder_metrics or []
        if self.probe:
            self.probe_loss = probe_loss
            self.probe_optimizer = probe_optimizer
            self.probe_metrics = probe_metrics or []

    @property
    def metrics(self):
        metrics = [self.loss_metric]
        if self.encoder_metrics:
            metrics += self.encoder_metrics
        if self.probe:
            metrics += [self.probe_loss_metric]
            metrics += self.probe_metrics
        return metrics

    def fit(
        self,
        x=None,
        y=None,
        sample_weight=None,
        batch_size=None,
        validation_data=None,
        **kwargs,
    ):
        # Force training to tf.data and apply augmentations
        train_ds = (
            convert_inputs_to_tf_dataset(x=x, y=y, sample_weight=sample_weight, batch_size=batch_size)
            .map(self.run_augmenters, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        # If validation data is provided, apply augmentations
        if validation_data:
            val_ds = (
                convert_inputs_to_tf_dataset(
                    x=validation_data,
                    batch_size=batch_size,
                )
                .map(self.run_augmenters, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            val_ds = None
        # END IF

        return super().fit(x=train_ds, validation_data=val_ds, **kwargs)

    def run_augmenters(self, x, y=None):
        if isinstance(x, dict):
            inputs = x
        else:
            inputs = {self.SAMPLES: x}

        if y is not None:
            inputs[self.LABELS] = y

        if self.AUG_SAMPLES_0 not in inputs:
            inputs[self.AUG_SAMPLES_0] = self.augmenters[0](x, training=True)
        if self.AUG_SAMPLES_1 not in inputs:
            inputs[self.AUG_SAMPLES_1] = self.augmenters[1](x, training=True)

        return inputs

    def _tensorflow_train_step(self, data):
        samples = data[self.SAMPLES]
        labels = data[self.LABELS] if self.LABELS in data else None
        augmented_samples_0 = data[self.AUG_SAMPLES_0]
        augmented_samples_1 = data[self.AUG_SAMPLES_1]

        with tf.GradientTape() as tape:
            features_0 = self.encoder(augmented_samples_0, training=True)
            features_1 = self.encoder(augmented_samples_1, training=True)

            projections_0 = self.projectors[0](features_0, training=True)
            projections_1 = self.projectors[1](features_1, training=True)

            loss = self.compiled_loss(
                projections_0,
                projections_1,
                regularization_losses=self.encoder.losses,
            )

        # If the projector is shared, then take the trainable weights of just
        # one of the projectors in the tuple. If not, use both the projectors.
        projector_weights = (
            self.projectors[0].trainable_weights
            if self._is_shared_projector
            else self.projectors[0].trainable_weights + self.projectors[1].trainable_weights
        )
        gradients = tape.gradient(
            loss,
            self.encoder.trainable_weights + projector_weights,
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + projector_weights,
            )
        )

        # Update the metrics
        self.loss_metric.update_state(loss)
        for metric in self.encoder_metrics:
            metric.update_state(features_0, features_1)

        if self.probe:
            if labels is None:
                raise ValueError("Targets must be provided when a probe is specified")
            with tf.GradientTape() as tape:
                features = tf.stop_gradient(self.encoder(samples, training=False))
                class_logits = self.probe(features, training=True)
                probe_loss = self.probe_loss(labels, class_logits)
            gradients = tape.gradient(probe_loss, self.probe.trainable_weights)
            self.probe_optimizer.apply_gradients(zip(gradients, self.probe.trainable_weights))
            self.probe_loss_metric.update_state(probe_loss)
            for metric in self.probe_metrics:
                metric.update_state(labels, class_logits)

        return {metric.name: metric.result() for metric in self.metrics}

    def _tensorflow_test_step(self, data):
        # Called by fit
        if isinstance(data, dict):
            labels = data[self.LABELS] if self.LABELS in data else None
            augmented_samples_0 = data[self.AUG_SAMPLES_0]
            augmented_samples_1 = data[self.AUG_SAMPLES_1]
        # Called by evaluate (need to compute augmentations)
        else:
            samples = data
            augmented_samples_0 = self.augmenters[0](samples, training=True)
            augmented_samples_1 = self.augmenters[1](samples, training=True)
            labels = None
        # END IF

        features_0 = self.encoder(augmented_samples_0, training=True)
        features_1 = self.encoder(augmented_samples_1, training=True)

        projections_0 = self.projectors[0](features_0, training=True)
        projections_1 = self.projectors[1](features_1, training=True)

        loss = self.compiled_loss(
            projections_0,
            projections_1,
            regularization_losses=self.encoder.losses,
        )

        # Update the metrics
        self.loss_metric.update_state(loss)
        for metric in self.encoder_metrics:
            metric.update_state(features_0, features_1)

        if self.probe:
            class_logits = self.probe(features_0, training=False)
            probe_loss = self.probe_loss(labels, class_logits)
            self.probe_loss_metric.update_state(probe_loss)
            for metric in self.probe_metrics:
                metric.update_state(labels, class_logits)
        return {metric.name: metric.result() for metric in self.metrics}

    def train_step(self, data):
        if keras.backend.backend() == "tensorflow":
            return self._tensorflow_train_step(data)
        elif keras.backend.backend() == "jax":
            raise NotImplementedError("JAX backend is not supported.")
        elif keras.backend.backend() == "torch":
            raise NotImplementedError("PyTorch backend is not supported.")

    def test_step(self, data):
        if keras.backend.backend() == "tensorflow":
            return self._tensorflow_test_step(data)
        elif keras.backend.backend() == "jax":
            raise NotImplementedError("JAX backend is not supported.")
        elif keras.backend.backend() == "torch":
            raise NotImplementedError("PyTorch backend is not supported.")

    def call(self, inputs):
        raise NotImplementedError("ContrastiveTrainer.call() is not implemented - please call your model directly.")

    @staticmethod
    def linear_probe(num_classes, **kwargs):
        return keras.Sequential(keras.layers.Dense(num_classes), **kwargs)

    def save(self, filepath, overwrite=True, zipped=True, **kwargs):
        """We only save the encoder model"""
        self.encoder.save(filepath, overwrite=overwrite, zipped=zipped, **kwargs)


# class MomentumContrastiveTrainer(ContrastiveTrainer):
#     """Base class for momentum contrastive learning models"""

#     def __init__(
#         self,
#         encoder: keras.Model,
#         projector: keras.Model,
#         contrastive_augmenter: Callable[[tf.Tensor], tf.Tensor] | None = None,
#         classification_augmenter: Callable[[tf.Tensor], tf.Tensor] | None = None,
#         linear_probe: keras.Model | None = None,
#         momentum_coeff: float = 0.999,
#     ):
#         super().__init__(
#             encoder=encoder,
#             projector=projector,
#             contrastive_augmenter=contrastive_augmenter,
#             classification_augmenter=classification_augmenter,
#             linear_probe=linear_probe,
#         )
#         self.momentum_coeff = momentum_coeff

#         # the momentum networks are initialized from their online counterparts
#         self.m_encoder = keras.models.clone_model(self.encoder)
#         self.m_projector = keras.models.clone_model(self.projector)

#     @abstractmethod
#     def contrastive_loss(
#         self,
#         projections_1,
#         projections_2,
#         m_projections_1,
#         m_projections_2,
#     ):  # pylint: disable=arguments-differ
#         pass

#     def train_step(self, data):
#         """Training step for the model"""
#         pair1, pair2 = data

#         # each input is augmented twice, differently
#         augmented_inputs_1 = self.contrastive_augmenter(pair1)
#         augmented_inputs_2 = self.contrastive_augmenter(pair2)

#         with tf.GradientTape() as tape:
#             # Encoder phase
#             features_1 = self.encoder(augmented_inputs_1)
#             features_2 = self.encoder(augmented_inputs_2)
#             # Projection phase
#             projections_1 = self.projector(features_1)
#             projections_2 = self.projector(features_2)
#             # Momentum encoder phase
#             m_features_1 = self.m_encoder(augmented_inputs_1)
#             m_features_2 = self.m_encoder(augmented_inputs_2)
#             # Momentum projection phase
#             m_projections_1 = self.m_projector(m_features_1)
#             m_projections_2 = self.m_projector(m_features_2)
#             contrastive_loss = self.contrastive_loss(projections_1, projections_2, m_projections_1, m_projections_2)
#         # END WITH

#         # backpropagation
#         gradients = tape.gradient(
#             contrastive_loss,
#             self.encoder.trainable_weights + self.projector.trainable_weights,
#         )
#         self.contrastive_optimizer.apply_gradients(
#             zip(
#                 gradients,
#                 self.encoder.trainable_weights + self.projector.trainable_weights,
#             )
#         )
#         self.contrastive_loss_tracker.update_state(contrastive_loss)
#         self.correlation_loss(m_features_1, m_features_2)

#         self.update_contrastive_accuracy(m_features_1, m_features_2)
#         self.update_correlation_accuracy(m_features_1, m_features_2)

#         # labeled_inputs = None
#         # labels = None
#         # preprocessed_inputs = self.classification_augmenter(labeled_inputs)
#         # with tf.GradientTape() as tape:
#         #     # the momentum encoder is used here as it moves more slowly
#         #     features = self.m_encoder(preprocessed_inputs)
#         #     class_logits = self.linear_probe(features)
#         #     probe_loss = self.probe_loss(labels, class_logits)
#         # gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
#         # self.probe_optimizer.apply_gradients(
#         #     zip(gradients, self.linear_probe.trainable_weights)
#         # )
#         # self.probe_accuracy.update_state(labels, class_logits)

#         # the momentum networks are updated by exponential moving average
#         for weight, m_weight in zip(self.encoder.weights, self.m_encoder.weights):
#             m_weight.assign(self.momentum_coeff * m_weight + (1 - self.momentum_coeff) * weight)
#         for weight, m_weight in zip(self.projector.weights, self.m_projector.weights):
#             m_weight.assign(self.momentum_coeff * m_weight + (1 - self.momentum_coeff) * weight)

#         return {m.name: m.result() for m in self.metrics}

#     def test_step(self, data):
#         """Test step for the model"""
#         pair1, pair2 = data
#         augmented_inputs_1 = self.contrastive_augmenter(pair1)
#         augmented_inputs_2 = self.contrastive_augmenter(pair2)
#         features_1 = self.encoder(augmented_inputs_1, training=False)
#         features_2 = self.encoder(augmented_inputs_2, training=False)
#         projections_1 = self.projector(features_1, training=False)
#         projections_2 = self.projector(features_2, training=False)
#         m_features_1 = self.m_encoder(augmented_inputs_1, training=False)
#         m_features_2 = self.m_encoder(augmented_inputs_2, training=False)
#         m_projections_1 = self.m_projector(m_features_1, training=False)
#         m_projections_2 = self.m_projector(m_features_2, training=False)

#         contrastive_loss = self.contrastive_loss(projections_1, projections_2, m_projections_1, m_projections_2)

#         self.contrastive_loss_tracker.update_state(contrastive_loss)
#         self.correlation_loss(m_features_1, m_features_2)

#         self.update_contrastive_accuracy(m_features_1, m_features_2)
#         self.update_correlation_accuracy(m_features_1, m_features_2)

#         return {m.name: m.result() for m in self.metrics}


# class MoCo(MomentumContrastiveModel):
#     """MoCo model for self-supervised learning"""

#     def __init__(
#         self,
#         encoder: keras.Model,
#         projector: keras.Model,
#         contrastive_augmenter: Callable[[tf.Tensor], tf.Tensor] | None = None,
#         classification_augmenter: Callable[[tf.Tensor], tf.Tensor] | None = None,
#         linear_probe: keras.Model | None = None,
#         momentum_coeff: float = 0.999,
#         temperature: float = 0.1,
#         queue_size: int = 65536,
#     ):
#         super().__init__(
#             encoder=encoder,
#             projector=projector,
#             contrastive_augmenter=contrastive_augmenter,
#             classification_augmenter=classification_augmenter,
#             linear_probe=linear_probe,
#             momentum_coeff=momentum_coeff,
#         )
#         self.temperature = temperature

#         feature_dimensions = encoder.output_shape[1]
#         self.feature_queue = tf.Variable(
#             tf.math.l2_normalize(tf.random.normal(shape=(queue_size, feature_dimensions)), axis=1),
#             trainable=False,
#         )

#     def contrastive_loss(
#         self,
#         projections_1,
#         projections_2,
#         m_projections_1,
#         m_projections_2,
#     ):
#         # similar to the SimCLR loss, however it uses the momentum networks'
#         # representations of the differently augmented views as targets
#         projections_1 = tf.math.l2_normalize(projections_1, axis=1)
#         projections_2 = tf.math.l2_normalize(projections_2, axis=1)
#         m_projections_1 = tf.math.l2_normalize(m_projections_1, axis=1)
#         m_projections_2 = tf.math.l2_normalize(m_projections_2, axis=1)

#         similarities_1_2 = (
#             tf.matmul(
#                 projections_1,
#                 tf.concat((m_projections_2, self.feature_queue), axis=0),
#                 transpose_b=True,
#             )
#             / self.temperature
#         )
#         similarities_2_1 = (
#             tf.matmul(
#                 projections_2,
#                 tf.concat((m_projections_1, self.feature_queue), axis=0),
#                 transpose_b=True,
#             )
#             / self.temperature
#         )

#         batch_size = tf.shape(projections_1)[0]
#         contrastive_labels = tf.range(batch_size)
#         loss = keras.losses.sparse_categorical_crossentropy(
#             tf.concat([contrastive_labels, contrastive_labels], axis=0),
#             tf.concat([similarities_1_2, similarities_2_1], axis=0),
#             from_logits=True,
#         )

#         # feature queue update
#         self.feature_queue.assign(
#             tf.concat(
#                 [
#                     m_projections_1,
#                     m_projections_2,
#                     self.feature_queue[: -(2 * batch_size)],
#                 ],
#                 axis=0,
#             )
#         )
#         return loss
