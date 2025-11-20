"""
# Distiller Trainer API

This module contains the implementation of a distiller trainer that can be used to train a student model

Classes:
    Distiller: A trainer for distillation

"""

import keras

from ..utils import helia_export


@helia_export(path="helia_edge.trainers.Distiller")
class Distiller(keras.Model):
    teacher: keras.models.Model
    student: keras.models.Model

    def __init__(self, student: keras.models.Model, teacher: keras.models.Model):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer: keras.optimizers.Optimizer,
        metrics: list[keras.metrics.Metric],
        student_loss_fn: keras.losses.Loss,
        distillation_loss_fn: keras.losses.Loss,
        alpha: float = 0.1,
        temperature: float = 3,
    ):
        """Configure the distiller.

        Args:
            optimizer (keras.optimizers.Optimizer): Keras optimizer for the student weights
            metrics (list[keras.metrics.Metric]): Keras metrics for evaluation
            student_loss_fn (keras.losses.Loss): Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn (keras.losses.Loss): Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha (float, optional): weight to student_loss_fn and 1-alpha to distillation_loss_fn. Defaults to 0.1.
            temperature (float, optional): Temperature for softening probability distributions. Defaults to 3.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False):
        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(
            keras.ops.softmax(teacher_pred / self.temperature, axis=1),
            keras.ops.softmax(y_pred / self.temperature, axis=1),
        ) * (self.temperature**2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, x):
        return self.student(x)
