"""Tensor annotations; import under TYPE_CHECKING to preserve backend isolation."""

from typing import Any

import keras
import numpy as np
import tensorflow as tf
import torch


type Tensor = keras.KerasTensor | tf.Tensor | torch.Tensor
type Array = Tensor | np.ndarray[Any, Any]
