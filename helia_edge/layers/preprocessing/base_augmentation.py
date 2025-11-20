"""
# Base Augmentation API

Classes:
    BaseAugmentation: Base augmentation
    BaseAugmentation1D: Base 1D augmentation
    BaseAugmentation2D: Base 2D augmentation

Functions:
    tf_keras_map: Map function for TensorFlow Keras
"""

from typing import Callable
import keras

from .tf_data_layer import TFDataLayer
from .defines import NestedTensorValue
from ...utils import helia_export


def tf_keras_map(f, xs):
    # NOTE: Workaround until (https://github.com/keras-team/keras/issues/20048)
    import tensorflow as tf

    xs = keras.tree.map_structure(tf.convert_to_tensor, xs)

    def get_fn_output_signature(x):
        out = f(x)
        return keras.tree.map_structure(tf.TensorSpec.from_tensor, out)

    # Grab single element unpacking and repacking single element
    xe = tf.nest.pack_sequence_as(xs, [y[0] for y in tf.nest.flatten(xs)])

    fn_output_signature = get_fn_output_signature(xe)
    return tf.map_fn(f, xs, fn_output_signature=fn_output_signature)


@helia_export(path="helia_edge.layers.preprocessing.BaseAugmentation")
class BaseAugmentation(TFDataLayer):
    SAMPLES = "data"
    LABELS = "labels"
    TARGETS = "targets"
    ALL_KEYS = (SAMPLES, LABELS, TARGETS)
    TRANSFORMS = "transforms"
    IS_DICT = "is_dict"
    BATCHED = "is_batched"
    USE_TARGETS = "use_targets"
    NDIMS = 4  # Modify in subclass (includes batch size)

    def __init__(
        self,
        seed: int | None = None,
        auto_vectorize: bool = True,
        data_format: str | None = None,
        name: str | None = None,
        **kwargs,
    ):
        """BaseAugmentation acts as a base class for various custom augmentation layers.
        This class provides a common interface for augmenting samples and labels. In the future, we will
        add support for segmentation and bounding boxes.

        The only method that needs to be implemented by the subclass is

        - augment_sample: Augment a single sample during training.

        Optionally, you can implement the following methods:

        - augment_label: Augment a single label during training.
        - get_random_transformations: Returns a nested structure of random transformations that should be applied to the batch.
            This is required to have unique transformations for each sample in the batch and maintain the same transformations for samples and labels.
        - batch_augment: Augment a batch of samples and labels during training. Needed if layer requires access to all samples (e.g. CutMix).

        By default, this method will coerce the input into a batch as well as a nested structure of inputs.
        If auto_vectorize is set to True, the augment_sample and augment_label methods will be vectorized using keras.ops.vectorized_map.
        Otherwise, it will use keras.ops.map which runs sequentially.

        Args:
            seed (int | None): Random seed. Defaults to None.
            auto_vectorize (bool): If True, augment_sample and augment_label methods will be vectorized using keras.ops.vectorized_map.
                Otherwise, it will use keras.ops.map which runs sequentially. Defaults to True.
            data_format (str | None): Data format. Defaults to None. Will use keras.backend.image_data_format() if None.
            name (str | None): Layer name. Defaults to None.

        """
        super().__init__(name=name, **kwargs)

        self.generator = keras.random.SeedGenerator(seed)
        self.data_format = data_format or keras.backend.image_data_format()

        # This is needed for compatibility with tf.data.Dataset pipeline
        self._allow_non_tensor_positional_args = True
        self.built = True
        self._convert_input_args = False

        self.training = True
        self.auto_vectorize = auto_vectorize

    @property
    def random_generator(self) -> keras.random.SeedGenerator:
        return self._get_seed_generator(self.backend._backend)

    def _map_fn(
        self, func: Callable[[NestedTensorValue], keras.KerasTensor], inputs: NestedTensorValue
    ) -> keras.KerasTensor:
        """Calls appropriate mapping function with given inputs.

        Args:
            func (Callable): Function to be mapped.
            inputs (dict): Dictionary containing inputs.

        Returns:
            KerasTensor: Augmented samples or labels
        """
        if self.auto_vectorize:
            return keras.ops.vectorized_map(func, inputs)
        # NOTE: Workaround until (https://github.com/keras-team/keras/issues/20048)
        if keras.backend.backend() == "tensorflow":
            return tf_keras_map(func, inputs)
        return keras.ops.map(func, inputs)

    def call(self, inputs: NestedTensorValue, training: bool = True) -> NestedTensorValue:
        """This method will serve as the main entry point for the layer. It will handle the input formatting and output formatting.

        Args:
            inputs (NestedTensorValue): Inputs to be augmented.
            training (bool): Whether the model is training or not.

        Returns:
            NestedTensorValue: Augmented samples or labels.
        """
        self.training = training
        inputs, metadata = self._format_inputs(inputs)
        return self._format_outputs(self.batch_augment(inputs), metadata)

    def augment_sample(self, inputs: NestedTensorValue) -> keras.KerasTensor:
        """Augment a single sample during training.

        !!! note

                This method should be implemented by the subclass.
        Args:
            input(NestedTensorValue): Single sample.

        Returns:
            KerasTensor: Augmented sample.
        """
        return inputs[self.SAMPLES]

    def augment_samples(self, inputs: NestedTensorValue) -> keras.KerasTensor:
        """Augment a batch of samples during training.

        Args:
            inputs (NestedTensorValue): Batch of samples.

        Returns:
            KerasTensor: Augmented batch of samples.
        """
        return self._map_fn(self.augment_sample, inputs=inputs)

    def augment_label(self, inputs: NestedTensorValue) -> keras.KerasTensor:
        """Augment a single label during training.

        !!! note

            Implement this method if you need to augment labels.

        Args:
            inputs (NestedTensorValue): Single label.

        Returns:
            keras.KerasTensor: Augmented label.
        """
        return inputs[self.LABELS]

    def augment_labels(self, inputs: NestedTensorValue) -> keras.KerasTensor:
        """Augment a batch of labels during training.

        Args:
            inputs (NestedTensorValue): Batch of labels.

        Returns:
            keras.KerasTensor: Augmented batch of labels.
        """
        return self._map_fn(self.augment_label, inputs=inputs)

    def get_random_transformations(self, input_shape: tuple[int, ...]) -> NestedTensorValue:
        """Generates random transformations needed for augmenting samples and labels.

        Args:
            input_shape (tuple[int,...]): Shape of the input (N, ...).

        Returns:
            NestedTensorValue: Batch of random transformations.

        !!! note
                This method should be implemented by the subclass if the layer requires random transformations.
        """
        return keras.ops.arange(input_shape[0])

    def batch_augment(self, inputs: NestedTensorValue) -> NestedTensorValue:
        """Handles processing entire batch of samples and labels in a nested structure.
        Responsible for calling augment_samples and augment_labels.

        Args:
            inputs (NestedTensorValue): Batch of samples and labels.

        Returns:
            NestedTensorValue: Augmented batch of samples and labels.
        """
        samples = inputs.get(self.SAMPLES, None)
        labels = inputs.get(self.LABELS, None)
        result = {}

        transformations = self.get_random_transformations(input_shape=keras.ops.shape(samples))

        result[self.SAMPLES] = self.augment_samples(inputs={self.SAMPLES: samples, self.TRANSFORMS: transformations})

        if labels is not None:
            result[self.LABELS] = self.augment_labels(inputs={self.LABELS: labels, self.TRANSFORMS: transformations})
        # END IF

        # preserve any additional inputs unmodified by this layer.
        for key in inputs.keys() - result.keys():
            result[key] = inputs[key]
        return result

    def _format_inputs(self, inputs: NestedTensorValue) -> tuple[NestedTensorValue, dict[str, bool]]:
        """Validate and force inputs to be batched and placed in structured format.

        Args:
            inputs (NestedTensorValue): Inputs to be formatted.

        Returns:
            tuple[NestedTensorValue, dict[str, bool]]: Formatted inputs and metadata.

        """
        metadata = {self.IS_DICT: True, self.USE_TARGETS: False, self.BATCHED: True}
        if not isinstance(inputs, dict):
            inputs = {self.SAMPLES: inputs}
            metadata[self.IS_DICT] = False

        samples = inputs.get(self.SAMPLES, None)
        if inputs.get(self.SAMPLES) is None:
            raise ValueError(f"Expect the inputs to have key {self.SAMPLES}. Got keys: {list(inputs.keys())}")
        # END IF
        if inputs[self.SAMPLES].shape.rank != self.NDIMS - 1 and samples.shape.rank != self.NDIMS:
            raise ValueError(f"Invalid input shape: {samples.shape}")
        # END IF
        if inputs[self.SAMPLES].shape.rank == self.NDIMS - 1:
            metadata[self.BATCHED] = False
            # Expand dims to make it batched for keys of interest
            for key in set(self.ALL_KEYS).intersection(inputs.keys()):
                if inputs[key] is not None:
                    inputs[key] = keras.ops.expand_dims(inputs[key], axis=0)
                # END IF
            # END FOR
        # END IF
        return inputs, metadata

    def _format_outputs(self, output: NestedTensorValue, metadata: dict[str, bool]) -> NestedTensorValue:
        """Format the output to match the initial input format.

        Args:
            output: Output to be formatted.
            metadata: Metadata used for formatting.

        Returns:
            Output in the original format.
        """
        if not metadata[self.BATCHED]:
            for key in set(self.ALL_KEYS).intersection(output.keys()):
                if output[key] is not None:  # check if tensor
                    output[key] = keras.ops.squeeze(output[key], axis=0)
                # END IF
            # END FOR
        # END IF
        if not metadata[self.IS_DICT]:
            return output[self.SAMPLES]
        if metadata[self.USE_TARGETS]:
            output[self.TARGETS] = output[self.LABELS]
            del output[self.LABELS]
        return output

    def compute_output_shape(self, input_shape: tuple[int, ...], *args, **kwargs) -> tuple[int, ...]:
        """By default assumes the shape of the input is the same as the output.

        Args:
            input_shape (tuple[int,...]): Input shape.

        Returns:
            tuple[int,...]: Output shape.

        !!! note
                This method should be implemented by the subclass if the output shape is different from the input shape.
        """
        return input_shape

    def get_config(self):
        """Serialize the layer configuration."""
        config = super().get_config()
        config.update(
            {
                "seed": self.seed,
                "auto_vectorize": self.auto_vectorize,
                "data_format": self.data_format,
            }
        )
        return config


@helia_export(path="helia_edge.layers.preprocessing.BaseAugmentation1D")
class BaseAugmentation1D(BaseAugmentation):
    NDIMS = 3  # (N, T, C) or (N, C, T)

    def __init__(self, **kwargs):
        """BaseAugmentation1D acts as a base class for various custom augmentation layers.
        This class provides a common interface for augmenting samples and labels. In the future, we will
        add support for segmentation and 1D bounding boxes.

        The only method that needs to be implemented by the subclass is

        - augment_sample: Augment a single sample during training.

        Optionally, you can implement the following methods:

        - augment_label: Augment a single label during training.
        - get_random_transformations: Returns a nested structure of random transformations that should be applied to the batch.
            This is required to have unique transformations for each sample in the batch and maintain the same transformations for samples and labels.
        - batch_augment: Augment a batch of samples and labels during training. Needed if layer requires access to all samples (e.g. CutMix).

        By default, this method will coerce the input into a batch as well as a nested structure of inputs.
        If auto_vectorize is set to True, the augment_sample and augment_label methods will be vectorized using keras.ops.vectorized_map.
        Otherwise, it will use keras.ops.map which runs sequentially.

        Example:

        ```python

        class NormalizeLayer1D(BaseAugmentation1D):

            def __init__(self, **kwargs):
                ...

            def augment_sample(self, inputs):
                sample = inputs["data"]
                mu = keras.ops.mean()
                std = keras.ops.std()
                return (sample - mu) / (std + self.epsilon)

        x = np.random.rand(100, 3)
        lyr = NormalizeLayer(...)
        y = lyr(x, training=True)
        ```
        """
        super().__init__(**kwargs)

        if self.data_format == "channels_first":
            self.data_axis = -1
            self.ch_axis = -2
        else:
            self.data_axis = -2
            self.ch_axis = -1
        # END IF


@helia_export(path="helia_edge.layers.preprocessing.BaseAugmentation2D")
class BaseAugmentation2D(BaseAugmentation):
    NDIMS = 4  # (N, H, W, C) or (N, C, H, W)

    def __init__(self, **kwargs):
        """BaseAugmentation2D acts as a base class for various custom augmentation layers.
        This class provides a common interface for augmenting samples and labels. In the future, we will
        add support for segmentation and 1D bounding boxes.

        The only method that needs to be implemented by the subclass is

        - augment_sample: Augment a single sample during training.

        Optionally, you can implement the following methods:

        - augment_label: Augment a single label during training.
        - get_random_transformations: Returns a nested structure of random transformations that should be applied to the batch.
            This is required to have unique transformations for each sample in the batch and maintain the same transformations for samples and labels.
        - batch_augment: Augment a batch of samples and labels during training. Needed if layer requires access to all samples (e.g. CutMix).

        By default, this method will coerce the input into a batch as well as a nested structure of inputs.
        If auto_vectorize is set to True, the augment_sample and augment_label methods will be vectorized using keras.ops.vectorized_map.
        Otherwise, it will use keras.ops.map which runs sequentially.

        Example:

        ```python

        class NormalizeLayer2D(BaseAugmentation2D):

            def __init__(self, name=None, **kwargs):
                ...

            def augment_sample(self, inputs):
                sample = inputs["data"]
                mu = keras.ops.mean()
                std = keras.ops.std()
                return (sample - mu) / (std + self.epsilon)

        x = np.random.rand(32, 32, 3)
        lyr = NormalizeLayer(...)
        y = lyr(x, training=True)
        ```
        """
        super().__init__(**kwargs)

        if self.data_format == "channels_first":
            self.ch_axis = -3
            self.height_axis = -2
            self.width_axis = -1
        else:
            self.ch_axis = -1
            self.height_axis = -3
            self.width_axis = -2
        # END IF
