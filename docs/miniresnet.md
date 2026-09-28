# MiniResNet-v1

`MiniResNetV1Params` describes a reusable spectrogram CNN. The default topology
matches ST's one-stack MiniResNet-v1 ESC-10 checkpoint: a padded 7×7 stem,
max-pooling, two residual blocks (1×1 then 3×3 convolutions), flatten and a
classification head. The first block downsamples both paths with a projection;
convolutions retain biases and batch normalization uses epsilon `1.001e-5`.

```python
import keras
from helia_edge.models import MiniResNetV1Model, MiniResNetV1Params

params = MiniResNetV1Params.from_config({"stacks": 1, "base_filters": 64})
model = MiniResNetV1Model.model_from_params(
    inputs=keras.Input((64, 50, 1), dtype="float32"),
    params=params,
    num_classes=10,
)
# The model is initialized, not pretrained. Hydrate explicitly from a local,
# verified checkpoint matching the architecture, input shape and class count:
# model.load_weights(checkpoint_path)
# model.save("classifier.keras")
```

Config serialization uses `params.get_config()` / `from_config()` or Pydantic
JSON methods. Unknown fields, invalid dimensions and coercions such as string
channel counts are rejected. The config is immutable. One to three stacks,
base channel count, pooling, head dropout and output activation are configurable;
changed values are new architectures, not claims of pretrained variants.
Inputs are NHWC regardless of the global image layout. Flatten needs fixed
spatial dimensions; global average/max pooling supports dynamic spatial sizes.
The output has `num_classes` scores. All layers are ordinary Keras layers, so
model serialization and weight loading use standard Keras APIs. Layers default
to trainable; callers control freezing when fine-tuning.

The constructor does not reset a session/seed, fetch weights, process audio or
export a model. Callers own input signatures, class labels, artifact integrity,
preprocessing and execution. Benchmark recipes reference their catalog's model
identity; EDGE does not maintain another asset catalog or YAML runner.

## Trained reference boundary

Architecture source: [ST services MiniResNet-v1 at 0f6210ed](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/0f6210ed5156126b782e1c43249063a477484b20/audio_event_detection/tf/src/models/miniresnetv1/miniresnetv1.py).
The TensorFlow/ST source notices and Apache-2.0 license are retained in
`helia_edge/models/licenses/miniresnet-apache-2.0.txt`.
The [matching ST model and configuration](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/1423c78953a830903485135febe1dd98ff31aed8/audio_event_detection/miniresnetv1/ST_pretrainedmodel_public_dataset/esc10/miniresnetv1_s1_64x50_tl)
are separately Apache-2.0 licensed by their model directory. Weights are not
bundled with EDGE. The `(64,50,1)`, ten-class default reconstruction has 126,922
parameters and 27 layers. Its trained host output parity and serialization are
separate from the upstream INT8 export, model accuracy and device qualification.

The matching frontend uses 16-kHz audio, 64 mel bands, 1024-sample FFT/Hann
window, hop320, 20–7500Hz, Slaney normalization, power2 and dB scaling, with
centered constant padding and 50-frame patches. Exact time cleanup, dB reference,
patch overlap, class ordering and clip aggregation belong to the reference
recipe, not the CNN constructor. Use the pinned training configuration and
loader together; raw waveform or arbitrary spectrogram normalization is not an
equivalent model input. No real-audio accuracy or quantized-output equivalence
is implied by host reconstruction.
