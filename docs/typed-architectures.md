# Typed architecture configuration

Model construction belongs to the reusable library. Callers own Keras backend
selection, input tensors, initialization seeds, weight loading, conversion and
execution. These factories do not clear sessions, reset random seeds or export
models. Backend selection must happen before importing Keras.

## Faithful MLPerf Tiny family

```python
from helia_edge.models import MlperfTinyParams, MlperfTinyModel

params = MlperfTinyParams(architecture="kws")
model = MlperfTinyModel.model_from_params(params)
restored_params = MlperfTinyParams.model_validate_json(params.model_dump_json())
```

The strict, immutable config selects `kws`, `vww`, `resnet` or `ad`. It rejects
unknown fields and unsupported architectures; there are no scaling, seed,
calibration or precision controls. These are the same fixed architectures as the
[reference constructors](mlperf-tiny.md). Existing `mlperf_tiny_*` functions and
their default names remain unchanged. An optional `name=` on the family factory
names the model instance without changing its architecture.

## Compact TCN preset and strict config ingestion

```python
import keras
from helia_edge.models import compact_tcn_params, TcnParams, TcnModel

params = compact_tcn_params(filters=8)
# Strict boundary for a decoded JSON object; rejects unknown nested fields.
params = TcnParams.from_config(params.model_dump(mode="json"))
inputs = keras.Input(shape=(240, 14), batch_size=1, dtype="float32")
model = TcnModel.model_from_params(inputs, params, num_classes=2)
```

The preset returns ordinary `TcnParams`: four small depthwise/pointwise blocks,
SE ratio 4, batch normalization/ReLU6, kernels 1×3, dilations 1/2/4/8 and a linear
1×1 output head. Integer filters must be at least 8 to retain the existing
builder's SE path. Input dimensions and output class count are supplied by the
caller; the example dimensions are not preset restrictions.

`TcnParams.from_config()` is an explicit unknown-field-rejecting boundary for
model and block fields. It retains normal Pydantic value coercion; it does not
claim strict primitive types or newly validate every historical TCN combination.
Existing `TcnParams.model_validate()` and dictionary factory calls retain their
previous behavior for compatibility. Use the explicit parser for new JSON
consumers, then pass typed objects internally. Parameter JSON describes the
architecture; Keras model/config and weight serialization describe the actual
initialized instance. A seed alone does not guarantee identical bytes across
backends or dependency versions.

## Export and benchmark consumers

A consumer may initialize a seed explicitly before construction, then call the
existing converter with the resulting model. Reuse the same initialized model
for each precision if comparing exports; keep representative-data selection and
export policy outside architecture configs. LiteRT export requires its optional
TensorFlow/converter dependencies; architecture construction is checked with
TensorFlow and Torch. Fixture arrays, receipts, replay and measurements belong
to the benchmark consumer, not these model APIs.
