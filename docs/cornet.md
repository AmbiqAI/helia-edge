# CorNET (PPG heart rate, convolution + LSTM)

`CorNetModel` rebuilds the CorNET heart-rate regressor from
[Biswas et al., IEEE TBioCAS 2019](https://doi.org/10.1109/TBCAS.2019.2892297):
two convolution stages followed by stacked LSTMs. `CorNetParams` holds the
backend-free config.

```python
import keras
from helia_edge.models import CorNetModel, CorNetParams

model = CorNetModel.model_from_params(keras.Input((1000, 1)), CorNetParams(), unroll=True)
```

**Input:** 8 s of wrist PPG at 125 Hz (1000 samples, one channel). The paper
band-passes the signal from 0.1 to 18 Hz and z-scores each window. This
preprocessing is the caller's responsibility.

**Default architecture** (paper Sec. III, Fig. 6, Table III):
- Two convolution stages. Each is Conv1D (32 filters, kernel 40, valid padding,
  stride 1), then BatchNorm, ReLU, MaxPooling 4 and Dropout 0.1. The window
  goes from 1000 samples to 961, 240, 201 and finally 50 timesteps.
- LSTM(128), returning its sequence.
- LSTM(128), returning the last step.
- A single linear output, `hr`.

Trainable parameters match Table III: 1,312 and 40,992 for the convolutions,
82,432 and 131,584 for the LSTMs, and 129 for the HR head.

Some details are not stated in the paper and are reconstructed here:
- Stride and padding are inferred from Table III's MAC counts.
- Fig. 6 places BatchNorm before ReLU, while the text places it after. The
  constructor follows Fig. 6.
- The dropout position is not stated.

The model processes one window per call. It is not a streaming model, and the
LSTM state starts at zero for every window.

## LSTM lowering

`unroll` changes only how the LSTMs are built, not their weights:

| Form | How to build it | LiteRT lowering | Notes |
|---|---|---|---|
| Rolled | `unroll=False` (default) | `WHILE` loop | Loop state is not quantizable, and some engines do not parse `WHILE`. |
| Unrolled | `unroll=True` | Per-timestep `FULLY_CONNECTED`, `LOGISTIC`, `TANH`, `MUL` and `ADD` operations | These can be quantized to INT8 and 16x8. |

Keras 3 does not emit the fused `UNIDIRECTIONAL_SEQUENCE_LSTM` operator. A
fused export needs a tf-keras rebuild of the same architecture with the
weights copied over; that is export tooling, not part of this constructor.

## Weights and terms

The paper publishes no code or weights. These constructors produce untrained
models, suited to performance measurement only; no heart-rate accuracy is
implied.
