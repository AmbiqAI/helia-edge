"""Static public architecture return contracts; checked without model execution."""

from typing import assert_type

import keras

from helia_edge.models import MlperfTinyModel, MlperfTinyParams, ModelSpec, TcnParams, build, compact_tcn_params
from helia_edge.models.tcn import build as tcn_build


def construction() -> None:
    params = compact_tcn_params(filters=8, num_classes=2)
    assert_type(params, TcnParams)
    assert_type(TcnParams.model_validate(params.model_dump()), TcnParams)
    assert_type(tcn_build(params, (240, 14), batch_size=1), keras.Model)
    assert_type(build(ModelSpec(params=params, input_shape=(240, 14))), keras.Model)
    assert_type(MlperfTinyModel.model_from_params(MlperfTinyParams(architecture="kws")), keras.Model)
