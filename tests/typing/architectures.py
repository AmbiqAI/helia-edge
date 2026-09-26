"""Static public architecture return contracts; checked without model execution."""

from typing import assert_type

import keras

from helia_edge.models import MlperfTinyModel, MlperfTinyParams, TcnModel, TcnParams, compact_tcn_params


def construction(inputs: keras.KerasTensor) -> None:
    params = compact_tcn_params(filters=8)
    assert_type(params, TcnParams)
    assert_type(TcnParams.from_config(params.model_dump()), TcnParams)
    assert_type(TcnModel.model_from_params(inputs, params, num_classes=2), keras.Model)
    assert_type(TcnModel.layer_from_params(inputs, params, num_classes=2), keras.KerasTensor)
    assert_type(MlperfTinyModel.model_from_params(MlperfTinyParams(architecture="kws")), keras.Model)
