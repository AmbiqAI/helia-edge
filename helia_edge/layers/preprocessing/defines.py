from typing import List, Mapping, TypeAlias

import keras

NestedTensorType: TypeAlias = List["NestedTensorValue"] | Mapping[str, "NestedTensorValue"]
NestedTensorValue: TypeAlias = keras.KerasTensor | NestedTensorType
