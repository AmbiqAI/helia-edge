from typing import TypeAlias, Mapping, List
import keras

NestedTensorType: TypeAlias = List["NestedTensorValue"] | Mapping[str, "NestedTensorValue"]
NestedTensorValue: TypeAlias = keras.KerasTensor | NestedTensorType
