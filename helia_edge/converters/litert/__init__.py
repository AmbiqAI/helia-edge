"""
# LiteRT Converter API

This module handles converting models to LiteRT format.

Classes:
    QuantizationType: Enum class for quantization types.
    LiteRTKerasConverter: LiteRT model converter.
    ConversionType: Enum class for conversion types.

"""

from helia_edge._lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "ConversionType": ("..tflite", "ConversionType"),
        "QuantizationType": ("..tflite", "QuantizationType"),
        "LiteRTKerasConverter": (".converter", "LiteRTKerasConverter"),
        "converter": (".converter", None),
    },
)
