"""
# :material-api: HeliaEdge API Documentation

## Available APIs

* **[Callbacks API](callbacks/)**: Provides classes to monitor training, save models, and more.
* **[Converters API](converters/)**: Provides classes to convert models to different formats.
* **[Interpreters API](interpreters/)**: Provides classes to interpret models.
* **[Layers API](layers/)**: Provides classes to build neural network layers.
* **[Losses API](losses/)**: Provides classes to compute loss functions.
* **[Metrics API](metrics/)**: Provides classes to compute evaluation metrics.
* **[Models API](models/)**: Provides classes to build neural network models.
* **[Optimizers API](optimizers/)**: Provides classes to optimize models.
* **[Quantizers API](quantizers/)**: Provides classes to quantize models.
* **[Trainers API](trainers/)**: Provides classes to train models.
* **[Utils API](utils/)**: Provides utility functions.

"""

from helia_edge._lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "register_keras_serializables": ("._serialization", "register_keras_serializables"),
        "callbacks": (".callbacks", None),
        "converters": (".converters", None),
        "interpreters": (".interpreters", None),
        "layers": (".layers", None),
        "losses": (".losses", None),
        "metrics": (".metrics", None),
        "models": (".models", None),
        "optimizers": (".optimizers", None),
        "plotting": (".plotting", None),
        "quantizers": (".quantizers", None),
        "trainers": (".trainers", None),
        "utils": (".utils", None),
    },
)
