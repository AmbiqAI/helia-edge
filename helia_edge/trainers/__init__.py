"""
# :simple-futurelearn: Trainers API

This module contains the implementations of various training routines that fall outside
the standard supervised learning paradigm. These include contrastive learning, distillation, and more.

## Available Trainers

- **[ContrastiveTrainer](./contrastive)**: A trainer for contrastive learning
- **[Distiller](./distiller)**: A trainer for distillation
- **[GSAutoencoder](./gs_autoencoder)**: A trainer for Gumbel Softmax autoencoder
- **[MaskedAutoencoder](./mask_autoencoder)**: A trainer for masked autoencoder
- **[SimCLRTrainer](./simclr)**: A trainer for SimCLR
- **[VQAutoencoder](./vq_autoencoder)**: A trainer for Vector Quantized autoencoder

"""

from helia_edge._lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "ContrastiveTrainer": (".contrastive", "ContrastiveTrainer"),
        "Distiller": (".distiller", "Distiller"),
        "GSAutoencoder": (".gs_autoencoder", "GSAutoencoder"),
        "MaskedAutoencoder": (".mask_autoencoder", "MaskedAutoencoder"),
        "SimCLRTrainer": (".simclr", "SimCLRTrainer"),
        "VQAutoencoder": (".vq_autoencoder", "VQAutoencoder"),
        "contrastive": (".contrastive", None),
        "distiller": (".distiller", None),
        "gs_autoencoder": (".gs_autoencoder", None),
        "mask_autoencoder": (".mask_autoencoder", None),
        "simclr": (".simclr", None),
        "vq_autoencoder": (".vq_autoencoder", None),
    },
)
