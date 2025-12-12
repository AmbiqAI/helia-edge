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

from .contrastive import ContrastiveTrainer
from .distiller import Distiller
from .gs_autoencoder import GSAutoencoder
from .mask_autoencoder import MaskedAutoencoder
from .simclr import SimCLRTrainer
from .vq_autoencoder import VQAutoencoder
