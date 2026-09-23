"""Static contracts for lazy public imports; checked by ty, not executed."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import helia_edge as helia
from helia_edge.models.tcn import TcnParams
from helia_edge.trainers.mask_autoencoder import MaskedAutoencoder, Reconstruction, ReconstructionLoss
from helia_edge.utils.sampling import StreamMode

if TYPE_CHECKING:
    from helia_edge._typing import Tensor


def public_contracts(model: MaskedAutoencoder) -> None:
    assert_type(helia.models.TcnParams(), TcnParams)
    assert_type(helia.utils.StreamMode("finite"), StreamMode)
    result = model.reconstruction_targets(np.zeros((2, 2, 2, 1), dtype="float32"))
    assert_type(result, Reconstruction)
    assert_type(result.targets, Tensor)
    assert_type(model.calculate_loss(np.zeros((2, 2, 2, 1))), ReconstructionLoss)


def plotting_contracts() -> None:
    assert_type(
        helia.plotting.plot_history_metrics({"loss": [1.0]}, ["loss"], include_val=False),
        tuple[Figure, Axes],
    )
    assert_type(
        helia.plotting.history.plot_history_metrics({"loss": [1.0]}, ["loss"], include_val=False),
        tuple[Figure, Axes],
    )
