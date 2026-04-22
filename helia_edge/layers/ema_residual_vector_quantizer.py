"""Residual Vector Quantizer with Exponential Moving Average codebook updates.

Replaces the gradient-based codebook loss with EMA updates to codebook
embeddings, following van den Oord et al. 2017 (VQ-VAE).  Only the
commitment loss is back-propagated; codebook vectors are updated via
running averages of assigned encoder outputs.
"""

from __future__ import annotations

from typing import Sequence

import keras

from ..utils import helia_export


@helia_export(path="helia_edge.layers.EmaResidualVectorQuantizer")
class EmaResidualVectorQuantizer(keras.layers.Layer):
    """Residual VQ with EMA codebook updates.

    Instead of learning codebook embeddings via gradient descent (which requires
    a codebook loss term), this layer maintains exponential moving averages of
    cluster assignment counts and embedding sums.  Codebook vectors are derived
    from these running statistics with Laplace smoothing for numerical stability.

    Only the *commitment loss* is back-propagated through the encoder; the
    straight-through estimator copies gradients from the decoder to the encoder
    as in the standard VQ-VAE.

    Input:  ``[..., D]``  (last dim = ``embedding_dim``)
    Output: ``[..., D]``  (sum of per-level dequantized vectors)

    Args:
        num_levels: Number of residual VQ stages (``M >= 1``).
        num_embeddings: Codebook size ``K`` per level (int or per-level list).
        embedding_dim: Latent dimensionality ``D``.
        beta: Commitment loss coefficient.
        ema_decay: EMA decay rate for codebook updates (``0.99``\u2013``0.999``
            typical).
        epsilon: Small constant for Laplace smoothing of cluster counts.

    Metrics (logged via ``metrics`` property):
        - ``rvq_l{l}_perplexity``, ``rvq_l{l}_usage``,
          ``rvq_l{l}_bits_per_index``
        - ``rvq_perplexity_mean``, ``rvq_usage_mean``,
          ``rvq_bits_per_index_sum`` (entropy lower bound)

    Losses added per level:
        - ``beta * ||stop(q_l) - r_l||^2``  (commitment only; no codebook
          gradient loss)

    Example:

    ```python
    rvq = EmaResidualVectorQuantizer(
        num_levels=4,
        num_embeddings=64,
        embedding_dim=16,
        ema_decay=0.99,
    )
    y = rvq(z, training=True)          # forward + EMA update
    y, indices = rvq(z, return_indices=True)  # also return codes
    ```

    References:
        - van den Oord, A., Vinyals, O. & Kavukcuoglu, K. (2017).
          *Neural Discrete Representation Learning*. NeurIPS.
    """

    def __init__(
        self,
        num_levels: int,
        num_embeddings: int | Sequence[int],
        embedding_dim: int,
        beta: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if num_levels < 1 or embedding_dim <= 0 or beta <= 0:
            raise ValueError("num_levels>=1, embedding_dim>0, beta>0 required.")
        self.M = int(num_levels)
        self.D = int(embedding_dim)
        if isinstance(num_embeddings, (list, tuple)):
            if len(num_embeddings) != self.M:
                raise ValueError("num_embeddings list must have length = num_levels.")
            self.Ks = [int(k) for k in num_embeddings]
        else:
            self.Ks = [int(num_embeddings)] * self.M
        self.beta = float(beta)
        self.ema_decay = float(ema_decay)
        self.epsilon = float(epsilon)

        # Per-level metric trackers
        self._lvl_perp = [
            keras.metrics.Mean(name=f"rvq_l{lvl + 1}_perplexity")
            for lvl in range(self.M)
        ]
        self._lvl_usage = [
            keras.metrics.Mean(name=f"rvq_l{lvl + 1}_usage")
            for lvl in range(self.M)
        ]
        self._lvl_bpi = [
            keras.metrics.Mean(name=f"rvq_l{lvl + 1}_bits_per_index")
            for lvl in range(self.M)
        ]
        # Aggregates
        self._perp_mean = keras.metrics.Mean(name="rvq_perplexity_mean")
        self._usage_mean = keras.metrics.Mean(name="rvq_usage_mean")
        self._bpi_sum = keras.metrics.Mean(name="rvq_bits_per_index_sum")

        self._codebooks: list = []
        self._ema_counts: list = []
        self._ema_weights: list = []

    def build(self, input_shape):
        last = input_shape[-1]
        if last is not None and int(last) != self.D:
            raise ValueError(f"Input last dim {int(last)} != embedding_dim {self.D}")

        self._codebooks = []
        self._ema_counts = []
        self._ema_weights = []
        for lvl, K in enumerate(self.Ks):
            limit = 1.0 / max(1, K)
            # Codebook embeddings -- NOT trainable (updated via EMA)
            cb = self.add_weight(
                name=f"codebook_l{lvl + 1}",
                shape=(K, self.D),
                initializer=keras.initializers.RandomUniform(-limit, limit),
                trainable=False,
                dtype=self.variable_dtype,
            )
            # EMA cluster counts -- shape (K,)
            ema_count = self.add_weight(
                name=f"ema_count_l{lvl + 1}",
                shape=(K,),
                initializer="ones",
                trainable=False,
                dtype=self.variable_dtype,
            )
            # EMA embedding sums -- shape (K, D)
            ema_weight = self.add_weight(
                name=f"ema_weight_l{lvl + 1}",
                shape=(K, self.D),
                initializer="zeros",
                trainable=False,
                dtype=self.variable_dtype,
            )
            self._codebooks.append(cb)
            self._ema_counts.append(ema_count)
            self._ema_weights.append(ema_weight)

        # Initialise ema_weight = codebook * 1.0 so the first update is stable
        for cb, ew in zip(self._codebooks, self._ema_weights):
            ew.assign(cb)

        super().build(input_shape)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest(self, r_flat, codebook):
        """Return indices ``[N]`` and gathered vectors ``[N, D]``."""
        r2 = keras.ops.sum(keras.ops.square(r_flat), axis=1, keepdims=True)
        e2 = keras.ops.sum(keras.ops.square(codebook), axis=1)
        sim = keras.ops.matmul(r_flat, keras.ops.transpose(codebook))
        dist = r2 + e2 - 2.0 * sim
        idx = keras.ops.argmax(-dist, axis=1)
        q = keras.ops.take(codebook, idx, axis=0)
        return idx, q

    def _ema_update(self, lvl, idx, r_flat, K):
        """Update ``codebook[lvl]`` via EMA using assigned vectors."""
        gamma = self.ema_decay
        one_hot = keras.ops.one_hot(idx, K)  # (N, K)

        # Batch cluster counts and embedding sums
        new_count = keras.ops.sum(one_hot, axis=0)  # (K,)
        new_weight = keras.ops.matmul(
            keras.ops.transpose(one_hot), r_flat
        )  # (K, D)

        # EMA update
        updated_count = gamma * self._ema_counts[lvl] + (1 - gamma) * new_count
        updated_weight = gamma * self._ema_weights[lvl] + (1 - gamma) * new_weight

        # Laplace smoothing of counts
        n = keras.ops.sum(updated_count)
        smoothed = (updated_count + self.epsilon) / (n + K * self.epsilon) * n

        # Normalise to get new codebook
        new_cb = updated_weight / keras.ops.expand_dims(smoothed, axis=1)

        self._ema_counts[lvl].assign(updated_count)
        self._ema_weights[lvl].assign(updated_weight)
        self._codebooks[lvl].assign(new_cb)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(
        self,
        x: keras.KerasTensor,
        training: bool = False,
        return_indices: bool = False,
    ) -> keras.KerasTensor | tuple[keras.KerasTensor, list[keras.KerasTensor]]:
        """Quantize *x* through all residual levels.

        Args:
            x: ``[..., D]`` latent to be quantized.
            training: If ``True``, run EMA codebook updates.
            return_indices: If ``True``, also return per-level flat indices.

        Returns:
            ``y`` or ``(y, indices_list)``: dequantized vector and optional
            per-level index tensors.
        """
        x = keras.ops.convert_to_tensor(x, dtype=self.compute_dtype)
        shape = keras.ops.shape(x)
        flat = keras.ops.reshape(x, (-1, self.D))

        residual = flat
        q_sum = keras.ops.zeros_like(flat)
        indices_list = []
        perp_vals, usage_vals, bpi_vals = [], [], []

        for lvl, (K, codebook) in enumerate(zip(self.Ks, self._codebooks)):
            idx, q_l = self._nearest(residual, codebook)
            indices_list.append(idx)
            q_sum = q_sum + q_l

            # Commitment loss only (EMA handles codebook)
            ql_st = keras.ops.stop_gradient(q_l)
            commitment = keras.ops.mean(keras.ops.square(ql_st - residual))
            self.add_loss(self.beta * commitment)

            # EMA codebook update during training
            if training:
                self._ema_update(lvl, idx, residual, K)

            residual = residual - ql_st

            # Per-level metrics
            one_hot = keras.ops.one_hot(idx, K)
            probs = keras.ops.mean(one_hot, axis=0)
            eps = keras.ops.convert_to_tensor(1e-10, dtype=self.compute_dtype)
            log2 = keras.ops.log(
                keras.ops.convert_to_tensor(2.0, self.compute_dtype)
            )
            H = -keras.ops.sum(probs * (keras.ops.log(probs + eps) / log2))
            perp = keras.ops.exp(H * log2)
            usage = keras.ops.sum(
                keras.ops.cast(probs > 0, self.compute_dtype)
            ) / float(K)

            self._lvl_perp[lvl].update_state(perp)
            self._lvl_usage[lvl].update_state(usage)
            self._lvl_bpi[lvl].update_state(H)
            perp_vals.append(perp)
            usage_vals.append(usage)
            bpi_vals.append(H)

        # Aggregate metrics
        self._perp_mean.update_state(sum(perp_vals) / float(self.M))
        self._usage_mean.update_state(sum(usage_vals) / float(self.M))
        self._bpi_sum.update_state(sum(bpi_vals))

        # Straight-through estimator
        y_flat = flat + keras.ops.stop_gradient(q_sum - flat)
        y = keras.ops.reshape(y_flat, shape)
        return (y, indices_list) if return_indices else y

    @property
    def metrics(self):
        """Expose per-level + aggregate metrics so ``Model.fit`` logs them."""
        return (
            self._lvl_perp
            + self._lvl_usage
            + self._lvl_bpi
            + [self._perp_mean, self._usage_mean, self._bpi_sum]
        )

    # ------------------------------------------------------------------
    # Encode / decode utilities
    # ------------------------------------------------------------------

    def encode(self, x: keras.KerasTensor) -> list[keras.KerasTensor]:
        """Return list of per-level flat index tensors ``[N]`` (no gradients)."""
        x = keras.ops.convert_to_tensor(x, dtype=self.compute_dtype)
        flat = keras.ops.reshape(x, (-1, self.D))
        residual = flat
        indices = []
        for codebook in self._codebooks:
            idx, q_l = self._nearest(residual, codebook)
            indices.append(idx)
            residual = residual - q_l
        return indices

    def decode(
        self,
        indices_list: list[keras.KerasTensor],
        original_shape: tuple[int, ...],
    ) -> keras.KerasTensor:
        """Sum per-level code vectors from *indices_list* and reshape."""
        q_sum = None
        for idx, codebook in zip(indices_list, self._codebooks):
            q_l = keras.ops.take(codebook, idx, axis=0)
            q_sum = q_l if q_sum is None else (q_sum + q_l)
        return keras.ops.reshape(q_sum, original_shape)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "num_levels": self.M,
                "num_embeddings": self.Ks,
                "embedding_dim": self.D,
                "beta": self.beta,
                "ema_decay": self.ema_decay,
                "epsilon": self.epsilon,
            }
        )
        return cfg
