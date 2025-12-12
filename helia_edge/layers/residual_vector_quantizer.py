import keras

from ..utils import helia_export

@helia_export(path="helia_edge.layers.ResidualVectorQuantizer")
class ResidualVectorQuantizer(keras.layers.Layer):
    """
    Residual Vector Quantizer (RVQ) with straight-through estimator.

    Input:  [..., D]  (last dim = embedding_dim)
    Output: [..., D]  (sum of per-level dequantized vectors; gradients pass through x)

    Args:
      num_levels: int, number of residual VQ stages (M >= 1)
      num_embeddings: int OR sequence[int], codebook size K for each level
      embedding_dim: int, latent dimensionality D
      beta: float, commitment coefficient per level

    Metrics (logged via `metrics` property):
      - rvq_l{l}_perplexity, rvq_l{l}_usage, rvq_l{l}_bits_per_index
      - rvq_perplexity_mean, rvq_usage_mean, rvq_bits_per_index_sum (entropy lower bound)

    Losses added per level:
      - beta * ||stop(q_l) - r_l||^2  +  ||q_l - stop(r_l)||^2,
        where r_l is the current residual and q_l the level-l code vector.
    """
    def __init__(self, num_levels, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        if num_levels < 1 or embedding_dim <= 0 or beta <= 0:
            raise ValueError("num_levels>=1, embedding_dim>0, beta>0 required.")
        self.M = int(num_levels)
        self.D = int(embedding_dim)
        # Allow int or per-level list/tuple for K
        if isinstance(num_embeddings, (list, tuple)):
            if len(num_embeddings) != self.M:
                raise ValueError("num_embeddings list must have length = num_levels.")
            self.Ks = [int(k) for k in num_embeddings]
        else:
            self.Ks = [int(num_embeddings)] * self.M
        self.beta = float(beta)

        # Per-level metric trackers
        self._lvl_perp = [keras.metrics.Mean(name=f"rvq_l{lvl+1}_perplexity") for lvl in range(self.M)]
        self._lvl_usage = [keras.metrics.Mean(name=f"rvq_l{lvl+1}_usage") for lvl in range(self.M)]
        self._lvl_bpi = [keras.metrics.Mean(name=f"rvq_l{lvl+1}_bits_per_index") for lvl in range(self.M)]
        # Aggregates
        self._perp_mean = keras.metrics.Mean(name="rvq_perplexity_mean")
        self._usage_mean = keras.metrics.Mean(name="rvq_usage_mean")
        self._bpi_sum = keras.metrics.Mean(name="rvq_bits_per_index_sum")

        self._codebooks = []  # created in build()

    def build(self, input_shape):
        last = input_shape[-1]
        if last is not None and int(last) != self.D:
            raise ValueError(f"Input last dim {int(last)} != embedding_dim {self.D}")

        self._codebooks = []
        for lvl, K in enumerate(self.Ks):
            limit = 1.0 / max(1, K)
            cb = self.add_weight(
                name=f"codebook_l{lvl+1}",
                shape=(K, self.D),  # [K, D]
                initializer=keras.initializers.RandomUniform(-limit, limit),
                trainable=True,
                dtype=self.variable_dtype,
            )
            self._codebooks.append(cb)
        super().build(input_shape)

    def _nearest(self, r_flat, codebook):
        """Return indices [N] and gathered vectors [N,D] for residual r_flat against codebook [K,D]."""
        # ||r||^2 + ||e||^2 - 2 r·e
        r2 = keras.ops.sum(keras.ops.square(r_flat), axis=1, keepdims=True)       # [N,1]
        e2 = keras.ops.sum(keras.ops.square(codebook), axis=1)                    # [K]
        sim = keras.ops.matmul(r_flat, keras.ops.transpose(codebook))             # [N,K]
        dist = r2 + e2 - 2.0 * sim
        idx = keras.ops.argmax(-dist, axis=1)                                     # [N]
        q = keras.ops.take(codebook, idx, axis=0)                                 # [N,D]
        return idx, q

    def call(self, x, return_indices: bool = False):
        """
        Args:
          x: [..., D] latent to be quantized.
          return_indices: if True, also returns list of flat indices (one tensor per level).

        Returns:
          y or (y, indices_list): dequantized vector and optional per-level indices.
        """
        x = keras.ops.convert_to_tensor(x, dtype=self.compute_dtype)
        shape = keras.ops.shape(x)
        flat = keras.ops.reshape(x, (-1, self.D))                                  # [N, D]

        residual = flat                                                            # r_1 = x
        q_sum = keras.ops.zeros_like(flat)                                         # accumulate ∑ q_l
        indices_list = []
        perp_vals, usage_vals, bpi_vals = [], [], []

        for lvl, (K, codebook) in enumerate(zip(self.Ks, self._codebooks)):
            idx, q_l = self._nearest(residual, codebook)                           # [N], [N,D]
            indices_list.append(idx)
            q_sum = q_sum + q_l                                                    # accumulate
            # Losses for this level on its residual
            ql_st = keras.ops.stop_gradient(q_l)
            res_st = keras.ops.stop_gradient(residual)
            commitment = keras.ops.mean(keras.ops.square(ql_st - residual))
            codebook_loss = keras.ops.mean(keras.ops.square(q_l - res_st))
            self.add_loss(self.beta * commitment + codebook_loss)

            # Update residual for next stage (quantize the residual)
            residual = residual - ql_st

            # Metrics for this level
            one_hot = keras.ops.one_hot(idx, K)                                    # [N,K]
            probs = keras.ops.mean(one_hot, axis=0)                                 # [K]
            eps = keras.ops.convert_to_tensor(1e-10, dtype=self.compute_dtype)
            log2 = keras.ops.log(keras.ops.convert_to_tensor(2.0, self.compute_dtype))
            H = -keras.ops.sum(probs * (keras.ops.log(probs + eps) / log2))        # bits/index
            # perp = keras.ops.pow(keras.ops.convert_to_tensor(2.0, self.compute_dtype), H)
            perp = keras.ops.exp(H * log2)
            usage = keras.ops.sum(keras.ops.cast(probs > 0, self.compute_dtype)) / float(K)

            self._lvl_perp[lvl].update_state(perp)
            self._lvl_usage[lvl].update_state(usage)
            self._lvl_bpi[lvl].update_state(H)
            perp_vals.append(perp)
            usage_vals.append(usage)
            bpi_vals.append(H)

        # Aggregate metrics across levels
        perp_mean = sum(perp_vals) / float(self.M)
        usage_mean = sum(usage_vals) / float(self.M)
        bpi_sum = sum(bpi_vals)                                                    # total entropy lower bound
        self._perp_mean.update_state(perp_mean)
        self._usage_mean.update_state(usage_mean)
        self._bpi_sum.update_state(bpi_sum)

        # Straight-through estimator for the whole stack: forward=q_sum, backward=identity
        y_flat = flat + keras.ops.stop_gradient(q_sum - flat)
        y = keras.ops.reshape(y_flat, shape)

        return (y, indices_list) if return_indices else y

    @property
    def metrics(self):
        # Expose per-level + aggregates so Model.fit logs them
        return (self._lvl_perp + self._lvl_usage + self._lvl_bpi +
                [self._perp_mean, self._usage_mean, self._bpi_sum])

    # Utilities for export/inference
    def encode(self, x):
        """Return list of per-level flat index tensors [N] (no gradients)."""
        x = keras.ops.convert_to_tensor(x, dtype=self.compute_dtype)
        flat = keras.ops.reshape(x, (-1, self.D))
        residual = flat
        indices = []
        for K, codebook in zip(self.Ks, self._codebooks):
            idx, q_l = self._nearest(residual, codebook)
            indices.append(idx)
            residual = residual - q_l
        return indices

    def decode(self, indices_list, original_shape):
        """Sum per-level code vectors from indices_list and reshape to original_shape."""
        # indices_list: list of 1D int tensors [N]
        q_sum = None
        for idx, codebook in zip(indices_list, self._codebooks):
            q_l = keras.ops.take(codebook, idx, axis=0)
            q_sum = q_l if q_sum is None else (q_sum + q_l)
        return keras.ops.reshape(q_sum, original_shape)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_levels": self.M,
            "num_embeddings": self.Ks,
            "embedding_dim": self.D,
            "beta": self.beta,
        })
        return cfg
