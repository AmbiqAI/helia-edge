import keras

from ..utils import helia_export

@helia_export(path="helia_edge.layers.GumbelSoftmaxBottleneck")
class GumbelSoftmaxBottleneck(keras.layers.Layer):
    """
    Discrete bottleneck via Gumbel-Softmax (Concrete) with optional straight-through hard one-hot.

    Inputs:
      x: [..., Din]  (features)  -- if input_is_logits=False (default), we learn a linear proj to K logits
         OR
      x: [..., K]    (logits)    -- if input_is_logits=True, we treat last dim as K logits directly

    Outputs:
      z: [..., D]    expected embedding  z = soft_one_hot @ embed   (D = embedding_dim)

    Adds loss:
      kl_weight * mean_bits_per_index    (KL(q || Uniform(K)) in *bits*, averaged over tokens)
    Tracks metrics (logged via `metrics`):
      - gs_bits_per_index   (lower bound, bits/index)
      - gs_perplexity       (empirical perplexity from hard argmax histogram)
      - gs_usage            (fraction of codes used at least once in the batch)
      - gs_temperature      (current τ; useful when annealing)
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        temperature: float = 1.0,
        hard: bool = True,
        input_is_logits: bool = False,
        use_bias: bool = True,
        kl_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if num_embeddings <= 1 or embedding_dim <= 0:
            raise ValueError("num_embeddings must be >=2 and embedding_dim > 0.")
        self.K = int(num_embeddings)
        self.D = int(embedding_dim)

        self.hard = bool(hard)
        self.input_is_logits = bool(input_is_logits)
        self.use_bias = bool(use_bias)
        self.kl_weight = float(kl_weight)

        # temperature stored as non-trainable weight (easy to anneal via a callback)
        self.tau = self.add_weight(
            name="temperature",
            shape=(),
            initializer=keras.initializers.Constant(float(temperature)),
            trainable=False,
            dtype="float32",
        )

        # Trackers
        self._bpi  = keras.metrics.Mean(name="gs_bits_per_index")   # KL lower bound (bits/index)
        self._perp = keras.metrics.Mean(name="gs_perplexity")
        self._usage= keras.metrics.Mean(name="gs_usage")
        self._tau  = keras.metrics.Mean(name="gs_temperature")

        # weights created in build()
        self._proj = None    # [Din, K] if input_is_logits=False
        self._bias = None    # [K]     if use_bias
        self._embed = None   # [K, D]

    def build(self, input_shape):
        last = int(input_shape[-1])
        # projection (if we aren't being fed logits directly)
        if not self.input_is_logits:
            lim = (1.0 / max(1, last)) ** 0.5
            self._proj = self.add_weight(
                name="proj", shape=(last, self.K),
                initializer=keras.initializers.RandomUniform(-lim, lim),
                trainable=True, dtype=self.variable_dtype,
            )
            if self.use_bias:
                self._bias = self.add_weight(
                    name="bias", shape=(self.K,),
                    initializer=keras.initializers.Zeros(),
                    trainable=True, dtype=self.variable_dtype,
                )
        else:
            if last != self.K:
                raise ValueError(f"input_is_logits=True but last dim {last} != K {self.K}")

        # code embeddings
        limE = 1.0 / max(1, self.K)
        self._embed = self.add_weight(
            name="embed", shape=(self.K, self.D),
            initializer=keras.initializers.RandomUniform(-limE, limE),
            trainable=True, dtype=self.variable_dtype,
        )
        super().build(input_shape)

    # ---- helpers ----
    def set_temperature(self, value: float):
        self.tau.assign(float(value))

    def _gumbel(self, shape, dtype):
        u = keras.random.uniform(shape, 0.0, 1.0, dtype=dtype)
        return -keras.ops.log(-keras.ops.log(u + 1e-9) + 1e-9)

    # ---- forward ----
    def call(self, x, training=False, return_indices: bool = False, return_probs: bool = False):
        x = keras.ops.convert_to_tensor(x, dtype=self.compute_dtype)

        # logits
        if self.input_is_logits:
            logits = x
        else:
            logits = keras.ops.matmul(x, self._proj)              # [..., K]
            if self.use_bias:
                logits = logits + self._bias

        tau = keras.ops.cast(self.tau, logits.dtype)

        # relaxed one-hot
        if training:
            g = self._gumbel(keras.ops.shape(logits), logits.dtype)
            y_soft = keras.ops.softmax((logits + g) / tau, axis=-1)
        else:
            y_soft = keras.ops.softmax(logits / keras.ops.maximum(tau, 1e-6), axis=-1)

        # optional straight-through
        idx = keras.ops.argmax(y_soft, axis=-1)                   # [...,]
        if self.hard:
            y_hard = keras.ops.one_hot(idx, self.K)              # [..., K]
            y = y_hard + keras.ops.stop_gradient(y_soft - y_hard)
            metrics_counts = y_hard                               # reuse for usage/perplexity
        else:
            y = y_soft
            metrics_counts = keras.ops.one_hot(idx, self.K)       # still use hard samples for metrics

        # expected embedding
        z = keras.ops.matmul(y, self._embed)                      # [..., D]

        # ---------- rate (KL to Uniform) and metrics ----------
        eps  = keras.ops.convert_to_tensor(1e-10, dtype=logits.dtype)
        ln2  = keras.ops.log(keras.ops.convert_to_tensor(2.0, dtype=logits.dtype))
        # tokenwise entropy in nats
        H_tok = -keras.ops.sum(y_soft * keras.ops.log(y_soft + eps), axis=-1)     # [...,]
        H_nats = keras.ops.mean(H_tok)                                            # scalar
        KL_nats = keras.ops.log(keras.ops.convert_to_tensor(float(self.K), dtype=logits.dtype)) - H_nats
        bpi = KL_nats / ln2                                                       # bits/index (uniform prior)
        self._bpi.update_state(bpi)

        # empirical perplexity & usage from hard indices
        probs = keras.ops.mean(keras.ops.reshape(metrics_counts, (-1, self.K)), axis=0)  # [K]
        H_bits = -keras.ops.sum(probs * (keras.ops.log(probs + eps) / ln2))       # bits
        perp = keras.ops.exp(H_bits * ln2)                                        # 2**H_bits
        usage = keras.ops.sum(keras.ops.cast(probs > 0, logits.dtype)) / float(self.K)
        self._perp.update_state(perp)
        self._usage.update_state(usage)
        self._tau.update_state(self.tau)

        # add KL (bits/index) as loss, weighted; mean over tokens is already taken
        if self.kl_weight != 0.0:
            self.add_loss(self.kl_weight * bpi)

        if return_indices and return_probs:
            return z, idx, y_soft
        if return_indices:
            return z, idx
        if return_probs:
            return z, y_soft
        return z

    @property
    def metrics(self):
        return [self._bpi, self._perp, self._usage, self._tau]

    def get_config(self):
        return {
            **super().get_config(),
            "num_embeddings": self.K,
            "embedding_dim": self.D,
            "temperature": float(self.tau.numpy()) if hasattr(self.tau, "numpy") else 1.0,
            "hard": self.hard,
            "input_is_logits": self.input_is_logits,
            "use_bias": self.use_bias,
            "kl_weight": self.kl_weight,
        }
