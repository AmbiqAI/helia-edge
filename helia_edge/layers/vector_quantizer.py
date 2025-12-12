import keras

from ..utils import helia_export


@helia_export(path="helia_edge.layers.VectorQuantizer")
class VectorQuantizer(keras.layers.Layer):
    """
    Vector-quantization bottleneck (VQ-VAE style) with straight-through estimator.

    Input:  [..., D]  (last dim == embedding_dim)
    Output: [..., D]  (quantized/dequantized vectors; gradients pass through x)

    Tracks (logged automatically via `metrics` property):
      - vq_perplexity       : effective # active codes (1..K)
      - vq_usage            : fraction of codes used at least once (0..1)
      - vq_bits_per_index   : entropy lower bound in bits/index (~ log2 perplexity)

    Adds losses via `add_loss`:
      - beta * ||stop(quant) - x||^2  (commitment)
      -        ||quant - stop(x)||^2  (codebook)
    """

    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kw):
        super().__init__(**kw)
        if num_embeddings <= 0 or embedding_dim <= 0 or beta <= 0:
            raise ValueError("num_embeddings>0, embedding_dim>0, beta>0 required.")
        self.K = int(num_embeddings)
        self.D = int(embedding_dim)
        self.beta = float(beta)
        self._perplexity = keras.metrics.Mean(name="vq_perplexity")
        self._usage = keras.metrics.Mean(name="vq_usage")
        self._bpi = keras.metrics.Mean(name="vq_bits_per_index")

    def build(self, input_shape):
        last = input_shape[-1]
        if last is not None and int(last) != self.D:
            raise ValueError(f"Input last dim {int(last)} != embedding_dim {self.D}")
        limit = 1.0 / max(1, self.K)
        self.codebook = self.add_weight(
            name="codebook",
            shape=(self.K, self.D),
            initializer=keras.initializers.RandomUniform(-limit, limit),
            trainable=True,
            dtype=self.variable_dtype,
        )
        super().build(input_shape)

    def call(self, x, return_indices=False):
        x = keras.ops.convert_to_tensor(x, dtype=self.compute_dtype)
        flat = keras.ops.reshape(x, (-1, self.D))
        x2 = keras.ops.sum(keras.ops.square(flat), axis=1, keepdims=True)
        e2 = keras.ops.sum(keras.ops.square(self.codebook), axis=1)
        sim = keras.ops.matmul(flat, keras.ops.transpose(self.codebook))
        dist = x2 + e2 - 2.0 * sim
        idx = keras.ops.argmax(-dist, axis=1)
        qf = keras.ops.take(self.codebook, idx, axis=0)
        q = keras.ops.reshape(qf, keras.ops.shape(x))

        q_st = keras.ops.stop_gradient(q)
        x_st = keras.ops.stop_gradient(x)
        commit = keras.ops.mean(keras.ops.square(q_st - x))
        codebk = keras.ops.mean(keras.ops.square(q - x_st))
        self.add_loss(self.beta * commit + codebk)

        one_hot = keras.ops.one_hot(idx, self.K)
        probs = keras.ops.mean(one_hot, axis=0)
        eps = keras.ops.convert_to_tensor(1e-10, dtype=self.compute_dtype)
        log_probs = keras.ops.log2(probs + eps)
        H = -keras.ops.sum(probs * log_probs)  # bits/index
        ln2 = keras.ops.log(keras.ops.convert_to_tensor(2.0, dtype=self.compute_dtype))
        perplex = keras.ops.exp(H * ln2)  # 2**H
        usage = keras.ops.sum(keras.ops.cast(probs > 0, self.compute_dtype)) / float(self.K)
        self._perplexity.update_state(perplex)
        self._bpi.update_state(H)
        self._usage.update_state(usage)

        y = x + keras.ops.stop_gradient(q - x)
        return (y, idx) if return_indices else y

    @property
    def metrics(self):  # so Model.fit logs them
        return [self._perplexity, self._usage, self._bpi]

    def get_config(self):
        return {**super().get_config(), "num_embeddings": self.K, "embedding_dim": self.D, "beta": self.beta}
