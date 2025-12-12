from __future__ import annotations
import keras
from helia_edge.layers.vector_quantizer import VectorQuantizer


class VQAutoencoder(keras.Model):
    """
    Convenience wrapper around (encoder -> VectorQuantizer -> decoder).

    - Supports extra reconstruction-side losses and metrics.
    - Can return discrete code indices from the VQ bottleneck.
    - Exposes VQ layer metrics alongside base model metrics.
    """
    def __init__(self, encoder: keras.Model, vq: VectorQuantizer, decoder: keras.Model, **kwargs):
        """Initialize the vector-quantized autoencoder.

        Args:
            encoder: Encoder model producing continuous latents.
            vq: VectorQuantizer layer that discretizes latents.
            decoder: Decoder model mapping bottleneck outputs to reconstructions.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.vq      = vq
        self.decoder = decoder

        self._recon_loss = None
        self._extra_loss_fns = []
        self._extra_metric_objs = []        # Metric trackers
        self._extra_metric_fns  = []        # (tracker, callable) pairs

    def call(self, x, training=False, return_indices: bool = False):
        """Run encoder -> VQ bottleneck -> decoder.

        Args:
            x: Input batch.
            training: Whether to run in training mode (affects encoder/decoder/VQ).
            return_indices: If True, also return discrete code indices.

        Returns:
            Reconstruction, optionally with indices.
        """
        z = self.encoder(x, training=training)
        v = self.vq(z, return_indices=return_indices)
        if return_indices:
            zq, indices = v
        else:
            zq, indices = v, None
        y = self.decoder(zq, training=training)
        return (y, indices) if return_indices else y

    def compile(
        self,
        optimizer: keras.optimizers.Optimizer,
        loss: keras.losses.Loss | None = None,
        metrics: list | None = None,
        extra_losses: list | None = None,
        extra_metrics: list | None = None,
        **kwargs,
    ):
        """
        Compile with optional extra losses/metrics.

        Args:
          optimizer: Keras optimizer
          loss: base reconstruction loss (e.g., keras.losses.MeanSquaredError())
          metrics: standard Keras metrics (Metric instances or callables)
          extra_losses: list[Callable(y_true, y_pred) -> scalar]
          extra_metrics: list of Metric OR Callable(y_true, y_pred) -> scalar
        """
        super().compile(optimizer=optimizer, metrics=metrics or [], **kwargs)
        self._recon_loss     = loss
        self._extra_loss_fns = list(extra_losses or [])

        # Normalize extra_metrics into Metric objects
        self._extra_metric_objs.clear()
        self._extra_metric_fns.clear()
        for m in (extra_metrics or []):
            if isinstance(m, keras.metrics.Metric):
                self._extra_metric_objs.append(m)
            else:
                name = getattr(m, "__name__", "extra_metric")
                tracker = keras.metrics.Mean(name=name)
                self._extra_metric_objs.append(tracker)
                self._extra_metric_fns.append((tracker, m))

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False):
        """Compute total loss = recon + extra losses + layer-added losses."""
        # Base reconstruction loss (respect sample_weight if provided)
        total = keras.ops.convert_to_tensor(0.0, dtype=self.compute_dtype)
        if self._recon_loss is not None and y is not None and y_pred is not None:
            if sample_weight is not None:
                total = total + self._recon_loss(y, y_pred, sample_weight=sample_weight)
            else:
                total = total + self._recon_loss(y, y_pred)

        # Extra user-defined losses
        for fn in self._extra_loss_fns:
            total = total + fn(y, y_pred)

        # Include layer-added losses (e.g., VQ commitment/codebook + any regularizers)
        if self.losses:
            total = total + keras.ops.add_n(self.losses)

        return total

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        """Update compiled metrics plus extra metric trackers."""
        # Update any compiled metrics (e.g., keras.metrics.MeanSquaredError())
        results = super().compute_metrics(x, y, y_pred, sample_weight)

        # Update extra metric trackers that wrap callables
        for tracker, fn in self._extra_metric_fns:
            val = fn(y, y_pred)
            tracker.update_state(val)
            results[tracker.name] = tracker.result()
        return results

    @property
    def metrics(self):
        # Expose: base model metrics + VQ metrics + extra metric trackers
        return super().metrics + self.vq.metrics + self._extra_metric_objs
