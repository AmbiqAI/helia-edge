import keras
from helia_edge.layers.gumbel_softmax_bottleneck import GumbelSoftmaxBottleneck

class GSAutoencoder(keras.Model):
    """
    Convenience wrapper around (encoder -> GumbelSoftmaxBottleneck -> decoder).

    - Supports extra reconstruction-side losses and metrics.
    - Can return discrete code indices and/or code probabilities from the bottleneck.
    - Exposes Gumbel-Softmax layer metrics alongside base model metrics.
    """
    def __init__(self, encoder: keras.Model, gs: GumbelSoftmaxBottleneck, decoder: keras.Model, **kwargs):
        """Initialize the Gumbel-Softmax autoencoder.

        Args:
            encoder: Encoder model producing continuous latents.
            gs: GumbelSoftmaxBottleneck layer that discretizes latents.
            decoder: Decoder model mapping bottleneck outputs to reconstructions.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.gs      = gs
        self.decoder = decoder

        self._recon_loss = None
        self._extra_loss_fns = []
        self._extra_metric_objs = []
        self._extra_metric_fns  = []

    def call(self, x, training=False, return_indices: bool = False, return_probs: bool = False):
        """Run encoder -> GS bottleneck -> decoder.

        Args:
            x: Input batch.
            training: Whether to run in training mode (affects encoder/decoder/gs).
            return_indices: If True, also return the discrete code indices.
            return_probs: If True, also return code probabilities.

        Returns:
            Reconstruction, optionally with indices and/or probabilities.
        """
        z = self.encoder(x, training=training)
        y = self.gs(z, training=training, return_indices=return_indices, return_probs=return_probs)
        if return_indices and return_probs:
            zq, idx, prob = y
        elif return_indices:
            zq, idx = y
            prob = None
        elif return_probs:
            zq, prob = y
            idx = None
        else:
            zq = y
            idx = prob = None
        out = self.decoder(zq, training=training)
        if return_indices and return_probs:
            return out, idx, prob
        if return_indices:
            return out, idx
        if return_probs:
            return out, prob
        return out

    def compile(
        self,
        optimizer: keras.optimizers.Optimizer,
        loss: keras.losses.Loss | None = None,
        metrics: list | None = None,
        extra_losses: list | None = None,
        extra_metrics: list | None = None,
        **kwargs,
    ):
        """Compile with optional extra losses/metrics.

        Args:
            optimizer: Keras optimizer.
            loss: Base reconstruction loss (e.g., keras.losses.MeanSquaredError()).
            metrics: Standard Keras metrics (Metric instances or callables).
            extra_losses: List of callables (y_true, y_pred) -> scalar to add to loss.
            extra_metrics: Metric instances or callables (y_true, y_pred) -> scalar.
        """
        super().compile(optimizer=optimizer, metrics=metrics or [], **kwargs)
        self._recon_loss     = loss
        self._extra_loss_fns = list(extra_losses or [])

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
        total = keras.ops.convert_to_tensor(0.0, dtype=self.compute_dtype)

        # base recon loss
        if self._recon_loss is not None and y is not None and y_pred is not None:
            if sample_weight is not None:
                total = total + self._recon_loss(y, y_pred, sample_weight=sample_weight)
            else:
                total = total + self._recon_loss(y, y_pred)

        # extra user losses
        for fn in self._extra_loss_fns:
            total = total + fn(y, y_pred)

        # include layer/model-added losses (e.g., KL from GS layer, regularizers)
        for loss in self.losses:
            total = total + loss

        return total

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        """Update compiled metrics plus extra metric trackers."""
        results = super().compute_metrics(x, y, y_pred, sample_weight)
        for tracker, fn in self._extra_metric_fns:
            tracker.update_state(fn(y, y_pred))
            results[tracker.name] = tracker.result()
        return results

    @property
    def metrics(self):
        # base model metrics + GS internal metrics + extra metric trackers
        return super().metrics + self.gs.metrics + self._extra_metric_objs

    def get_config(self):
        """Return serialized config for saving/loading."""
        config = super().get_config()
        config.update(
            {
                "encoder": keras.saving.serialize_keras_object(self.encoder),
                "gs": keras.saving.serialize_keras_object(self.gs),
                "decoder": keras.saving.serialize_keras_object(self.decoder),
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Recreate model from serialized config."""
        cfg = dict(config)
        encoder_cfg = cfg.pop("encoder")
        gs_cfg = cfg.pop("gs")
        decoder_cfg = cfg.pop("decoder")
        encoder = keras.saving.deserialize_keras_object(
            encoder_cfg, custom_objects=custom_objects
        )
        gs = keras.saving.deserialize_keras_object(gs_cfg, custom_objects=custom_objects)
        decoder = keras.saving.deserialize_keras_object(
            decoder_cfg, custom_objects=custom_objects
        )
        return cls(encoder=encoder, gs=gs, decoder=decoder, **cfg)
