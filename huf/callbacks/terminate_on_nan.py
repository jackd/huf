import jax.numpy as jnp

from huf.callbacks.core import Callback
from huf.types import Metrics


class TerminateOnNaN(Callback):
    def on_train_step_end(self, step: int, metrics: Metrics):
        loss = metrics["loss"]
        if jnp.isnan(loss):
            raise ValueError("nan detected in loss")
