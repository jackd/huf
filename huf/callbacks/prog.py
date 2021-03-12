import typing as tp

import tqdm

from huf.callbacks.core import Callback
from huf.types import Metrics, ModelState, PRNGKey


def _metrics_str(metrics: Metrics):
    return ", ".join((f"{k}: {metrics[k]:.5f}" for k in sorted(metrics)))


class ProgbarLogger(Callback):
    def __init__(self, print_fun: tp.Callable[[str], None] = print):
        self.print = print_fun
        self._steps_per_epoch = None
        self._prog = None
        self._epochs = None
        self._prefix = None

    def on_train_begin(
        self, epochs: int, steps_per_epoch: tp.Optional[int], state: ModelState
    ):
        self._epochs = epochs
        self._steps_per_epoch = steps_per_epoch

    def on_epoch_begin(self, epoch: int, rng: PRNGKey, state: ModelState):
        self._prefix = f"Epoch {epoch + 1} / {self._epochs}"
        self._prog = tqdm.tqdm(desc=self._prefix, total=self._steps_per_epoch)

    def on_epoch_end(
        self,
        epoch: int,
        rng: PRNGKey,
        state: ModelState,
        train_metrics: Metrics,
        validation_metrics: tp.Optional[Metrics] = None,
    ):
        self._prog.close()
        if validation_metrics is not None:
            self.print(f"validation_metrics: {_metrics_str(validation_metrics)}")

    def on_train_end(
        self,
        epochs: int,
        rng: PRNGKey,
        state: ModelState,
        train_metrics: Metrics,
        validation_metrics: tp.Optional[Metrics] = None,
    ):
        self._prog.close()
        self._prog = None
        self._epochs = None
        self._prefix = None
        self._steps_per_epoch = None

    def on_train_step_end(self, step: int, metrics: Metrics):
        self._prog.n = step + 1
        self._prog.set_description(f"{self._prefix}: {_metrics_str(metrics)}")


class EpochProgbarLogger(Callback):
    """Like ProgbarLogger but ticks on each epoch end, rather than each train step."""

    def __init__(self, print_fun: tp.Callable[[str], None] = print):
        self.print = print_fun
        self._prog = None
        self._epochs = None

    def on_train_begin(
        self, epochs: int, steps_per_epoch: tp.Optional[int], state: ModelState
    ):
        self._epochs = epochs
        self._prog = tqdm.tqdm(total=self._epochs)

    def on_epoch_end(
        self,
        epoch: int,
        rng: PRNGKey,
        state: ModelState,
        train_metrics: Metrics,
        validation_metrics: tp.Optional[Metrics] = None,
    ):
        self._prog.n = epoch + 1
        metrics = train_metrics.copy()
        metrics.update({f"val_{k}": v for k, v in validation_metrics.items()})
        self._prog.set_description(_metrics_str(metrics))

    def on_train_end(
        self,
        epochs: int,
        rng: PRNGKey,
        state: ModelState,
        train_metrics: Metrics,
        validation_metrics: tp.Optional[Metrics] = None,
    ):
        self._prog.close()
        self._prog = None
        self._epochs = None
