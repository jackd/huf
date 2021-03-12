import typing as tp
from collections import defaultdict

from huf.callbacks.core import Callback
from huf.types import Metrics, ModelState, PRNGKey


class History(Callback):
    def __init__(self):
        self.train_history = None
        self.validation_history = None
        super().__init__()

    def on_train_begin(
        self, epochs: int, steps_per_epoch: tp.Optional[int], state: ModelState
    ):
        self.train_history = defaultdict(list)
        self.validation_history = defaultdict(list)

    def on_epoch_end(
        self,
        epoch: int,
        rng: PRNGKey,
        state: ModelState,
        train_metrics: Metrics,
        validation_metrics: tp.Optional[Metrics] = None,
    ):
        for k, v in train_metrics.items():
            self.train_history[k].append(v)
        if validation_metrics is not None:
            for k, v in validation_metrics.items():
                self.validation_history[k].append(v)
