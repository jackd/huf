import typing as tp
from collections import defaultdict

from huf.callbacks.core import Callback
from huf.types import FitResult


class History(Callback):
    def __init__(self):
        self.train_history = None
        self.validation_history = None
        super().__init__()

    def on_train_begin(self, epochs: int, steps_per_epoch: tp.Optional[int]):
        self.train_history = defaultdict(list)
        self.validation_history = defaultdict(list)

    def on_epoch_end(self, result: FitResult):
        for k, v in result.train_metrics.items():
            self.train_history[k].append(v)
        if result.validation_metrics is not None:
            for k, v in result.validation_metrics.items():
                self.validation_history[k].append(v)
