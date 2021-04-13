import typing as tp

from huf.callbacks.core import Callback
from huf.data import as_dataset
from huf.types import FitResult


class TestOnEnd(Callback):
    def __init__(self, dataset: tp.Iterable, on_done: tp.Callable):
        self.data = as_dataset(dataset)
        self.on_done = on_done

    def on_train_end(self, result: FitResult):
        metrics = self.model.evaluate(result.state.model_state, self.data)
        self.on_done(metrics)
