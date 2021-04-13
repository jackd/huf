import typing as tp

from huf.types import FitResult, FitState, Metrics

Model = tp.Any  # importing huf.models creates circular imports


class Callback:
    @property
    def model(self) -> tp.Optional[Model]:
        return getattr(self, "_model", None)

    @model.setter
    def model(self, model):
        self._model = model

    def on_epoch_begin(self, state: FitState):
        pass

    def on_epoch_end(self, result: FitResult):
        pass

    def on_train_begin(self, epochs: int, steps_per_epoch: tp.Optional[int]):
        pass

    def on_train_end(self, result: FitResult):
        pass

    def on_train_step_begin(self, step: int):
        pass

    def on_train_step_end(self, step: int, metrics: Metrics):
        pass

    def on_test_begin(self):
        pass

    def on_test_end(self, metrics: Metrics):
        pass

    def on_test_step_begin(self, step: int):
        pass

    def on_test_step_end(self, step: int, metrics: Metrics):
        pass
