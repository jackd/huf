import typing as tp

from huf.types import Metrics, ModelState, PRNGKey


class Callback:
    def on_epoch_begin(self, epoch: int, rng: PRNGKey, state: ModelState):
        pass

    def on_epoch_end(
        self,
        epoch: int,
        rng: PRNGKey,
        state: ModelState,
        train_metrics: Metrics,
        validation_metrics: tp.Optional[Metrics] = None,
    ):
        pass

    def on_train_begin(
        self, epochs: int, steps_per_epoch: tp.Optional[int], state: ModelState
    ):
        pass

    def on_train_end(
        self,
        epochs: int,
        rng: PRNGKey,
        state: ModelState,
        train_metrics: Metrics,
        validation_metrics: tp.Optional[Metrics] = None,
    ):
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
