import typing as tp
import warnings

import jax.numpy as jnp
from huf.callbacks.core import Callback
from huf.errors import FitInterrupt
from huf.objectives import DEFAULT_OBJECTIVE
from huf.types import FitResult, Modes, Objective, Splits


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.

    Assuming the goal of a training is to minimize the loss. With this, the
    metric to be monitored would be 'loss', and mode would be 'min'. A
    `model.fit()` training loop will check at end of every epoch whether
    the loss is no longer decreasing, considering the `min_delta` and
    `patience` if applicable. Once it's found no longer decreasing,
    `model.stop_training` is marked True and the training terminates.

    The quantity to be monitored needs to be available in `logs` dict.
    This will be
    """

    def __init__(
        self,
        objective: Objective = DEFAULT_OBJECTIVE,
        min_delta: float = 0.0,
        patience: int = 0,
        verbose: int = 0,
        baseline: tp.Optional[float] = None,
        restore_best: bool = False,
    ):
        """Initialize an EarlyStopping callback.

        Arguments:
            objective: Objective to be monitored.
            min_delta: Minimum change in the monitored quantity
                to qualify as an improvement, i.e. an absolute
                change of less than min_delta, will count as no
                improvement.
            patience: Number of epochs with no improvement
                after which training will be stopped.
            verbose: verbosity mode.
            baseline: Baseline value for the monitored quantity.
                Training will stop if the model doesn't show improvement over the
                baseline.
            restore_best: Whether to return `FitState` from
                the epoch with the best value of the monitored quantity.
                If False, the model result obtained at the last step of
                training is used.
        """
        super().__init__()
        self.objective = objective
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_step = 0
        self.restore_best = restore_best

        self.best = None
        self.best_result = None

        self.monitor_op = Modes.comparator(objective.mode)

        if self.monitor_op is jnp.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, epochs: int, steps_per_epoch: tp.Optional[int]):
        del epochs, steps_per_epoch
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_step = 0
        self.best_result = None
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = Modes.default_value(self.objective.mode)

    def on_epoch_end(self, result: FitResult):
        current = self.get_monitor_value(
            result.train_metrics, result.validation_metrics
        )
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best:
                self.best_result = result
        else:
            self.wait += 1
            if self.wait >= self.patience:
                raise FitInterrupt(self.best_result)

    def get_monitor_value(self, train_metrics, validation_metrics):
        if self.objective.split == Splits.VALIDATION:
            if validation_metrics is None:
                warnings.warn(
                    "EarlyStopping condition on `validation_metrics` which is `None`"
                )
            metrics = validation_metrics
        else:
            metrics = train_metrics
        monitor_value = metrics.get(self.objective.key)
        if monitor_value is None:
            warnings.warn(
                f"Early stopping conditioned on `{self.objective.mode}_metrics["
                f"{self.objective.key}]` which is not available. "
                f"Available `{self.objective.mode}_metrics` are: "
                ", ".join(metrics.keys())
            )

        return monitor_value
