from functools import partial

import gin
from ray import tune
from ray.tune import checkpoint_manager as chkpt_lib
from ray.tune.trial import checkpoint_deleter

from huf.callbacks.core import Callback
from huf.metrics import get_combined_metrics
from huf.objectives import DEFAULT_OBJECTIVE
from huf.ray.tune.checkpoints import save_checkpoint
from huf.ray.tune.utils import get_checkpoint_score_attr
from huf.types import FitResult, Objective

configurable = partial(gin.configurable, module="huf.ray.tune.callbacks")


@configurable
def report_metrics(train_metrics, validation_metrics, test_metrics=None):
    metrics = get_combined_metrics(train_metrics, validation_metrics, test_metrics)
    tune.report(
        **{k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}
    )


@configurable
class EpochReporter(Callback):
    """Report results `on_epoch_end`."""

    def on_epoch_end(self, result: FitResult):
        report_metrics(result.train_metrics, result.validation_metrics)


@configurable
class FitReporter(Callback):
    """Report results `on_train_end`."""

    def on_train_end(self, result: FitResult):
        report_metrics(result.train_metrics, result.validation_metrics)


@configurable
class Checkpointer(Callback):
    def __init__(
        self,
        checkpoint_freq: int = 1,
        objective: Objective = DEFAULT_OBJECTIVE,
        keep_checkpoints_num: int = 1,
        storage: str = chkpt_lib.Checkpoint.PERSISTENT,
    ):
        self.objective = objective
        self.manager = chkpt_lib.CheckpointManager(
            keep_checkpoints_num=keep_checkpoints_num,
            checkpoint_score_attr=get_checkpoint_score_attr(objective),
            delete_fn=checkpoint_deleter(tune.get_trial_id(), None),
        )
        self.storage = storage
        self.checkpoint_freq = checkpoint_freq

    def on_epoch_end(self, result: FitResult):
        if self.checkpoint_freq and result.state.epochs % self.checkpoint_freq == 0:
            combined_metrics = get_combined_metrics(
                result.train_metrics, result.validation_metrics
            )
            if self.storage == chkpt_lib.Checkpoint.PERSISTENT:
                result = save_checkpoint(result)
            else:
                assert self.storage == chkpt_lib.Checkpoint.MEMORY
            print(combined_metrics)
            chkpt = chkpt_lib.Checkpoint(self.storage, result, combined_metrics)
            self.manager.on_checkpoint(chkpt)
