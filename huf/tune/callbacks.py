import typing as tp
from functools import partial

import gin
from ray import tune
from ray.tune import checkpoint_manager as chkpt_lib
from ray.tune.trial import checkpoint_deleter

from huf.callbacks.core import Callback
from huf.tune.checkpoints import CheckpointData, save_checkpoint
from huf.types import Metrics, ModelState, PRNGKey

configurable = partial(gin.configurable, module="huf.tune.callbacks")


def get_combined_metrics(
    train_metrics: Metrics, validation_metrics: tp.Optional[Metrics] = None
):
    metrics = {f"train_{k}": v for k, v in train_metrics.items()}
    if validation_metrics is not None:
        metrics.update({f"val_{k}": v for k, v in validation_metrics.items()})
    return metrics


def split_combined_metrics(metrics: Metrics):
    train_metrics = {}
    validation_metrics = {}
    for k, v in metrics.items():
        if k.startswith("train_"):
            train_metrics[k[len("train_")] :] = v
        elif k.startswith("val_"):
            validation_metrics[k[len("val_")] :] = v
        else:
            raise KeyError(
                f"Invalid combined metrics key {k}: "
                "should start with 'train_' or 'validation_"
            )
    return train_metrics, validation_metrics


@configurable
class EpochReporter(Callback):
    def on_epoch_end(
        self,
        epoch: int,
        rng: PRNGKey,
        state: ModelState,
        train_metrics: Metrics,
        validation_metrics: tp.Optional[Metrics] = None,
    ):
        metrics = get_combined_metrics(train_metrics, validation_metrics)
        metrics = {k: float(v) for k, v in metrics.items()}
        tune.report(**metrics)


@configurable
class Checkpointer(Callback):
    def __init__(
        self,
        checkpoint_freq: int,
        checkpoint_score_attr: str = "min-val_loss",
        keep_checkpoints_num: int = 1,
        storage: str = chkpt_lib.Checkpoint.PERSISTENT,
    ):
        self.manager = chkpt_lib.CheckpointManager(
            keep_checkpoints_num=keep_checkpoints_num,
            checkpoint_score_attr=checkpoint_score_attr,
            delete_fn=checkpoint_deleter(tune.get_trial_id(), None),
        )
        self.storage = storage
        self.checkpoint_freq = checkpoint_freq

    def on_epoch_end(
        self,
        epoch: int,
        rng: PRNGKey,
        state: ModelState,
        train_metrics: Metrics,
        validation_metrics: tp.Optional[Metrics] = None,
    ):
        if self.checkpoint_freq and epoch % self.checkpoint_freq == 0:
            value = CheckpointData(epoch, rng, state)
            if self.storage == chkpt_lib.Checkpoint.PERSISTENT:
                value = save_checkpoint(value)
            else:
                assert self.storage == chkpt_lib.Checkpoint.MEMORY
            result = get_combined_metrics(train_metrics, validation_metrics)
            chkpt = chkpt_lib.Checkpoint(self.storage, value, result)
            self.manager.on_checkpoint(chkpt)
