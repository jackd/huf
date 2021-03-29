import typing as tp
from functools import partial

import gin

from huf.callbacks.core import Callback
from huf.models import Model
from huf.models import evaluate as _models_evaluate
from huf.models import fit as _models_fit
from huf.tune.checkpoints import load_checkpoint
from huf.types import PRNGKey

configurable = partial(gin.configurable, module="huf.tune.models")


@configurable
def fit(
    model: Model,
    rng: PRNGKey,
    train_data: tp.Iterable,
    epochs: int = 1,
    validation_data: tp.Optional[tp.Iterable] = None,
    checkpoint_dir: tp.Optional[str] = None,
    callbacks: tp.Iterable[Callback] = (),
):
    if checkpoint_dir is None:
        initial_state = None
        initial_epoch = 0
    else:
        data = load_checkpoint(checkpoint_dir)
        rng = data.rng
        initial_state = data.state
        initial_epoch = data.epoch

    return _models_fit(
        model,
        rng,
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        initial_state=initial_state,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )


def _pretty_print_map(x: tp.Mapping):
    longest_key = max((len(k) for k in x))
    for k in sorted(x):
        print(f"{k.ljust(longest_key)} = {x[k]}")


@configurable(denylist=["config", "checkpoint_dir"])
def evaluate(
    config: tp.Mapping[str, tp.Any],
    checkpoint_dir: str,
    model: Model,
    validation_data: tp.Iterable,
    callbacks: tp.Iterable[Callback] = (),
):
    print("Running evaluation with config:")
    _pretty_print_map(config)

    with gin.unlock_config():
        gin.parse_config([f"{k} = {v}" for k, v in config.items()])

    checkpoint_data = load_checkpoint(checkpoint_dir)
    print(f"Loading checkpoint data from epoch {checkpoint_data.epoch}")
    state = checkpoint_data.state
    metrics = _models_evaluate(
        model,
        state.params,
        state.net_state,
        validation_data=validation_data,
        callbacks=callbacks,
    )
    print("Finished tune evaluation")
    _pretty_print_map(metrics)
    return metrics
