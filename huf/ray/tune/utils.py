import os
import typing as tp
from functools import partial

import gin
from ray import tune
from ray.tune.trial import Trial

from huf.io_utils import load_json_lines
from huf.objectives import DEFAULT_OBJECTIVE
from huf.ray.tune.checkpoints import load_checkpoint
from huf.types import Modes, Objective

configurable = partial(gin.configurable, module="huf.ray.tune.utils")


@configurable
def get_results(trial: Trial, checkpoint: tp.Optional[str] = None) -> None:
    del checkpoint
    assert trial.results is None
    with open(os.path.join(trial.logdir, "result.json"), "rb") as fp:
        results = load_json_lines(fp)
    if trial.results is None:
        trial.results = results
    else:
        assert trial.results == results
    return trial.results


@configurable
def full_metric_name(objective: Objective) -> str:
    return f"{objective.split}_{objective.key}"


@configurable
def get_checkpoint_score_attr(objective: Objective) -> str:
    name = full_metric_name(objective)
    if objective.mode == Modes.MIN:
        name = f"min-{name}"
    else:
        assert objective.mode == Modes.MAX
    return name


def _map_as_config_strs(config: tp.Mapping) -> tp.Iterable[str]:
    return (f"{k} = {v}" for k, v in config.items())


@configurable(allowlist=["fun"])
def configured_trainable(fun: tp.Callable = gin.REQUIRED, **kwargs):
    return fun(**kwargs)


def gin_trainable(
    config: tp.Mapping[str, tp.Any],
    *,
    checkpoint_dir=None,
    base_config_str: str = "",
    **kwargs,
):
    config_strs = [
        base_config_str,
        *_map_as_config_strs(config),
        f"checkpoint_dir = {checkpoint_dir}",
    ]
    gin.parse_config(config_strs)
    gin.finalize()
    return configured_trainable(**kwargs)


def get_best_trial(
    analysis: tune.ExperimentAnalysis,
    objective: Objective = DEFAULT_OBJECTIVE,
    scope: str = "all",
):
    metric = full_metric_name(objective)
    best_trial = analysis.get_best_trial(
        metric=metric, mode=objective.mode, scope=scope
    )
    return best_trial


@configurable
def summarize_analysis(
    analysis: tune.ExperimentAnalysis,
    objective: Objective = DEFAULT_OBJECTIVE,
    scope: str = "all",
):
    print(f"Completed {len(analysis.trials)} trials")
    best_trial = get_best_trial(analysis, objective=objective, scope=scope)
    print("best config: ", best_trial.config)
    print(best_trial.metric_analysis)


@configurable
def with_best_trial(
    analysis: tune.ExperimentAnalysis,
    fun: tp.Callable,
    objective: Objective = DEFAULT_OBJECTIVE,
    scope: str = "all",
):
    summarize_analysis(analysis, objective=objective, scope=scope)
    best_trial = get_best_trial(analysis, objective=objective, scope=scope)
    metric = full_metric_name(objective)
    best_checkpoint = analysis.get_best_checkpoint(
        best_trial, metric, mode=objective.mode
    )
    return fun(best_trial, best_checkpoint)


@configurable
def load_then(
    trial: Trial,
    checkpoint: str,
    fun: tp.Callable,
    checkpoint_loader: tp.Callable[[str], tp.Any],
):
    with gin.unlock_config():
        update_config(trial.config)

    args_or_kwargs = checkpoint_loader(checkpoint)
    if hasattr(args_or_kwargs, "items"):
        result = fun(**args_or_kwargs)
    elif hasattr(args_or_kwargs, "__iter__"):
        result = fun(*args_or_kwargs)
    else:
        result = fun(args_or_kwargs)
    return result


def update_config(config: tp.Mapping[str, tp.Any]):
    def as_config_str(key, value):
        if isinstance(value, str):
            return f"{key} = '{value}'"
        return f"{key} = {value}"

    gin.parse_config([as_config_str(k, v) for k, v in config.items()])


def reconfigure(
    base_config: tp.Optional[str], config: tp.Mapping[str, tp.Any],
):
    with gin.unlock_config():
        if base_config is not None:
            gin.parse_config(base_config)
        update_config(config)


def _maybe_item(x):
    return x.item() if hasattr(x, "item") else x


def report(_metric=None, **kwargs):
    tune.report(_maybe_item(_metric), **{k: _maybe_item(v) for k, v in kwargs.items()})


@configurable
def report_star(metrics: tp.Mapping[str, tp.Any]):
    report(**metrics)


@configurable
def report_result(fun: tp.Callable, dict_transform: tp.Callable = lambda x: x):
    result = fun()
    report_dict = dict_transform(result)
    if hasattr(result, "items"):
        report(**result)
    else:
        report(report_dict)
    return result


@configurable
def load_model_state(path: str, key: str = "state"):
    result = load_checkpoint(path)
    return {key: result.state.model_state}


@configurable
def load_fit_state(path: str, key: str = "state"):
    return {key: load_checkpoint(path).state}
