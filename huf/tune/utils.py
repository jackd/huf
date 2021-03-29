import typing as tp
from functools import partial

import gin
from ray import tune

configurable = partial(gin.configurable, module="huf.tune.utils")


@configurable
def full_metric_name(metric_name: str, is_validation: bool = True):
    return f"{'validation' if is_validation else 'train'}_{metric_name}"


@configurable
def get_checkpoint_score_attr(
    metric_name: str = "loss", is_validation: bool = True, mode: str = "min"
):
    name = full_metric_name(metric_name, is_validation)
    if mode == "min":
        name = f"min-{name}"
    else:
        assert mode == "max"
    return name


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
        *(f"{k} = {v}" for k, v in config.items()),
        f"checkpoint_dir = {checkpoint_dir}",
    ]
    gin.parse_config(config_strs)
    gin.finalize()
    return configured_trainable(checkpoint_dir=checkpoint_dir, **kwargs)


@configurable
def summarize_analysis(
    analysis: tune.ExperimentAnalysis,
    metric_name: str = "loss",
    is_validation: bool = True,
    mode: str = "min",
    scope: str = "all",
):
    print(f"Completed {len(analysis.trials)} trials")
    metric = full_metric_name(metric_name, is_validation)
    best_trial = analysis.get_best_trial(metric=metric, mode=mode, scope=scope)
    print("best config: ", best_trial.config)
    print(best_trial.metric_analysis)


@configurable
def with_best_trial(
    analysis: tune.ExperimentAnalysis,
    fun: tp.Callable,
    metric_name: str = "loss",
    is_validation: bool = True,
    mode: str = "min",
    scope: str = "all",
):
    metric = full_metric_name(metric_name, is_validation)
    best_trial = analysis.get_best_trial(metric=metric, mode=mode, scope=scope,)
    return fun(
        best_trial.config, analysis.get_best_checkpoint(best_trial, metric, mode=mode)
    )


@configurable
def run(
    config=None,
    *,
    on_done: tp.Callable[[tune.ExperimentAnalysis], tp.Any] = summarize_analysis,
    **kwargs,
):
    analysis = tune.run(
        tune.with_parameters(
            gin_trainable, base_config_str=gin.config.config_str(), **kwargs
        ),
        config=config or {},
    )
    if on_done is None:
        return analysis
    return on_done(analysis)


for fun in (
    tune.run,
    tune.grid_search,
    tune.choice,
    tune.randint,
    tune.randn,
    tune.uniform,
):
    gin.external_configurable(fun, module="ray.tune")
