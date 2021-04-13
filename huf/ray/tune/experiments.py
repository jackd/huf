import typing as tp
from functools import partial

import gin
from ray import tune

from huf import experiments
from huf.ray.tune.configurables import run
from huf.ray.tune.utils import reconfigure, report, summarize_analysis

configurable = partial(gin.configurable, module="huf.ray.tune.experiments")


@configurable
class Reporter(experiments.ExperimentCallback):
    def on_done(self, result: tp.Mapping[str, tp.Any]):
        report(**result)


@configurable
class Summarizer(experiments.ExperimentCallback):
    def report(self, result: tune.ExperimentAnalysis):
        summarize_analysis(result)


@configurable
def run_many(
    *, config: tp.Optional[tp.Mapping] = None, **kwargs,
) -> tune.ExperimentAnalysis:
    def trainable(config, _base_config):
        reconfigure(_base_config, config)
        return experiments.run()

    return run(
        tune.with_parameters(trainable, _base_config=gin.config.config_str(), **kwargs),
        config=config,
    )
