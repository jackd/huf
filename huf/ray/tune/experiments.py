import typing as tp
from functools import partial

import gin
from ray import tune

from huf import experiments
from huf.ray.tune.configurables import run
from huf.ray.tune.utils import reconfigure, report

configurable = partial(gin.configurable, module="huf.ray.tune.experiments")
Result = tp.Mapping[str, tp.Any]


@configurable
class Reporter(experiments.ExperimentCallback):
    def on_done(self, result: tp.Union[Result, tp.Iterable[tp.Mapping[str, tp.Any]]]):
        if hasattr(result, "items"):
            report(**result)
        for r in result:
            report(**r)


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
