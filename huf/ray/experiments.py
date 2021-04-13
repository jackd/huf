from functools import partial

import gin

from huf.experiments import ExperimentCallback
from huf.ray import init

configurable = partial(gin.configurable, module="huf.ray.experiments")


@configurable
class RayInit(ExperimentCallback):
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def on_start(self):
        init(**self._kwargs)
