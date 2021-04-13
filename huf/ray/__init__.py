from functools import partial

import gin
import ray

external_configurable = partial(gin.external_configurable, module="ray")

init = external_configurable(ray.init)
