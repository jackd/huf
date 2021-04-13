from functools import partial

import gin
from ray import tune

external_configurable = partial(gin.external_configurable, module="ray.tune")

run = external_configurable(tune.run)
grid_search = external_configurable(tune.grid_search)
choice = external_configurable(tune.choice)
randint = external_configurable(tune.randint)
randn = external_configurable(tune.randn)
uniform = external_configurable(tune.uniform)
with_parameters = external_configurable(tune.with_parameters)
