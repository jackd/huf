import gin

from . import (
    callbacks,
    data,
    losses,
    metrics,
    models,
    module_ops,
    objectives,
    ops,
    types,
)

objective = gin.external_configurable(objectives.objective, module="huf")

__all__ = [
    "callbacks",
    "data",
    "losses",
    "models",
    "metrics",
    "module_ops",
    "ops",
    "types",
    "objective",
]
