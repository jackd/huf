import typing as tp
from functools import partial

import gin
import jax.numpy as jnp

import haiku as hk
from huf import module_ops, ops
from huf.types import Labels, Preds, SampleWeight

configurable = partial(gin.configurable, module="huf.metrics")


@configurable
class Mean(hk.Module):
    def __init__(self, fun: tp.Callable[..., jnp.ndarray], name=None, **kwargs):
        super().__init__(name=name)
        self._fun = fun
        self._kwargs = kwargs

    def __call__(
        self, labels: Labels, preds: Preds, sample_weight: tp.Optional[SampleWeight]
    ):
        value = self._fun(labels, preds, **self._kwargs)
        return module_ops.mean(value, sample_weight)


@configurable
class SparseCategoricalAccuracy(Mean):
    def __init__(self, name=None):
        super().__init__(fun=ops.sparse_categorical_accuracy, name=name)


@configurable
class SparseCategoricalCrossentropy(Mean):
    def __init__(self, from_logits: bool = False, name=None):
        super().__init__(
            fun=ops.sparse_categorical_crossentropy, name=name, from_logits=from_logits
        )
