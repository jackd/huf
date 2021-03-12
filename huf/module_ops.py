import typing as tp
from functools import partial

import gin
import jax
import jax.numpy as jnp

import haiku as hk
from huf.types import SampleWeight

configurable = partial(gin.configurable, module="huf.module_ops")


@configurable
def mean(value, sample_weight: tp.Optional[SampleWeight] = None):
    if sample_weight is not None:
        assert sample_weight.shape == value.shape
    summed = hk.get_state("sum", shape=(), dtype=value.dtype, init=jnp.zeros)
    weight = hk.get_state(
        "weight",
        shape=(),
        dtype=jnp.int32 if sample_weight is None else sample_weight.dtype,
        init=jnp.zeros,
    )
    if sample_weight is None:
        if value.ndim == 0:
            summed = summed + value
            weight = weight + 1
        else:
            summed = summed + jnp.sum(value, axis=0)
            weight = weight + value.shape[0]
    else:
        if value.ndim == 0:
            summed = summed + value
            weight = weight + sample_weight
        else:
            summed = summed + jnp.sum(value * sample_weight, axis=0)
            weight = weight + jnp.sum(sample_weight)
    hk.set_state("sum", summed)
    hk.set_state("weight", weight)
    return summed / weight


@configurable
class Mean(hk.Module):
    def __call__(self, value):
        return mean(value)


@configurable
def dropout(x: jnp.ndarray, rate: float, is_training: bool):
    if not rate:
        return x

    if isinstance(is_training, bool):
        if is_training and rate:
            return hk.dropout(hk.next_rng_key(), rate, x)
        return x

    def if_is_training(operand):
        x, key = operand
        return hk.dropout(key, rate, x)

    def otherwise(operand):
        x, key = operand
        del key
        return x

    return jax.lax.cond(is_training, if_is_training, otherwise, (x, hk.next_rng_key()))
