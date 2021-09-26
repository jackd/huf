import typing as tp
from functools import partial

import gin
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import COO, CSC, CSR

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


def _dropout(x: jnp.ndarray, rate: float, is_training: bool):
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


@configurable
def dropout(x: tp.Union[jnp.ndarray, COO, CSR, CSC], rate: float, is_training: bool):
    fn = partial(_dropout, rate=rate, is_training=is_training)
    if isinstance(x, jnp.ndarray):
        return fn(x)
    if isinstance(x, COO):
        return COO((fn(x.data), x.row, x.col), shape=x.shape)
    if isinstance(x, CSR):
        return CSR((fn(x.data), x.indices, x.indptr), shape=x.shape)
    if isinstance(x, CSC):
        return CSC((fn(x.data), x.indices, x.indptr), shape=x.shape)
    raise TypeError(f"Unrecognized type for x `{type(x)}`")


class Linear(hk.Linear):
    """
    Equivalent to hk.Linear but supports anything with `__matmul__` defined.

    Specifically, this means `JAXSparse` are supported.
    """

    def __call__(self, inputs) -> jnp.ndarray:
        """Computes a linear transform of the input."""
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

        out = inputs @ w

        if self.with_bias:
            b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        return out
