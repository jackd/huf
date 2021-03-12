import jax
import jax.numpy as jnp

from huf.types import Spec

ShapeDtypeStruct = jax.ShapeDtypeStruct


def to_spec(x) -> Spec:
    return jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), x)


def zeros_like(spec: Spec):
    return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), spec)
