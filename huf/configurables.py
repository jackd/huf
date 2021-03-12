"""
Register gin bindings for jax, haiku and optax.

Intended to be imported from gin config files rather than python.
"""
import typing as tp

import gin
import jax
import optax

import haiku as hk

# jax configurables
for fun in (jax.random.PRNGKey, jax.random.split):
    gin.register(fun, module="jax.random")

# haiku configurables
for k in dir(hk):
    if not k.startswith("_"):
        fun = getattr(hk, k)
        if getattr(fun, "__name__", None) == k and callable(fun):
            try:
                gin.register(fun, module="hk")
            except TypeError:
                pass

# optax configurables
for k in dir(optax):
    if not k.startswith("_"):
        fun = getattr(optax, k)
        if getattr(fun, "__name__", None) == k and callable(fun):
            gin.register(fun, module="optax")
        del fun
    del k


@gin.register(module="optax")
def chain_star(transforms: tp.Iterable):
    """optax.chain(*transforms) - with a signature that's easier to configure."""
    return optax.chain(*transforms)
