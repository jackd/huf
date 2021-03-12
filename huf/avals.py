import typing as tp

import jax
import jax.numpy as jnp

from huf.types import AbstractTree

_zeros_like = {}


def register_zeros(cls: type) -> tp.Callable[[tp.Callable], tp.Callable]:
    def fun(zeros_fun: tp.Callable[[jax.core.AbstractValue], tp.Any]):
        _zeros_like[cls] = zeros_fun
        return zeros_fun

    return fun


def zeros_like(aval: jax.core.AbstractValue):
    for typ in type(aval).mro():
        fun = _zeros_like.get(typ, None)
        if fun:
            return fun(aval)
    raise TypeError(
        f"No zeros function registered for aval {type(aval)}. "
        "Custom `AbstractValue`s classes can be registered with "
        "`huf.avals.register_zeros`"
    )


register_zeros(jax.core.ShapedArray)(lambda x: jnp.zeros(x.shape, x.dtype))


def assert_compatible(a: AbstractTree, b: AbstractTree):
    a_flat, a_def = jax.tree_util.tree_flatten(a)
    b_flat, b_def = jax.tree_util.tree_flatten(b)
    if a_def != b_def:
        raise ValueError(f"tree structures not the same, a={a_def}, b={b_def}")
    for i, (ai, bi) in enumerate(zip(a_flat, b_flat)):
        if ai != bi:
            raise ValueError(f"entry {i} incompatible, {ai} vs {bi}")


def is_compatible(a: AbstractTree, b: AbstractTree) -> bool:
    try:
        assert_compatible(a, b)
        return True
    except ValueError:
        return False


def abstract_tree(values_tree) -> AbstractTree:
    return jax.tree_util.tree_map(lambda x: x.aval, values_tree)


def abstract_eval(fun, *args, **kwargs):
    args_flat, in_tree = jax.api.tree_flatten((args, kwargs))
    wrapped_fun, out_tree = jax.api.flatten_fun(jax.linear_util.wrap_init(fun), in_tree)
    out = jax.interpreters.partial_eval.abstract_eval_fun(
        wrapped_fun.call_wrapped, *args_flat
    )

    return jax.tree_util.tree_unflatten(out_tree(), out)
