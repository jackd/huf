import typing as tp

import jax
import tensorflow as tf

_spec_converters: tp.Mapping[type, tp.Callable[[tp.Any], jax.core.AbstractValue]] = {}


def register_spec_converter(cls: type):
    def fun(converter: tp.Callable):
        _spec_converters[cls] = converter
        return converter

    return fun


def spec_to_aval(tf_spec) -> jax.core.AbstractValue:
    fun = _spec_converters.get(type(tf_spec), None)
    if fun:
        return fun(tf_spec)
    raise ValueError(
        f"No tf_spec_to_aval function registered for spec {tf_spec}. "
        "Use `register_spec_converter` with the class."
    )


@register_spec_converter(tf.TensorSpec)
def convert_tensor_spec(spec: tf.TensorSpec) -> jax.core.ShapedArray:
    assert spec.shape.is_fully_defined(), spec.shape
    return jax.core.ShapedArray(shape=spec.shape, dtype=spec.dtype.as_numpy_dtype)
