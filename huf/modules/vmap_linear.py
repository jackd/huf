import typing as tp

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

# def _as_iterable(x):
#     if hasattr(x, "__iter__"):
#         return x
# return (x,)


# def _canonicalize_axis(axis, ndim):
#     if axis < 0:
#         axis += ndim
#     if axis >= ndim:
#         raise ValueError(f"invalid axis value {axis} for {ndim} ndims")
#     return axis


# class VmapLinear(hk.Module):
#     def __init__(
#         self,
#         output_size: int,
#         with_bias: bool = True,
#         w_init: tp.Optional[hk.initializers.Initializer] = None,
#         b_init: tp.Optional[hk.initializers.Initializer] = None,
#         in_axes: int = 0,
#         out_axes: tp.Optional[int] = None,
#         name: tp.Optional[str] = None,
#     ):
#         super().__init__(name=name)
#         self.output_size = output_size
#         self.in_axes = tuple(int(x) for x in _as_iterable(in_axes))
#         if out_axes is None:
#             out_axes = self.in_axes
#         else:
#             self.out_axes = tuple(int(x) for x in _as_iterable(out_axes))
#         self.w_init = w_init
#         self.b_init = b_init or jnp.zeros
#         self.with_bias = with_bias

#     def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
#         input_shape = list(inputs.shape)
#         dtype = inputs.dtype
#         in_axes = [_canonicalize_axis(a, len(input_shape)) for a in self.in_axes]
#         in_axes.sort()
#         for a in in_axes[-1::-1]:
#             del input_shape[a]
#         input_size = input_shape[-1]
#         vmap_shape = [inputs.shape[a] for a in self.in_axes]
#         w_init = self.w_init
#         if w_init is None:
#             stddev = 1.0 / np.sqrt(self.input_size)
#             w_init = hk.initializers.TruncatedNormal(stddev=stddev)
#         w = hk.get_parameter(
#             "w", vmap_shape + [input_size, self.output_size], dtype, init=w_init
#         )

#         param_in_axes = range(len(vmap_shape))

#         def linear(inputs, w, b=None):
#             out = jnp.dot(inputs, w)
#             if b is not None:
#                 out = out + b
#             return out

#         if self.with_bias:
#             b = hk.get_parameter(
#                 "b", vmap_shape + [self.output_size], dtype, init=self.b_init
#             )
#             return jax.vmap(
#                 linear,
#                 in_axes=(in_axes, param_in_axes, param_in_axes),
#                 out_axes=self.out_axes,
#             )(inputs, w, b)
#         return jax.vmap(
#             linear, in_axes=(in_axes, param_in_axes), out_axes=self.out_axes
#         )(inputs, w)


class VmapLinear(hk.Module):
    """Parallel independent Linear transform applied across `axis=-2`."""

    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        w_init: tp.Optional[hk.initializers.Initializer] = None,
        b_init: tp.Optional[hk.initializers.Initializer] = None,
        name=None,
    ):
        super().__init__(name=name)
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros
        self.num_heads = None
        self.input_size = None

    def __call__(self, inputs):
        assert inputs.ndim >= 2
        dtype = inputs.dtype
        self.num_heads, self.input_size = inputs.shape[-2:]
        w_init = self.w_init

        if w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)

        w = hk.get_parameter(
            "w",
            shape=(self.num_heads, self.input_size, self.output_size),
            dtype=dtype,
            init=w_init,
        )
        out = jax.vmap(jnp.dot, in_axes=(-2, 0), out_axes=-2)(inputs, w)

        if self.with_bias:
            b = hk.get_parameter(
                "b",
                shape=(self.num_heads, self.output_size),
                dtype=dtype,
                init=self.b_init,
            )
            out = out + b
        return out
