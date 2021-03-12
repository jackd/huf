import jax.numpy as jnp

import haiku as hk

glorot_uniform = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
glorot_normal = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
lecun_uniform = hk.initializers.VarianceScaling(1.0, "fan_in", "uniform")
lecun_normal = hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal")
he_uniform = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
he_normal = hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")


# default pytorch initializers
pytorch_linear_kernel_initializer = hk.initializers.VarianceScaling(
    1 / 3, "fan_in", "uniform"
)


def pytorch_linear_bias_initializer(fan_in: int):
    limit = jnp.asarray(fan_in, jnp.float32) ** -0.5
    return hk.initializers.RandomUniform(-limit, limit)
