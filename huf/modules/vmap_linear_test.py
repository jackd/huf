import haiku as hk
import jax
import jax.numpy as jnp
from absl.testing import absltest
from jax import test_util as jtu

from huf.modules.vmap_linear import VmapLinear


class VmapLinearTest(jtu.JaxTestCase):
    def test_vmap_linear(self):
        leading_dims = (7, 11)
        input_size = 3
        output_size = 5
        num_heads = 13
        dtype = jnp.float32

        def f(x):
            return VmapLinear(output_size)(x)

        transform = hk.transform(f)

        k0, k1 = jax.random.split(jax.random.PRNGKey(0))
        x = jax.random.normal(
            k0, shape=leading_dims + (num_heads, input_size), dtype=dtype
        )
        params = transform.init(k1, x)
        out = transform.apply(params, None, x)
        self.assertEqual(out.shape, leading_dims + (num_heads, output_size))
        w = params["vmap_linear"]["w"]
        b = params["vmap_linear"]["b"]
        expected = jax.vmap(
            lambda i, w, b: jnp.dot(i, w) + b, in_axes=(-2, 0, 0), out_axes=-2
        )(x, w, b)
        self.assertAllClose(out, expected)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
