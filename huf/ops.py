import typing as tp
from functools import partial

import gin
import jax
import jax.numpy as jnp

configurable = partial(gin.configurable, module="huf.ops")

EPSILON = 1e-7


@configurable
def sparse_categorical_accuracy(labels: jnp.ndarray, preds: jnp.ndarray):
    assert labels.ndim == preds.ndim - 1, (labels.shape, preds.shape)
    return jnp.argmax(preds, axis=-1) == labels


@configurable
def mean_sparse_categorical_accuracy(
    labels: jnp.ndarray,
    preds: jnp.ndarray,
    sample_weights: tp.Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    return weighted_mean(sparse_categorical_accuracy(labels, preds), sample_weights)


@configurable
def sparse_categorical_crossentropy(
    labels: jnp.ndarray, preds: jnp.ndarray, from_logits: bool = False,
) -> jnp.ndarray:
    if from_logits:
        preds = jax.nn.log_softmax(preds)
        loss = -jnp.take_along_axis(preds, labels[..., None], axis=-1)[..., 0]
    else:
        # select output value
        preds = jnp.take_along_axis(preds, labels[..., None], axis=-1)[..., 0]

        # calculate log
        y_pred = jnp.maximum(preds, EPSILON)
        y_pred = jnp.log(y_pred)
        loss = -y_pred

    return loss


@configurable
def mean_sparse_categorical_crossentropy(
    labels: jnp.ndarray,
    preds: jnp.ndarray,
    sample_weight: tp.Optional[jnp.ndarray] = None,
    from_logits: bool = False,
) -> jnp.ndarray:
    return weighted_mean(
        sparse_categorical_crossentropy(labels, preds, from_logits=from_logits),
        sample_weight,
    )


@configurable
def weighted_mean(value, sample_weight=None):
    if sample_weight is None:
        return value.mean()
    return (value * sample_weight).sum() / sample_weight.sum()


@configurable
def weighted_mean_fun(fun: tp.Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
    def ret_f(labels, preds, sample_weight=None):
        return weighted_mean(fun(labels, preds), sample_weight)

    return ret_f
