import typing as tp
from collections import defaultdict
from functools import partial

import gin
import jax.numpy as jnp

import haiku as hk
from huf import module_ops, ops
from huf.types import Labels, Metrics, Preds, SampleWeight, Splits

configurable = partial(gin.configurable, module="huf.metrics")


@configurable
def get_combined_metrics(
    train_metrics: tp.Optional[Metrics] = None,
    validation_metrics: tp.Optional[Metrics] = None,
    test_metrics: tp.Optional[Metrics] = None,
):
    metrics = {}
    for split, mets in (
        (Splits.TRAIN, train_metrics),
        (Splits.VALIDATION, validation_metrics),
        (Splits.TEST, test_metrics),
    ):
        if mets is not None:
            metrics.update({f"{split}_{k}": v for k, v in mets.items()})
    return metrics


@configurable
def split_combined_metrics(
    metrics: Metrics, splits: tp.Iterable[str] = Splits.all()
) -> tp.Mapping[str, Metrics]:
    all_metrics = defaultdict(dict)
    for k, v in metrics.items():
        for split in splits:
            if k.startswith(f"{split}_"):
                all_metrics[split][len(split) + 1 :] = v
                break
        else:
            raise KeyError(
                f"Invalid combined metrics key {k}: "
                f"should start with one of {Splits.all()}"
            )
    return all_metrics


@configurable
class Mean(hk.Module):
    def __init__(self, fun: tp.Callable[..., jnp.ndarray], name=None, **kwargs):
        super().__init__(name=name)
        self._fun = fun
        self._kwargs = kwargs

    def __call__(
        self, labels: Labels, preds: Preds, sample_weight: tp.Optional[SampleWeight]
    ):
        value = self._fun(labels, preds, **self._kwargs)
        return module_ops.mean(value, sample_weight)


@configurable
class SparseCategoricalAccuracy(Mean):
    def __init__(self, name=None):
        super().__init__(fun=ops.sparse_categorical_accuracy, name=name)


@configurable
class SparseCategoricalCrossentropy(Mean):
    def __init__(self, from_logits: bool = False, name=None):
        super().__init__(
            fun=ops.sparse_categorical_crossentropy, name=name, from_logits=from_logits
        )
