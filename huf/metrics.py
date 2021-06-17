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
) -> tp.Mapping[str, jnp.ndarray]:
    metrics = {}
    for split, mets in (
        (Splits.TRAIN, train_metrics),
        (Splits.VALIDATION, validation_metrics),
        (Splits.TEST, test_metrics),
    ):
        if mets is not None:
            metrics.update(
                {
                    f"{split}_{k}": v.item() if v.size == 1 else v
                    for k, v in mets.items()
                }
            )
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
class UnweightedMean(hk.Module):
    def __init__(self, fun: tp.Callable[..., jnp.ndarray], name=None, **kwargs):
        super().__init__(name=name)
        self._fun = fun
        self._kwargs = kwargs

    def __call__(
        self, labels: Labels, preds: Preds, sample_weight: tp.Optional[SampleWeight]
    ):
        value = self._fun(labels, preds, **self._kwargs)
        return module_ops.mean(value)


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


@configurable
class MultiClassAccuracy(Mean):
    def __init__(self, from_logits: bool = False, name=None):
        super().__init__(
            fun=ops.multi_class_accuracy, name=name, from_logits=from_logits
        )


@configurable
class MicroF1(hk.Module):
    def __init__(self, from_logits: bool = False, name=None):
        super().__init__(name=name)
        self._from_logits = from_logits

    def __call__(
        self, labels: Labels, preds: Preds, sample_weight: tp.Optional[SampleWeight]
    ):
        preds = preds >= (0 if self._from_logits else 0.5)
        correct = preds == labels
        incorrect = jnp.logical_not(correct)
        true_positives = jnp.logical_and(correct, preds)
        false_positives = jnp.logical_and(incorrect, preds)
        false_negatives = jnp.logical_and(incorrect, jnp.logical_not(preds))
        if sample_weight is None:

            def map_fun(x):
                return jnp.count_nonzero(x).astype(jnp.float32)

        else:
            assert sample_weight.dtype == jnp.float32

            def map_fun(x):
                return jnp.sum(x * sample_weight[:, jnp.newaxis])

        def update(value, name):
            value = hk.get_state(name, (), jnp.float32, jnp.zeros) + map_fun(value)
            hk.set_state(name, value)
            return value

        true_positives, false_positives, false_negatives = (
            update(x, name)
            for x, name in (
                (true_positives, "true_positives"),
                (false_positives, "false_positives"),
                (false_negatives, "false_negatives"),
            )
        )

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        return f1
