from functools import partial

import gin

from huf import ops

configurable = partial(gin.configurable, module="huf.losses")


@configurable
def sparse_categorical_crossentropy(
    labels, preds, sample_weight=None, from_logits: bool = False
):
    return ops.weighted_mean(
        ops.sparse_categorical_crossentropy(labels, preds, from_logits=from_logits),
        sample_weight,
    )
