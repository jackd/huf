import typing as tp

import gin
import jax
from ray import tune

from huf import avals, data, gin_utils, models


def _get_or_macro(
    mapping: tp.Mapping[str, tp.Any], key: str, default=gin_utils.NO_DEFAULT
):
    if key in mapping:
        return mapping[key]
    return gin_utils.get_macro(key, default)


# pylint: disable=attribute-defined-outside-init
class Trainable(tune.Trainable):  # pylint: disable=abstract-method
    def setup(  # pylint: disable=arguments-differ
        self, config, base_config: str = "", **kwargs
    ):
        gin.parse_config(
            [base_config, *(f"{k} = {v}" for k, v in config.items()),]
        )
        self.model: models.Model = _get_or_macro(kwargs, "model")
        self.rng = _get_or_macro(kwargs, "rng")
        self.train_data = data.as_dataset(_get_or_macro(kwargs, "train_data"))
        self.epochs = _get_or_macro(kwargs, "epochs", 1)
        self.callbacks = tuple(_get_or_macro(kwargs, "callbacks", ()))
        self.validation_data = _get_or_macro(kwargs, "validation_data", None)
        if self.validation_data is not None:
            self.validation_data = data.as_dataset(self.validation_data)
        self.rng, rng = jax.random.split(self.rng)
        self.state = self.model.init(
            rng, avals.zeros_like(self.train_data.element_spec)
        )
        self.epoch = 0

    def step(self):
        self.rng, rng = jax.random.split(self.rng)
        train_metrics, validation_metrics, self.state = self.model.fit_epoch(
            epoch=self.epoch,
            rng=rng,
            state=self.state,
            train_data=self.train_data,
            callbacks=self.callbacks,
            validation_data=self.validation_data,
        )
        self.epoch += 1


# pylint: enable=attribute-defined-outside-init
