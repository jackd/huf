import inspect
import typing as tp
from functools import partial

import gin
import jax
import jax.numpy as jnp
import optax

import haiku as hk
from huf import avals, data, module_ops
from huf.callbacks.core import Callback
from huf.errors import FitInterrupt
from huf.types import (
    Example,
    FitResult,
    Inputs,
    Labels,
    MetricFactory,
    ModelSpec,
    ModelState,
    Params,
    Preds,
    PRNGKey,
    SampleWeight,
    State,
)

configurable = partial(gin.configurable, module="huf.models")


def _pack_example(
    inputs: Inputs,
    labels: tp.Optional[Labels] = None,
    sample_weight: tp.Optional[SampleWeight] = None,
) -> Example:
    return Example(inputs, labels, sample_weight)


def as_example(example_data) -> Example:
    if not isinstance(example_data, tuple):
        example_data = (example_data,)
    return _pack_example(*example_data)


def pack_metrics(*args, **kwargs):
    """
    Pack function args and kwargs into a single function.

    Args:
        *args, **kwargs: functions with signature
            (labels, preds, sample_weight) -> float

    Returns:
        function with signature
            `(labels, preds, sample_weight, loss=None) -> Mapping[str, float]`
        where keys are given by kwarg keys or arg `__name__`s and possibly 'loss'.
    """
    names = [arg.__name__ for arg in args]
    funs = list(args)
    for k, v in kwargs.items():
        names.append(k)
        funs.append(v)

    assert "loss" not in names

    def wrapped(labels, preds, sample_weight, loss=None):
        out = {
            name: fun()(labels, preds, sample_weight) for name, fun in zip(names, funs)
        }
        if loss is not None:
            assert loss.ndim == 0
            out["loss"] = module_ops.Mean()(loss)
        return out

    return wrapped


@configurable
def with_is_training(
    f: tp.Union[hk.TransformedWithState, tp.Callable]
) -> hk.TransformedWithState:
    """Adds `is_training` (default False) kwarg. The value is not used."""
    if callable(f):
        f = hk.transform_with_state(f)

    def init_fn(
        rng: tp.Optional[tp.Union[PRNGKey, int]], inputs: Inputs, is_training: bool,
    ) -> tp.Tuple[hk.Params, hk.State]:
        del is_training
        return f.init(rng, inputs)

    def apply_fn(params, inputs: Inputs, is_training: bool):
        del is_training
        return f.apply(params, inputs)

    # hk.tie_in_original_fn(f, init_fn, apply_fn)
    return hk.TransformedWithState(init_fn, apply_fn)


@configurable
class Model:
    def __init__(
        self,
        net_transform: tp.Union[
            tp.Callable[[Inputs, bool], Preds],  # (inputs, is_training) -> preds
            tp.Callable[[Inputs], Preds],
            hk.TransformedWithState,
        ],
        loss: tp.Callable[[Labels, Preds, SampleWeight], jnp.ndarray],
        optimizer: optax.GradientTransformation,
        metrics: tp.Optional[
            tp.Union[
                tp.Mapping[str, MetricFactory],
                tp.Iterable[MetricFactory],
                MetricFactory,
            ]
        ] = None,
    ):
        if callable(net_transform):
            net_transform = hk.transform_with_state(net_transform)
        assert isinstance(net_transform, hk.TransformedWithState)
        net_args = inspect.signature(net_transform.apply._original_fn).parameters
        if len(net_args) == 1:
            net_transform = with_is_training(net_transform)
        elif len(net_args) != 2 and any(
            arg.default is None for arg in tuple(net_args.values())[2:]
        ):
            raise ValueError(
                "net_tranform should take (inputs,) or (inputs, is_training), got "
                f"{net_args}"
            )
        del net_args
        self.net_transform = net_transform
        self.loss = loss
        if hasattr(metrics, "items"):
            metrics = pack_metrics(**metrics)
        elif hasattr(metrics, "__iter__"):
            metrics = pack_metrics(*metrics)
        elif metrics is None:
            metrics = pack_metrics()
        else:
            assert callable(metrics)
        metrics = hk.transform_with_state(metrics)
        self._optimizer = optimizer
        self._metrics = metrics
        self._train_step = None
        self._test_step = None
        self._update_metrics = None
        self._model_spec = None
        self._init_metrics_state = None

        self._train_step = jax.jit(self.train_step)
        self._test_step = jax.jit(self.test_step)
        self._update_metrics = jax.jit(self.update_metrics)

    def train_step(
        self,
        params: Params,
        net_state: State,
        rng: PRNGKey,
        opt_state: State,
        metrics_state: State,
        inputs: Inputs,
        labels: Labels = None,
        sample_weight: SampleWeight = None,
    ):
        def loss_fun(params, state, rng, inputs, labels, sample_weight):
            is_training = True
            preds, state = self.net_transform.apply(
                params, state, rng, inputs, is_training
            )
            loss = self.loss(labels, preds, sample_weight)
            return loss, (state, preds)

        (loss, (net_state, preds)), grad = jax.value_and_grad(loss_fun, has_aux=True)(
            params, net_state, rng, inputs, labels, sample_weight
        )
        updates, opt_state = self._optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics, metrics_state = self.update_metrics(
            metrics_state, preds, labels, sample_weight, loss=loss
        )
        return params, net_state, opt_state, metrics_state, preds, loss, metrics

    def test_step(
        self, params, net_state, metrics_state, inputs, labels=None, sample_weight=None
    ):
        rng = None
        is_training = False
        preds, _ = self.net_transform.apply(params, net_state, rng, inputs, is_training)
        loss = self.loss(labels, preds, sample_weight)
        metrics, metrics_state = self.update_metrics(
            metrics_state, preds, labels, sample_weight, loss,
        )
        return metrics_state, preds, loss, metrics

    def update_metrics(
        self, metrics_state, preds, labels=None, sample_weight=None, loss=None
    ):
        rng = None
        metrics, metrics_state = self._metrics.apply(
            {}, metrics_state, rng, labels, preds, sample_weight, loss=loss
        )
        return metrics, metrics_state

    @property
    def compiled_train_step(self) -> tp.Callable:
        return self._train_step

    @property
    def compiled_test_step(self) -> tp.Callable:
        return self._test_step

    @property
    def compiled_update_metrics(self) -> tp.Callable:
        return self._update_metrics

    @property
    def init_metrics_state(self) -> State:
        return self._init_metrics_state

    @property
    def spec(self) -> ModelSpec:
        return self._model_spec

    def model_summary(self) -> str:
        if self._model_spec is None:
            raise NotImplementedError("Cannot print model summary before compile.")

        return hk.experimental.tabulate(
            self.net_transform,
            columns=(
                "module",
                # "config",
                "owned_params",
                "input",
                "output",
                "params_size",
                "params_bytes",
            ),
        )(
            jax.tree_util.tree_map(avals.zeros_like, self._model_spec.inputs),
            is_training=True,
        )

    def compile(
        self,
        inputs: Inputs,
        labels: tp.Optional[Labels] = None,
        sample_weight: tp.Optional[SampleWeight] = None,
    ):
        inputs_spec, labels_spec, sample_weight_spec = jax.tree_util.tree_map(
            lambda x: None if x is None else x.aval, (inputs, labels, sample_weight)
        )
        if self._model_spec is not None and avals.is_compatible(
            (inputs_spec, labels_spec, sample_weight_spec),
            (
                self._model_spec.inputs,
                self._model_spec.labels,
                self._model_spec.sample_weight,
            ),
        ):
            return

        def f(inputs, labels, sample_weight):
            is_training = True
            rng = jax.random.PRNGKey(0)
            params, net_state = self.net_transform.init(rng, inputs, is_training)
            opt_state = self._optimizer.init(params)
            preds, net_state = self.net_transform.apply(
                params, net_state, rng, inputs, is_training
            )
            metrics, metrics_state = self._metrics.init(
                None, labels, preds, sample_weight
            )
            return preds, metrics, params, net_state, opt_state, metrics_state

        self._model_spec = ModelSpec(
            inputs_spec,
            labels_spec,
            sample_weight_spec,
            *avals.abstract_eval(f, inputs_spec, labels_spec, sample_weight_spec),
        )
        metric_params, self._init_metrics_state = self._metrics.init(
            None,
            labels,
            jax.tree_util.tree_map(avals.zeros_like, self._model_spec.preds),
            sample_weight,
        )
        if metric_params:
            raise ValueError(
                "metrics should not introduce parameters. Check constructor arg."
            )

    def init(self, rng: PRNGKey, inputs: Inputs) -> ModelState:
        is_training = True
        params, net_state = self.net_transform.init(rng, inputs, is_training)
        opt_state = self._optimizer.init(params)
        return ModelState(params, net_state, opt_state)

    def evaluate(
        self,
        params: Params,
        net_state: State,
        validation_data: tp.Iterable,
        callbacks=(),
    ):
        validation_data = data.as_dataset(validation_data)
        dummy_example = as_example(
            jax.tree_util.tree_map(avals.zeros_like, validation_data.element_spec)
        )
        self.compile(*dummy_example)  # pylint: disable=not-an-iterable

        for callback in callbacks:
            callback.on_test_begin()
        metrics = self._evaluate(params, net_state, validation_data, callbacks)
        for callback in callbacks:
            callback.on_test_end(metrics)
        return metrics

    def _evaluate(
        self,
        params: Params,
        net_state: State,
        validation_data: data.Dataset,
        callbacks: tp.Iterable[Callback],
    ):
        metrics_state = self._init_metrics_state
        metrics = {}

        for step, example in enumerate(validation_data):
            example = as_example(example)
            for callback in callbacks:
                callback.on_test_step_begin(step)
            metrics_state, preds, loss, metrics = self._test_step(
                params,
                net_state,
                metrics_state,
                example.inputs,
                example.labels,
                example.sample_weight,
            )
            del preds, loss
            for callback in callbacks:
                callback.on_test_step_end(step, metrics)

        return metrics

    def fit(
        self,
        rng: PRNGKey,
        train_data: tp.Iterable,
        epochs: int = 1,
        validation_data: tp.Optional[tp.Iterable] = None,
        initial_state: tp.Optional[ModelState] = None,
        initial_epoch: int = 0,
        callbacks: tp.Iterable[Callback] = (),
        verbose: bool = True,
    ):
        train_data = data.as_dataset(train_data)
        try:
            steps_per_epoch = len(train_data)
        except TypeError:
            steps_per_epoch = None
        if validation_data is not None:
            validation_data = data.as_dataset(validation_data)
            assert train_data.element_spec == validation_data.element_spec, (
                train_data.element_spec,
                validation_data.element_spec,
            )
        if not hasattr(callbacks, "__iter__"):
            callbacks = (callbacks,)

        dummy_example = as_example(
            jax.tree_util.tree_map(avals.zeros_like, train_data.element_spec)
        )
        self.compile(*dummy_example)  # pylint: disable=not-an-iterable
        if verbose:
            print(self.model_summary())
        if initial_state is None:
            rng, rng_ = jax.random.split(rng)
            initial_state = self.init(rng_, dummy_example.inputs)

        model_state = initial_state
        for callback in callbacks:
            callback.on_train_begin(epochs, steps_per_epoch, model_state)

        validation_metrics = None
        try:
            for epoch in range(initial_epoch, epochs):
                metrics_state = self._init_metrics_state
                for callback in callbacks:
                    callback.on_epoch_begin(epoch, rng, model_state)

                for step, example in enumerate(train_data):
                    example = as_example(example)
                    for callback in callbacks:
                        callback.on_train_step_begin(step)
                    rng, rng_ = jax.random.split(rng)
                    (
                        params,
                        net_state,
                        opt_state,
                        metrics_state,
                        preds,
                        loss,
                        train_metrics,
                    ) = self.compiled_train_step(  # pylint: disable=not-callable
                        model_state.params,
                        model_state.net_state,
                        rng_,
                        model_state.opt_state,
                        metrics_state,
                        *example,  # pylint: disable=not-an-iterable
                    )
                    del preds, loss
                    for callback in callbacks:
                        callback.on_train_step_end(step, train_metrics)
                    model_state = ModelState(params, net_state, opt_state)
                if validation_data is not None:
                    validation_metrics = self._evaluate(
                        model_state.params,
                        model_state.net_state,
                        validation_data,
                        callbacks,
                    )
                result = FitResult(
                    epoch + 1, rng, model_state, train_metrics, validation_metrics
                )
                for callback in callbacks:
                    callback.on_epoch_end(
                        epoch, rng, model_state, train_metrics, validation_metrics
                    )
        except FitInterrupt as interrupt:
            if interrupt.result is not None:
                result = interrupt.result

        for callback in callbacks:
            callback.on_train_end(*result)

        return result


@configurable
def fit(
    model: Model,
    rng: PRNGKey,
    train_data: tp.Iterable,
    epochs: int = 1,
    validation_data: tp.Optional[tp.Iterable] = None,
    initial_state: tp.Optional[ModelState] = None,
    initial_epoch: int = 0,
    callbacks: tp.Iterable[Callback] = (),
):
    """Configurable version of `Model.fit`."""
    return model.fit(
        rng=rng,
        train_data=train_data,
        epochs=epochs,
        validation_data=validation_data,
        initial_state=initial_state,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )
