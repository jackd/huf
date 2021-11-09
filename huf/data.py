import abc
import collections
import itertools
import typing as tp

import haiku as hk
import jax
import jax.numpy as jnp

from huf import avals
from huf.avals import abstract_eval
from huf.types import AbstractTree, PRNGKey


class Dataset(collections.abc.Iterable):
    """Minimal class for managing iterables of examples."""

    @abc.abstractproperty
    def element_spec(self) -> AbstractTree:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        """May raise a TypeError."""
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError("Abstract method")

    @property
    def state(self):
        return None

    # factory constructors

    @staticmethod
    def from_iterable(
        iterable: tp.Iterable, spec: tp.Optional[AbstractTree] = None
    ) -> "Dataset":
        return IterableDataset(iterable, spec)

    @staticmethod
    def from_generator(
        iterable: tp.Iterable,
        spec: tp.Optional[AbstractTree] = None,
        length: tp.Optional[int] = None,
    ):
        return GeneratorDataset(iterable, spec, length)

    @staticmethod
    def from_sequence(sequence: tp.Sequence):
        return SequenceDataset(sequence)

    @staticmethod
    def from_tf(dataset) -> "Dataset":
        return TfDatasetWrapper(dataset)

    def to_tf(self):
        """
        Convert into a `tf.data.Dataset`.

        In conjunction with `TfDatasetWrapper`, this allows full utilization of the
        `tf.data` API.

        Args:
            dataset: `huf.data.Dataset`. Only arbitrarily nested lists/tuples/dicts of
                regular `jax.numpy.ndarray`s are currently supported.
            currently supported.

        Returns:
            Generator-based `tf.data.Dataset` with `assert_cardinality` if known.
        """
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        def gen():
            return (jax.tree_map(tf.convert_to_tensor, el) for el in self)

        spec = jax.tree_map(
            lambda x: tf.TensorSpec(x.shape, x.dtype), self.element_spec
        )
        tf_dataset = tf.data.Dataset.from_generator(gen, output_signature=spec)
        size = len(self)
        if size is not None:
            tf_dataset = tf_dataset.apply(tf.data.experimental.assert_cardinality(size))
        return tf_dataset

    # transformation methods

    def repeat(self, repeats: tp.Optional[int] = None) -> "Dataset":
        return RepeatedDataset(self, repeats)

    def take(self, n: int) -> "Dataset":
        return TakenDataset(self, n)

    def batch(self, batch_size: int, drop_remainder: bool = False) -> "Dataset":
        return BatchedDataset(self, batch_size, drop_remainder)

    def map(self, fn: tp.Callable) -> "Dataset":
        return MappedDataset(self, fn)


def save(dataset: Dataset, path: str, compression=None, shard_func=None):
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    return tf.data.experimental.save(
        dataset.to_tf(), path, compression=compression, shard_func=shard_func
    )


def load(path: str, spec, compression=None, reader_func=None):
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    spec = jax.tree_map(lambda x: tf.TensorSpec(x.shape, x.dtype), spec)
    return TfDatasetWrapper(
        tf.data.experimental.load(
            path, spec, compression=compression, reader_func=reader_func
        )
    )


class MappedDataset(Dataset):
    def __init__(self, dataset: Dataset, map_fn: tp.Callable):
        self._dataset = dataset
        self._map_fn = map_fn
        self._element_spec = None

    def __len__(self):
        return len(self._dataset)

    @property
    def element_spec(self):
        if self._element_spec is None:
            self._element_spec = abstract_eval(self._map_fn, self._dataset.element_spec)
        return self._element_spec

    def __iter__(self):
        return iter((self._map_fn(el) for el in self._dataset))


class BatchedDataset(Dataset):
    def __init__(self, dataset: Dataset, batch_size: int, drop_remainder: bool):
        self._dataset = dataset
        self._batch_size = batch_size
        self._drop_remainder = drop_remainder
        self._spec = None
        self._add_weights = len(self._dataset.element_spec) == 2

    @property
    def element_spec(self):
        if self._spec is None:
            spec = jax.tree_map(
                lambda x: jax.core.ShapedArray((self._batch_size, *x.shape), x.dtype),
                self._dataset.element_spec,
            )
            if self._add_weights:
                spec = (*spec, jax.core.ShapedArray((self._batch_size,), jnp.float32))
            self._spec = spec
        return self._spec

    def __iter__(self):
        one = jnp.ones((), dtype=jnp.float32)
        zeros = jax.tree_map(jnp.zeros_like, self.element_spec)

        def gen():
            it = iter(self._dataset)

            def as_element(batch):
                return jax.tree_map(lambda *b: jnp.asarray(b), *batch)

            try:
                while True:
                    batch = []
                    for _ in range(self._batch_size):
                        el = next(it)
                        if self._add_weights:
                            el = (*el, one)
                        batch.append(el)
                    yield as_element(batch)

            except StopIteration:
                if len(batch) > 0 and not self._drop_remainder:
                    yield jax.tree_map(
                        lambda b, z: jnp.concatenate(
                            (b, z[: self._batch_size - len(batch)]), axis=0
                        ),
                        as_element(batch),
                        zeros,
                    )

        return iter(gen())

    def __len__(self):
        n = len(self._dataset)
        if n is None:
            return None
        return n // self._batch_size + bool(
            n % self._batch_size > 0 and not self._drop_remainder
        )

    @property
    def state(self):
        return self._dataset.state

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def drop_remainder(self) -> bool:
        return self._drop_remainder


class IterableDataset(Dataset):
    """Thin wrapper around an `Iterable`."""

    def __init__(self, iterable: tp.Iterable, spec: tp.Optional[AbstractTree] = None):
        self._iterable = iterable
        if spec is None:
            try:
                first = next(iter(self._iterable))
            except StopIteration as e:
                raise ValueError("Iterable must have at least one element") from e
            spec = avals.abstract_tree(first)
        self._spec = spec

    def __iter__(self):
        return iter(self._iterable)

    def __len__(self) -> int:
        return len(self._iterable)

    @property
    def element_spec(self) -> AbstractTree:
        return self._spec


class GeneratorDataset(Dataset):
    def __init__(
        self,
        generator: tp.Callable[[], tp.Iterable],
        spec: tp.Optional[AbstractTree] = None,
        length: tp.Optional[int] = None,
    ):
        self._generator = generator
        self._length = length
        if spec is None:
            try:
                first = next(iter(generator()))
            except StopIteration as e:
                raise ValueError("Generator must yield at least one element") from e
            spec = avals.abstract_tree(first)
        self._spec = spec

    def __iter__(self):
        return iter(self._generator())

    def __len__(self) -> int:
        if self._length is None:
            raise TypeError("`length` must be provided in constructor")
        return self._length

    @property
    def element_spec(self) -> AbstractTree:
        return self._spec


class TfDatasetWrapper(Dataset):
    """Thin wrapper around `tf.data.Dataset`."""

    def __init__(self, tf_dataset):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        assert isinstance(tf_dataset, tf.data.Dataset)
        self._tf_dataset = tf_dataset
        self._spec = None

    def __iter__(self):
        return self._tf_dataset.as_numpy_iterator()

    def to_tf(self):
        return self._tf_dataset

    @property
    def element_spec(self) -> AbstractTree:
        from huf.tf import spec_to_aval  # pylint: disable=import-outside-toplevel

        if self._spec is None:
            self._spec = jax.tree_util.tree_map(
                spec_to_aval, self._tf_dataset.element_spec
            )
        return self._spec

    def __len__(self) -> int:
        return len(self._tf_dataset)


class RepeatedDataset(Dataset):
    def __init__(self, base: Dataset, repeats: tp.Optional[int] = None):
        self._base = base
        self._repeats = repeats

    @property
    def element_spec(self) -> AbstractTree:
        return self._base.element_spec

    def __len__(self) -> tp.Optional[int]:
        base_len = len(self._base)
        if self._repeats is not None and base_len:
            return base_len * self._repeats
        return None

    def __iter__(self):
        if self._repeats is None:
            return iter(itertools.cycle(self._base))
        return iter(
            itertools.chain.from_iterable(itertools.repeat(self._base, self._repeats))
        )


class TakenDataset(Dataset):
    def __init__(self, base: Dataset, n: int):
        self._base = base
        self._n = n

    @property
    def element_spec(self):
        return self._base.element_spec

    def __len__(self) -> tp.Optional[int]:
        n = len(self._base)
        return None if n is None else min(n, self._n)

    def __iter__(self):
        return iter(itertools.islice(self._base, self._n))


def as_dataset(data, spec=None) -> Dataset:
    if isinstance(data, Dataset):
        if spec is not None:
            assert spec == data.element_spec, (spec, data.element_spec)
        return data
    if hasattr(data, "element_spec"):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        assert isinstance(data, tf.data.Dataset), type(data)
        return TfDatasetWrapper(data)
    if hasattr(data, "__iter__"):
        return IterableDataset(data, spec)
    raise TypeError(f"data must be a Dataset, tf.data.Dataset or iterable, got {data}")


class SequenceDataset(Dataset):
    def __init__(self, data):
        lengths = jax.tree_flatten(jax.tree_map(len, data))[0]
        self._size = lengths[0]
        assert all((l == self._size for l in lengths[1:]))
        self._data = data
        self._spec = None

    @property
    def element_spec(self) -> AbstractTree:
        if self._spec is None:
            self._spec = jax.tree_map(
                lambda x: jax.core.ShapedArray(x.shape, x.dtype), self[0]
            )
        return self._spec

    def __len__(self):
        return self._size

    def __iter__(self):
        def gen():
            for i in range(self._size):
                yield self[i]

        return iter(gen())

    def __getitem__(self, i):
        return jax.tree_map(lambda x: x[i], self._data)

    def shuffle(self, key: tp.Union[int, PRNGKey]):
        return ShuffledSequenceDataset(self, key)


class ShuffledSequenceDataset(Dataset):
    def __init__(
        self,
        data,
        key: tp.Union[int, PRNGKey],
    ):
        if not isinstance(data, SequenceDataset):
            data = SequenceDataset(data)
        if not isinstance(key, PRNGKey):
            key = jax.random.PRNGKey(key)
        self._dataset = data
        self._rng = hk.PRNGSequence(key)

    @property
    def element_spec(self):
        return self._dataset.element_spec

    def __iter__(self):
        order = jax.random.permutation(next(self._rng), len(self._dataset))
        return iter((self._dataset[i] for i in order))

    def __len__(self):
        return len(self._dataset)

    @property
    def state(self):
        return self._rng.internal_state
