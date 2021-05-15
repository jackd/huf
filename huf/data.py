import abc
import collections
import itertools
import typing as tp

import jax
from huf import avals
from huf.types import AbstractTree


class Dataset(collections.abc.Iterable):
    """Minimal class for managing iterables of examples."""

    @abc.abstractproperty
    def element_spec(self) -> AbstractTree:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        """May raise a TypeError."""
        raise NotImplementedError

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
    def from_tf(dataset) -> "Dataset":
        return TfDatasetWrapper(dataset)

    def repeat(self, repeats: tp.Optional[int] = None) -> "Dataset":
        return RepeatedDataset(self, repeats)

    def take(self, n: int) -> "Dataset":
        return TakenDataset(self, n)


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


def as_dataset(data) -> Dataset:
    if isinstance(data, Dataset):
        return data
    if hasattr(data, "element_spec"):
        return TfDatasetWrapper(data)
    if hasattr(data, "__iter__"):
        return IterableDataset(data)
    raise TypeError(f"data must be a Dataset, tf.data.Dataset or iterable, got {data}")
