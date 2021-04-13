import typing as tp
from collections import defaultdict

V = tp.TypeVar("V")
K = tp.TypeVar("K")


def group_by(objs: tp.Iterable[V], key: tp.Callable[[V], K]) -> tp.Dict[K, tp.List[V]]:
    out = defaultdict(list)
    for obj in objs:
        out[key(obj)].append(obj)
    return out
