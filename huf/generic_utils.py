import typing as tp
from collections import defaultdict

E = tp.TypeVar("E")


def group_by(objs: tp.Iterable[E], key: tp.Callable[[E], tp.Any]):
    out = defaultdict(list)
    for obj in objs:
        out[key(obj)].append(obj)
    return out
