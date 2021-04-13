import typing as tp
from functools import partial

import gin

configurable = partial(gin.configurable, module="huf.functional")


@configurable
def chain(funs: tp.Iterable[tp.Callable]):
    def f(arg):
        for fun in funs:
            arg = fun(arg)
        return arg

    return f
