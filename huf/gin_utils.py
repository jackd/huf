from functools import partial

import gin

configurable = partial(gin.configurable, module="huf.gin_utils")


@configurable
def divide(a, b):
    return a / b


@configurable
def int_divide(a, b):
    return a // b


@configurable
def add(a, b):
    return a + b


@configurable
def subtract(a, b):
    return a - b


@configurable
def neg(a):
    return -a


@configurable
def mul(a, b):
    return a * b


@configurable
def pow(a, b):  # pylint: disable=redefined-builtin
    return a ** b


class _NoDefault:
    pass


NO_DEFAULT = _NoDefault()


def get_macro(scope: str, default=NO_DEFAULT):
    with gin.config.config_scope(scope):
        try:
            return gin.config.macro()  # pylint: disable=no-value-for-parameter
        except TypeError as ex:
            if default is NO_DEFAULT:
                raise ValueError(f"No macro '{scope}' registered") from ex
            return default
