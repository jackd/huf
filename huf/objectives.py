import typing as tp
from functools import partial

import gin

configurable = partial(gin.configurable, module="huf.objectives")


class Objective(tp.NamedTuple):
    base_name: str
    split: str
    mode: str


class Splits:
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"

    @classmethod
    def all(cls):
        return (Splits.TRAIN, Splits.VALIDATION, Splits.TEST)

    @classmethod
    def validate(cls, split: str):
        if split not in cls.all():
            raise ValueError(f"invalid split '{split}' - must be one of {cls.all()}")


class Modes:
    MIN = "min"
    MAX = "max"

    @classmethod
    def all(cls):
        return (Modes.MIN, Modes.MAX)

    @classmethod
    def validate(cls, mode: str):
        if mode not in cls.all():
            raise ValueError(f"invalid mode '{mode}' - must be one of {cls.all()}")


@configurable
def objective(base_name: str = "loss", split: str = Splits.VALIDATION, mode=Modes.MIN):
    Modes.validate(mode)
    Splits.validate(split)
    return Objective(base_name, split=split, mode=mode)
