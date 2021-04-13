import typing as tp

from huf.types import Modes, Objective, Splits


def objective(
    key: str = "loss", split: str = Splits.VALIDATION, mode: tp.Optional[str] = None
):
    if mode is None:
        if "acc" in key:
            mode = Modes.MAX
        else:
            mode = Modes.MIN
    else:
        Modes.validate(mode)
    Splits.validate(split)
    return Objective(key=key, split=split, mode=mode)


DEFAULT_OBJECTIVE = objective()  # min-val_loss
