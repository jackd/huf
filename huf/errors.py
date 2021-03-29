import typing as tp

from huf.types import FitState


class FitInterrupt(Exception):
    def __init__(self, result: tp.Optional[FitState] = None):
        self.result = result
        super().__init__()
