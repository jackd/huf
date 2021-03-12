import typing as tp

from huf.types import FitResult


class FitInterrupt(Exception):
    def __init__(self, result: tp.Optional[FitResult] = None):
        self.result = result
        super().__init__()
