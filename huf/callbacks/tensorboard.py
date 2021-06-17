import os

import tensorboardX as tb

from huf.callbacks.core import Callback
from huf.types import FitResult


class TensorBoard(Callback):
    def __init__(self, logdir: str):
        logdir = os.path.expanduser(os.path.expandvars(logdir))
        if os.path.isdir(logdir):
            raise ValueError(f"Directory already exists at {logdir}")
        self.logdir = logdir
        self.train_writer = tb.SummaryWriter(os.path.join(logdir, "train"))
        self.validation_writer = tb.SummaryWriter(os.path.join(logdir, "val"))

    def on_epoch_end(self, result: FitResult):
        train_metrics = result.train_metrics
        validation_metrics = result.validation_metrics
        epochs = result.state.epochs
        for k, v in train_metrics.items():
            self.train_writer.add_scalar(k, v, epochs)
        for k, v in validation_metrics.items():
            self.validation_writer.add_scalar(k, v, epochs)

    def on_train_end(self, result: FitResult):
        for writer in (self.train_writer, self.validation_writer):
            writer.flush()
