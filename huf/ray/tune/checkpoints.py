import os
import pickle

from ray import tune

from huf.types import FitResult


def _save_path(folder: str):
    if os.path.isfile(folder):
        return folder
    return os.path.join(folder, "data.pkl")


def save_checkpoint(data: FitResult):
    step = data.state.epochs
    with tune.checkpoint_dir(step=step) as checkpoint_dir:
        with open(_save_path(checkpoint_dir), "wb") as fp:
            pickle.dump(data, fp)
    return checkpoint_dir


def load_checkpoint(checkpoint_dir: str) -> FitResult:
    with open(_save_path(checkpoint_dir), "rb") as fp:
        return pickle.load(fp)


def get_step(checkpoint_dir: str):
    if checkpoint_dir.endswith("/"):
        checkpoint_dir = checkpoint_dir[:-1]
    return int(checkpoint_dir.split("_")[-1])
