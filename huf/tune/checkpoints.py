import os
import pickle
import typing as tp

from ray import tune

from huf.types import ModelState, PRNGKey


class CheckpointData(tp.NamedTuple):
    epoch: int
    rng: PRNGKey
    state: ModelState


def _save_path(folder: str):
    return os.path.join(folder, "data.pkl")


def save_checkpoint(data: CheckpointData):
    step = data.epoch
    with tune.checkpoint_dir(step=step) as checkpoint_dir:
        with open(_save_path(checkpoint_dir), "wb") as fp:
            pickle.dump(data, fp)
    return checkpoint_dir


def load_checkpoint(checkpoint_dir: str):
    with open(_save_path(checkpoint_dir), "rb") as fp:
        return pickle.load(fp)


def get_step(checkpoint_dir: str):
    if checkpoint_dir.endswith("/"):
        checkpoint_dir = checkpoint_dir[:-1]
    return int(checkpoint_dir.split("_")[-1])
