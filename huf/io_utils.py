import io
import json
import pickle
import typing as tp
from functools import partial
from pathlib import Path

import gin

configurable = partial(gin.configurable, module="huf.io_utils")


@configurable
def load(file_or_path: tp.Union[str, Path, io.IOBase]):
    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, "rb") as fp:
            return pickle.load(fp)
    return pickle.load(file_or_path)


@configurable
def load_json_lines(file_or_path: tp.Union[str, Path, io.IOBase]) -> tp.List:
    if isinstance(file_or_path, (str, Path)):
        file_or_path = open(file_or_path, "rb")

    return [json.loads(line) for line in file_or_path]


def save(obj, file_or_path: tp.Union[str, Path, io.IOBase]):
    if isinstance(file_or_path, (str, Path)):
        with open(file_or_path, "wb") as fp:
            pickle.dump(obj, fp)
    else:
        pickle.dump(obj, file_or_path)
