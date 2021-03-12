from functools import partial

import gin

from .core import Callback
from .early_stopping import EarlyStopping
from .history import History
from .prog import EpochProgbarLogger, ProgbarLogger

configurable = partial(gin.configurable, module="huf.callbacks")

Callback = configurable(Callback)
ProgbarLogger = configurable(ProgbarLogger)
EpochProgbarLogger = configurable(EpochProgbarLogger)
History = configurable(History)
EarlyStopping = configurable(EarlyStopping)

__all__ = [
    "Callback",
    "ProgbarLogger",
    "EpochProgbarLogger",
    "History",
    "EarlyStopping",
]
