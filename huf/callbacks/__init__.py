from functools import partial

import gin

from .core import Callback
from .early_stopping import EarlyStopping
from .history import History
from .logger import EpochProgbarLogger, EpochVerboseLogger, ProgbarLogger, VerboseLogger
from .terminate_on_nan import TerminateOnNaN

configurable = partial(gin.configurable, module="huf.callbacks")

Callback = configurable(Callback)
ProgbarLogger = configurable(ProgbarLogger)
EpochProgbarLogger = configurable(EpochProgbarLogger)
VerboseLogger = configurable(VerboseLogger)
EpochVerboseLogger = configurable(EpochVerboseLogger)
History = configurable(History)
EarlyStopping = configurable(EarlyStopping)
TerminateOnNaN = configurable(TerminateOnNaN)

__all__ = [
    "Callback",
    "ProgbarLogger",
    "EpochProgbarLogger",
    "VerboseLogger",
    "EpochVerboseLogger",
    "History",
    "EarlyStopping",
    "TerminateOnNaN",
]
