import typing as tp
from functools import partial

import gin

configurable = partial(gin.configurable, module="huf.experiments")


def _call_or_skip(fun, *args, **kwargs):
    if fun is None:
        return
    fun(*args, **kwargs)


class ExperimentCallback:
    def on_start(self):
        pass

    def on_done(self, result):
        pass

    def on_interrupt(self):
        pass

    def on_exception(self, exception: Exception):
        pass


def maybe_chain(funs: tp.Optional[tp.Union[tp.Callable, tp.Iterable[tp.Callable]]]):
    if funs is None:
        return None

    if hasattr(funs, "__iter__"):

        def f(arg=None):
            for fun in funs:
                if arg is None:
                    arg = fun()
                else:
                    arg = fun(arg)
            return arg

        return f

    return funs


@configurable
class LambdaCallback(ExperimentCallback):
    def __init__(
        self,
        on_start: tp.Optional[tp.Callable[[], tp.Any]] = None,
        on_done: tp.Optional[tp.Callable[[tp.Any], tp.Any]] = None,
        on_interrupt: tp.Optional[tp.Callable[[], tp.Any]] = None,
        on_exception: tp.Optional[tp.Callable[[Exception], tp.Any]] = None,
    ):
        self._on_start = maybe_chain(on_start)
        self._on_done = maybe_chain(on_done)
        self._on_interrupt = maybe_chain(on_interrupt)
        self._on_exception = maybe_chain(on_exception)
        super().__init__()

    def on_start(self):
        _call_or_skip(self._on_start)

    def on_done(self, result):
        _call_or_skip(self._on_done, result)

    def on_interrupt(self):
        _call_or_skip(self._on_interrupt)

    def on_exception(self, exception: Exception):
        _call_or_skip(self._on_exception, exception)


@configurable
class ConfigLogger(ExperimentCallback):
    def __init__(self, print_fun: tp.Callable[[tp.Any], None] = print):
        self.print = print_fun

    def on_start(self):
        self.print(
            "\n".join(("# Starting experiment with config:", gin.config.config_str()))
        )


@configurable
class OperativeConfigLogger(ExperimentCallback):
    def __init__(self, print_fun: tp.Callable[[tp.Any], None] = print):
        self.print = print_fun

    def on_done(self, result):
        del result
        self.print(
            "\n".join(
                (
                    "# Finished experiment with operative_config:",
                    gin.operative_config_str(),
                )
            )
        )


@configurable
class Logger(ExperimentCallback):
    def __init__(
        self, print_fun=print,
    ):
        self.print = print_fun

    def on_start(self):
        self.print("Starting experiment")

    def on_done(self, result):
        self.print("Done. result:")
        self.print(result)

    def on_interrupt(self):
        self.print("Interrupted")

    def on_exception(self, exception: Exception):
        self.print("Unhandled exception raised")
        self.print(exception)


@configurable
def run(
    fun: tp.Callable[[], tp.Any] = gin.REQUIRED,
    callbacks: tp.Iterable[ExperimentCallback] = (Logger(),),
):
    for callback in callbacks:
        callback.on_start()
    try:
        result = fun()
        for callback in callbacks:
            callback.on_done(result)
        return result
    except KeyboardInterrupt:
        for callback in callbacks:
            callback.on_interrupt()
        raise
    except Exception as exception:
        for callback in callbacks:
            callback.on_exception(exception)
        raise
