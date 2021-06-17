import typing as tp

import gin
from absl import flags
from jax.config import config as jax_config

from huf import experiments

flags.DEFINE_multi_string("gin_file", default=[], help="gin files to include")
flags.DEFINE_multi_string("bindings", default=[], help="Additional gin bindings")
flags.DEFINE_boolean(
    "finalize_config", default=True, help="Finalize gin config in main"
)
flags.DEFINE_multi_string(
    "config_path", default=[], help="Additional paths to search for .gin files"
)
flags.DEFINE_bool("jax_enable_x64", default=False, help="enable float64")


@gin.configurable(module="huf.cli")
def main(
    fun: tp.Callable[[], tp.Any] = gin.REQUIRED,
    callbacks: tp.Iterable[experiments.ExperimentCallback] = (
        experiments.ConfigLogger(),
    ),
):
    return experiments.run(fun=fun, callbacks=callbacks)


def app_main(args):
    FLAGS = flags.FLAGS
    if FLAGS.jax_enable_x64:
        jax_config.update("jax_enable_x64", True)
    files = FLAGS.gin_file + args[1:]
    bindings = FLAGS.bindings
    for path in FLAGS.config_path:
        gin.config.add_config_file_search_path(path)
    gin.parse_config_files_and_bindings(
        files, bindings, finalize_config=FLAGS.finalize_config
    )
    main()
