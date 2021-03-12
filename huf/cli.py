import typing as tp

import gin
from absl import flags

import huf.config  # pylint:disable=unused-import

flags.DEFINE_multi_string("gin_file", default=[], help="gin files to include")
flags.DEFINE_multi_string("bindings", default=[], help="Additional gin bindings")
flags.DEFINE_boolean(
    "finalize_config", default=True, help="Finalize gin config in main"
)
flags.DEFINE_multi_string(
    "config_path", default=[], help="Additional paths to search for .gin files"
)


@gin.configurable(module="huf.cli")
def main(fun: tp.Optional[tp.Callable[[], tp.Any]] = None):
    if fun is None:
        raise ValueError("`main.fun` not configured.")
    return fun()


def app_main(args):
    FLAGS = flags.FLAGS
    files = FLAGS.gin_file + args[1:]
    bindings = FLAGS.bindings
    for path in FLAGS.config_path:
        gin.config.add_config_file_search_path(path)
    gin.parse_config_files_and_bindings(
        files, bindings, finalize_config=FLAGS.finalize_config
    )
    print(gin.config.config_str())
    main()