from absl import app

import huf.config  # pylint:disable=unused-import
from huf import cli

if __name__ == "__main__":
    app.run(cli.app_main)
