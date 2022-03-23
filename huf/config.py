"""
Can be imported from `gin` to add directory to gin search path.

Example config file:

```gin
import grax.config
include "grax_config/profile.gin"
...
```

"""
import os

import gin

gin.add_config_file_search_path(os.path.dirname(__file__))
