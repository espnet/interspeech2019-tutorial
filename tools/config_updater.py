#!/usr/bin/env python3

from pathlib import Path
from traitlets.config.manager import BaseJSONConfigManager

path = Path.home() / ".jupyter" / "nbconfig"
cm = BaseJSONConfigManager(config_dir=str(path))
cm.update(
    "rise",
    {
        "scroll": True,
        "transition": "none",
        "enable_chalkboard": True,
     }
)
