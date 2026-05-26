from functools import lru_cache
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)
