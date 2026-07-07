from functools import lru_cache
from copy import deepcopy

DEFAULT_CONFIG = {
    "semantic_classes": {
        "names": [
            "arch",
            "column",
            "moldings",
            "floor",
            "door_window",
            "wall",
            "stairs",
            "vault",
            "roof",
            "other",
        ],
        "structural": [
            "arch",
            "column",
            "wall",
            "vault",
            "roof",
        ],
        "finishing": [
            "moldings",
            "floor",
            "door_window",
            "stairs",
            "other",
        ],
        "colors": {
            "arch": [0.85, 0.37, 0.01],
            "column": [0.20, 0.63, 0.17],
            "moldings": [0.12, 0.47, 0.71],
            "floor": [0.58, 0.40, 0.74],
            "door_window": [1.00, 0.50, 0.05],
            "wall": [0.65, 0.65, 0.65],
            "stairs": [0.84, 0.15, 0.16],
            "vault": [0.09, 0.75, 0.81],
            "roof": [0.45, 0.24, 0.07],
            "other": [0.50, 0.50, 0.50],
        },
    }
}


@lru_cache(maxsize=1)
def get_config() -> dict:
    return deepcopy(DEFAULT_CONFIG)
