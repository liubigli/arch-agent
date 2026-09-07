"""Semantic class registry for architectural point-cloud scenes."""

SEMANTIC_CLASS_REGISTRY = {
    "arch": {
        "id": 0,
        "category": "structural",
        "role": "structural",
        "segmentation_method": "DBSCAN",
        "segmentation_reason": "density-based object extraction",
        "can_support": {"vault", "roof"},
        "can_rest_on": {"column", "wall"},
        "part_of": {
            "wall": "is_attached_to",
            "vault": "is_rib_of",
        },
    },
    "column": {
        "id": 1,
        "category": "structural",
        "role": "structural",
        "segmentation_method": "DBSCAN",
        "segmentation_reason": "density-based object extraction",
        "can_support": {"arch", "vault", "roof"},
        "can_rest_on": {"floor"},
        "part_of": {},
    },
    "moldings": {
        "id": 2,
        "category": "finishing",
        "role": "ornamental",
        "segmentation_method": "DBSCAN",
        "segmentation_reason": "often continuous decorative geometry",
        "can_support": set(),
        "can_rest_on": set(),
        "part_of": {
            "wall": "is_ornament_of",
            "arch": "is_ornament_of",
            "column": "is_ornament_of",
        },
    },
    "floor": {
        "id": 3,
        "category": "finishing",
        "role": "support_surface",
        "segmentation_method": "DBSCAN",
        "segmentation_reason": "continuous surface",
        "can_support": {"column", "wall", "stairs"},
        "can_rest_on": set(),
        "part_of": {},
    },
    "door_window": {
        "id": 4,
        "category": "finishing",
        "role": "opening",
        "segmentation_method": "DBSCAN",
        "segmentation_reason": "density-based object extraction",
        "can_support": set(),
        "can_rest_on": set(),
        "part_of": {
            "wall": "is_opening_in",
        },
    },
    "wall": {
        "id": 5,
        "category": "structural",
        "role": "structural",
        "segmentation_method": "DBSCAN",
        "segmentation_reason": "continuous or irregular geometry",
        "can_support": {"arch", "vault", "roof"},
        "can_rest_on": {"floor"},
        "part_of": {},
    },
    "stairs": {
        "id": 6,
        "category": "finishing",
        "role": "circulation",
        "segmentation_method": "DBSCAN",
        "segmentation_reason": "variable topology",
        "can_support": set(),
        "can_rest_on": {"floor"},
        "part_of": {
            "floor": "is_placed_on",
            "wall": "is_connected_to",
        },
    },
    "vault": {
        "id": 7,
        "category": "structural",
        "role": "structural",
        "segmentation_method": "DBSCAN",
        "segmentation_reason": "irregular/continuous curved geometry",
        "can_support": {"roof"},
        "can_rest_on": {"arch", "column", "wall"},
        "part_of": {},
    },
    "roof": {
        "id": 8,
        "category": "structural",
        "role": "structural",
        "segmentation_method": "DBSCAN",
        "segmentation_reason": "irregular/continuous geometry",
        "can_support": set(),
        "can_rest_on": {"arch", "column", "wall", "vault"},
        "part_of": {},
    },
    "other": {
        "id": 9,
        "category": "finishing",
        "role": "unknown",
        "segmentation_method": "DBSCAN",
        "segmentation_reason": "unknown or mixed topology",
        "can_support": set(),
        "can_rest_on": set(),
        "part_of": {
            "wall": "part_of",
            "floor": "part_of",
            "column": "part_of",
            "arch": "part_of",
        },
    },
}

SEMANTIC_CLASS_NAMES = tuple(
    label
    for label, _ in sorted(
        SEMANTIC_CLASS_REGISTRY.items(),
        key=lambda item: item[1]["id"],
    )
)

ARCHITECTURAL_CLASS_RULES = {
    label: {
        "role": data["role"],
        "can_support": set(data["can_support"]),
        "can_rest_on": set(data["can_rest_on"]),
        "part_of": dict(data["part_of"]),
    }
    for label, data in SEMANTIC_CLASS_REGISTRY.items()
}

SEMANTIC_LABEL_ALIASES = {
    "arch": "arch",
    "arches": "arch",
    "arco": "arch",
    "archi": "arch",
    "column": "column",
    "columns": "column",
    "colonna": "column",
    "colonne": "column",
    "molding": "moldings",
    "moldings": "moldings",
    "moulding": "moldings",
    "mouldings": "moldings",
    "modanatura": "moldings",
    "modanature": "moldings",
    "floor": "floor",
    "floors": "floor",
    "pavimento": "floor",
    "pavimenti": "floor",
    "door": "door_window",
    "doors": "door_window",
    "window": "door_window",
    "windows": "door_window",
    "opening": "door_window",
    "openings": "door_window",
    "porta": "door_window",
    "porte": "door_window",
    "finestra": "door_window",
    "finestre": "door_window",
    "wall": "wall",
    "walls": "wall",
    "muro": "wall",
    "muri": "wall",
    "parete": "wall",
    "pareti": "wall",
    "stairs": "stairs",
    "stair": "stairs",
    "scala": "stairs",
    "scale": "stairs",
    "vault": "vault",
    "vaults": "vault",
    "volta": "vault",
    "volte": "vault",
    "roof": "roof",
    "roofs": "roof",
    "tetto": "roof",
    "tetti": "roof",
    "copertura": "roof",
    "coperture": "roof",
    "other": "other",
    "altro": "other",
    "altri": "other",
}


def semantic_labels_by_category(category: str) -> list[str]:
    return [
        label
        for label in SEMANTIC_CLASS_NAMES
        if SEMANTIC_CLASS_REGISTRY[label]["category"] == category
    ]


def canonical_semantic_label(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace(" ", "_")
    if not normalized:
        return None
    return SEMANTIC_LABEL_ALIASES.get(normalized, normalized if normalized in SEMANTIC_CLASS_REGISTRY else None)
