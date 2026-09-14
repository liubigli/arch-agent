"""Semantic class registry for architectural point-cloud scenes."""

import re
import unicodedata

SEMANTIC_CLASS_REGISTRY = {
    "arch": {
        "lexicon": (
            "arch",
            "arches",
            "archi",
            "arco",
        ),
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
        "lexicon": (
            "column",
            "columns",
            "colonna",
            "colonne",
        ),
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
        "lexicon": (
            "molding",
            "moldings",
            "moulding",
            "mouldings",
            "modanatur",
            "modanatura",
            "modanature",
        ),
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
        "lexicon": (
            "floor",
            "floors",
            "pavimento",
            "pavimenti",
        ),
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
        "lexicon": (
            "door window",
            "door",
            "doors",
            "window",
            "windows",
            "opening",
            "openings",
            "porta",
            "porte",
            "finestra",
            "finestre",
            "apertura",
            "aperture",
            "porta finestra",
            "porte finestre",
        ),
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
        "lexicon": (
            "wall",
            "walls",
            "muro",
            "muri",
            "parete",
            "pareti",
        ),
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
        "lexicon": (
            "stairs",
            "stair",
            "scala",
            "scale",
        ),
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
        "lexicon": (
            "vault",
            "vaults",
            "volta",
            "volte",
        ),
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
        "lexicon": (
            "roof",
            "roofs",
            "tetto",
            "tetti",
            "copertura",
            "coperture",
        ),
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
        "lexicon": (
            "other",
            "altro",
            "altri",
        ),
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

def normalize_text(value: object) -> str:
    """Lowercase, strip accents, and collapse runs of whitespace."""
    normalized = unicodedata.normalize("NFKD", str(value).strip().lower())
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return " ".join(without_accents.split())


def normalize_lexicon_text(value: object) -> str:
    """normalize_text, with "-" and "_" folded to spaces.

    Lexicon matching must not depend on whether a CSV says "load-bearing",
    "load bearing" or "load_bearing", nor on whether a class is written
    "door_window" or "door window".
    """
    return normalize_text(str(value).replace("-", " ").replace("_", " "))


# Derived from the per-class "lexicon" above: the one place that maps human
# wording onto a class. It exists for prose written by people - CSV annotation
# text and user questions - and never for tool arguments, which are validated
# against a closed Literal enum instead.
SEMANTIC_LABEL_LEXICON = {
    normalize_lexicon_text(term): label
    for label, data in SEMANTIC_CLASS_REGISTRY.items()
    for term in (label, *data["lexicon"])
}

# Longest first, so "porta finestra" is preferred over "porta".
SEMANTIC_LEXICON_TERMS = tuple(
    sorted(SEMANTIC_LABEL_LEXICON, key=lambda term: (-len(term), term))
)


def semantic_labels_by_category(category: str) -> list[str]:
    return [
        label
        for label in SEMANTIC_CLASS_NAMES
        if SEMANTIC_CLASS_REGISTRY[label]["category"] == category
    ]


def canonical_semantic_label(value: object | None) -> str | None:
    """Map a single human term onto a class name, or None if it is not one."""
    if value is None:
        return None
    return SEMANTIC_LABEL_LEXICON.get(normalize_lexicon_text(value))


def labels_mentioned_in_text(
    value: object | None,
    exclude: set[str] | None = None,
) -> list[str]:
    """Classes named anywhere in a free-text value, in order of appearance.

    Overlapping matches resolve to the longest term, so "porta finestra" reads
    as door_window rather than as porta followed by finestra. Each class is
    reported once, however many times it is mentioned.
    """
    if value is None:
        return []
    text = normalize_lexicon_text(value)
    if not text:
        return []
    exclude = exclude or set()

    matches: list[tuple[int, int, str]] = []
    for term in SEMANTIC_LEXICON_TERMS:
        label = SEMANTIC_LABEL_LEXICON[term]
        if label in exclude:
            continue
        for match in re.finditer(rf"\b{re.escape(term)}\b", text):
            matches.append((match.start(), -len(term), label))

    labels: list[str] = []
    occupied: set[int] = set()
    for start, negative_length, label in sorted(matches):
        span = set(range(start, start - negative_length))
        if occupied & span:
            continue
        occupied.update(span)
        if label not in labels:
            labels.append(label)
    return labels
