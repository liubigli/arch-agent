import numpy as np

Relationship = tuple[str, str, str, str]

GEOMETRIC_LEVEL = "geometric"
STRUCTURAL_EVIDENCE_LEVEL = "csv_structural_evidence"
CIDOC_KG_LEVEL = "cidoc_knowledge_graph"

RELATIONSHIP_LAYER_ORDER = ("L1",)
RELATIONSHIP_LAYER_NAMES = {
    "L1": GEOMETRIC_LEVEL,
    "L2": "csv_detail",
    "L3": CIDOC_KG_LEVEL,
}

ARCHITECTURAL_CLASS_RULES = {
    "arch": {
        "role": "structural",
        "can_support": {"vault", "roof"},
        "can_rest_on": {"column", "wall"},
        "part_of": {
            "wall": "is_attached_to",
            "vault": "is_rib_of",
        },
    },
    "column": {
        "role": "structural",
        "can_support": {"arch", "vault", "roof"},
        "can_rest_on": {"floor"},
        "part_of": {},
    },
    "wall": {
        "role": "structural",
        "can_support": {"arch", "vault", "roof"},
        "can_rest_on": {"floor"},
        "part_of": {},
    },
    "vault": {
        "role": "structural",
        "can_support": {"roof"},
        "can_rest_on": {"arch", "column", "wall"},
        "part_of": {},
    },
    "roof": {
        "role": "structural",
        "can_support": set(),
        "can_rest_on": {"arch", "column", "wall", "vault"},
        "part_of": {},
    },
    "floor": {
        "role": "support_surface",
        "can_support": {"column", "wall", "stairs"},
        "can_rest_on": set(),
        "part_of": {},
    },
    "stairs": {
        "role": "circulation",
        "can_support": set(),
        "can_rest_on": {"floor"},
        "part_of": {
            "floor": "is_placed_on",
            "wall": "is_connected_to",
        },
    },
    "moldings": {
        "role": "ornamental",
        "can_support": set(),
        "can_rest_on": set(),
        "part_of": {
            "wall": "is_ornament_of",
            "arch": "is_ornament_of",
            "column": "is_ornament_of",
        },
    },
    "door_window": {
        "role": "opening",
        "can_support": set(),
        "can_rest_on": set(),
        "part_of": {
            "wall": "is_opening_in",
        },
    },
    "other": {
        "role": "unknown",
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

SUPPORTING_LABELS = {
    label
    for label, rules in ARCHITECTURAL_CLASS_RULES.items()
    if rules.get("can_support")
}

MEREOLOGICAL_RULES = {
    label: list(rules["part_of"])
    for label, rules in ARCHITECTURAL_CLASS_RULES.items()
    if rules.get("part_of")
}

UNSUPPORTED_ABOVE_PAIRS = {
    ("column", "arch"),
}


def _xy_area(bounds: dict) -> float:
    dims = bounds["max"][:2] - bounds["min"][:2]
    return float(max(dims[0], 0.0) * max(dims[1], 0.0))


def _overlap_xy_ratio(b1: dict, b2: dict) -> float:
    x_overlap = max(
        0.0,
        min(b1["max"][0], b2["max"][0]) - max(b1["min"][0], b2["min"][0]),
    )
    y_overlap = max(
        0.0,
        min(b1["max"][1], b2["max"][1]) - max(b1["min"][1], b2["min"][1]),
    )
    overlap_area = x_overlap * y_overlap

    reference_area = min(_xy_area(b1), _xy_area(b2))
    if reference_area <= 0:
        return 0.0

    return float(overlap_area / reference_area)


def _axis_gap(min1: float, max1: float, min2: float, max2: float) -> float:
    if max1 < min2:
        return float(min2 - max1)
    if max2 < min1:
        return float(min1 - max2)
    return 0.0


def _bounds_gap(b1: dict, b2: dict) -> float:
    gaps = [
        _axis_gap(
            float(b1["min"][axis]),
            float(b1["max"][axis]),
            float(b2["min"][axis]),
            float(b2["max"][axis]),
        )
        for axis in range(3)
    ]
    return float(np.linalg.norm(gaps))


def _horizontal_gap(b1: dict, b2: dict) -> float:
    gaps = [
        _axis_gap(
            float(b1["min"][axis]),
            float(b1["max"][axis]),
            float(b2["min"][axis]),
            float(b2["max"][axis]),
        )
        for axis in range(2)
    ]
    return float(np.linalg.norm(gaps))


def _axis_overlap_ratio(b1: dict, b2: dict, axis: int) -> float:
    overlap = max(
        0.0,
        min(float(b1["max"][axis]), float(b2["max"][axis]))
        - max(float(b1["min"][axis]), float(b2["min"][axis])),
    )
    size1 = max(float(b1["max"][axis]) - float(b1["min"][axis]), 0.0)
    size2 = max(float(b2["max"][axis]) - float(b2["min"][axis]), 0.0)
    reference = min(size1, size2)
    if reference <= 0:
        return 0.0
    return float(overlap / reference)


def _vertical_gap(upper_bounds: dict, lower_bounds: dict) -> float:
    return float(upper_bounds["min"][2] - lower_bounds["max"][2])


def _is_above(upper: dict, lower: dict, max_gap: float = 0.75) -> bool:
    upper_label = upper.get("semantic_label")
    lower_label = lower.get("semantic_label")

    if (upper_label, lower_label) in UNSUPPORTED_ABOVE_PAIRS:
        return False

    upper_bounds = upper["bounds"]
    lower_bounds = lower["bounds"]
    z_gap = _vertical_gap(upper_bounds, lower_bounds)
    return (
        upper["centroid"][2] > lower["centroid"][2]
        and -0.15 <= z_gap <= max_gap
        and _overlap_xy_ratio(upper_bounds, lower_bounds) >= 0.05
    )


def supports_label_pair(lower_label: str | None, upper_label: str | None) -> bool:
    lower_rules = ARCHITECTURAL_CLASS_RULES.get(lower_label or "", {})
    upper_rules = ARCHITECTURAL_CLASS_RULES.get(upper_label or "", {})
    return (
        upper_label in lower_rules.get("can_support", set())
        and lower_label in upper_rules.get("can_rest_on", set())
    )


def mereological_relation_type(
    child_label: str | None,
    parent_label: str | None,
) -> str | None:
    rules = ARCHITECTURAL_CLASS_RULES.get(child_label or "", {})
    return rules.get("part_of", {}).get(parent_label)


def architectural_role(label: str | None) -> str:
    rules = ARCHITECTURAL_CLASS_RULES.get(label or "", {})
    return rules.get("role", "unknown")


def _rests_on(upper: dict, lower: dict) -> bool:
    upper_label = upper.get("semantic_label")
    lower_label = lower.get("semantic_label")

    if not supports_label_pair(lower_label, upper_label):
        return False

    upper_bounds = upper["bounds"]
    lower_bounds = lower["bounds"]

    z_gap = _vertical_gap(upper_bounds, lower_bounds)

    return (
        upper["centroid"][2] > lower["centroid"][2]
        and -0.15 <= z_gap <= 0.35
        and _overlap_xy_ratio(upper_bounds, lower_bounds) >= 0.10
    )


def _is_adjacent_laterally(
    obj1: dict,
    obj2: dict,
    max_gap: float,
) -> bool:
    label1 = obj1.get("semantic_label")
    label2 = obj2.get("semantic_label")

    # A floor is a horizontal support surface: relations with vertical
    # architectural elements should be represented by above/below, not
    # lateral adjacency.
    if "floor" in {label1, label2} and label1 != label2:
        return False

    if _is_above(obj1, obj2) or _is_above(obj2, obj1):
        return False

    b1 = obj1["bounds"]
    b2 = obj2["bounds"]
    if _horizontal_gap(b1, b2) > max_gap:
        return False

    # Lateral adjacency requires the two elements to share a comparable
    # vertical range. This avoids treating stacked elements as adjacent.
    if _axis_overlap_ratio(b1, b2, axis=2) < 0.10:
        return False

    x_overlap = _axis_overlap_ratio(b1, b2, axis=0)
    y_overlap = _axis_overlap_ratio(b1, b2, axis=1)
    x_gap = _axis_gap(float(b1["min"][0]), float(b1["max"][0]), float(b2["min"][0]), float(b2["max"][0]))
    y_gap = _axis_gap(float(b1["min"][1]), float(b1["max"][1]), float(b2["min"][1]), float(b2["max"][1]))

    return (
        (x_gap <= max_gap and y_overlap >= 0.05)
        or (y_gap <= max_gap and x_overlap >= 0.05)
    )


def _deduplicate(relationships: list[Relationship]) -> list[Relationship]:
    deduped = []
    seen = set()

    for relationship in relationships:
        if relationship not in seen:
            deduped.append(relationship)
            seen.add(relationship)

    return deduped


def flatten_relationship_layers(
    relationship_layers: dict[str, list[Relationship]],
) -> list[Relationship]:
    relationships: list[Relationship] = []
    for level in RELATIONSHIP_LAYER_ORDER:
        relationships.extend(relationship_layers.get(level, []))
    return _deduplicate(relationships)


def auto_threshold(objects: dict, scale: float = 2.5, fallback: float = 3.0) -> float:
    if len(objects) < 2:
        return fallback

    centroids = np.array([obj["centroid"] for obj in objects.values()], dtype=float)
    distances = []
    for i, c1 in enumerate(centroids):
        for c2 in centroids[i + 1:]:
            distances.append(float(np.linalg.norm(c1 - c2)))

    if not distances:
        return fallback

    return max(float(np.median(distances) / scale), 0.5)


def compute_all_relations(
    objects: dict,
    distance_threshold: float | None = None,
    surface_contact_thresh: float = 0.10,
) -> list[Relationship]:
    relationship_layers = compute_all_relations_stratified(
        objects,
        distance_threshold=distance_threshold,
        surface_contact_thresh=surface_contact_thresh,
    )
    return relationship_layers["all"]


def compute_all_relations_stratified(
    objects: dict,
    distance_threshold: float | None = None,
    surface_contact_thresh: float = 0.10,
) -> dict[str, list[Relationship]]:
    threshold = distance_threshold or auto_threshold(objects)

    geometric = compute_spatial_relationships(objects, threshold)
    relationship_layers = {
        "L1": geometric,
    }
    all_relationships = flatten_relationship_layers(relationship_layers)

    print(f"L1 geometric : {len(geometric):>4} relationships")
    print("L2 CSV detail: loaded from annotation CSV, not a graph")
    print("L3 CIDOC/KG  : built from CSV detail when requested, not a mereological layer")

    return {**relationship_layers, "all": all_relationships}


def compute_spatial_relationships(
    objects: dict,
    distance_threshold: float = 3.0,
) -> list[Relationship]:
    relationships: list[Relationship] = []
    names = list(objects.keys())

    for i, obj1 in enumerate(names):
        for obj2 in names[i + 1:]:
            relationships.extend(
                _determine_geometric_relationships(
                    obj1,
                    objects[obj1],
                    obj2,
                    objects[obj2],
                    distance_threshold,
                )
            )

    return _deduplicate(relationships)


def _determine_geometric_relationships(
    name1: str,
    obj1: dict,
    name2: str,
    obj2: dict,
    distance_threshold: float,
) -> list[Relationship]:
    relationships: list[Relationship] = []
    c1 = np.asarray(obj1["centroid"], dtype=float)
    c2 = np.asarray(obj2["centroid"], dtype=float)
    centroid_distance = float(np.linalg.norm(c1 - c2))

    if centroid_distance <= distance_threshold:
        relationships.append((name1, name2, "near", GEOMETRIC_LEVEL))
        relationships.append((name2, name1, "near", GEOMETRIC_LEVEL))

    obj1_above_obj2 = _is_above(obj1, obj2)
    obj2_above_obj1 = _is_above(obj2, obj1)

    if obj1_above_obj2:
        relationships.append((name1, name2, "above", GEOMETRIC_LEVEL))
        relationships.append((name2, name1, "below", GEOMETRIC_LEVEL))
    elif obj2_above_obj1:
        relationships.append((name2, name1, "above", GEOMETRIC_LEVEL))
        relationships.append((name1, name2, "below", GEOMETRIC_LEVEL))

    adjacent_gap = min(distance_threshold * 0.25, 0.75)
    if _is_adjacent_laterally(obj1, obj2, adjacent_gap):
        relationships.append((name1, name2, "adjacent_to", GEOMETRIC_LEVEL))
        relationships.append((name2, name1, "adjacent_to", GEOMETRIC_LEVEL))

    return relationships


def compute_structural_relations(objects: dict) -> list[Relationship]:
    """Deprecated compatibility hook.

    Structural evidence is no longer inferred from geometry/class rules here.
    It must come from scene CSV annotations or explicit user-provided metadata.
    """
    return []


def compute_mereological_relations(
    objects: dict,
    surface_contact_thresh: float = 0.10,
) -> list[Relationship]:
    """Deprecated compatibility hook.

    L3 is now a CIDOC/knowledge-graph layer built from CSV/user metadata and
    supported geometric context, not a mereological edge set inferred here.
    """
    return []


def _mereological_relation(child_label: str, parent_label: str) -> str:
    return mereological_relation_type(child_label, parent_label) or "is_attached_to"
