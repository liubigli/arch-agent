import numpy as np

Relationship = tuple[str, str, str, str]

GEOMETRIC_LEVEL = "geometric"
STRUCTURAL_LEVEL = "structural"
MEREOLOGICAL_LEVEL = "mereological"

MEREOLOGICAL_RULES = {
    "door_window": ["wall"],
    "moldings": ["wall", "arch", "vault", "column"],
    "other": ["wall", olumn", "arch"],
    "arch": ["wall", "vault"],
    "stairs": ["floor", "wall"],
}

SUPPORTING_LABELS = {
    "floor",
    "wall",
    "column",
    "arch",
    "vault",
    "stairs",
    "roof",
}

def _xy_area(bounds: dict) -> float:
    dims = bounds["max"][:2] - bounds["min"][:2]
    return float(max(dims[0], 0.0) * max(dims[1], 0.0))

def _overlap_xy_ratio(b1: dict, b2: dict) -> float:
    x_overlap = max(0.0, min(b1["max"][0], b2["max"][0]) - max(b1["min"][0], b2["min"][0]))
    y_overlap = max(0.0, min(b1["max"][1], b2["max"][1]) - max(b1["min"][1], b2["min"][1]))
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
        _axis_gap(float(b1["min"][axis]), float(b1["max"][axis]), float(b2["min"][axis]), float(b2["max"][axis]))
        for axis in range(3)
    ]
    return float(np.linalg.norm(gaps))


def _vertical_gap(upper_bounds: dict, lower_bounds: dict) -> float:
    return float(upper_bounds["min"][2] - lower_bounds["max"][2])


def _is_above(upper: dict, lower: dict, max_gap: float = 0.75) -> bool:
    upper_bounds = upper["bounds"]
    lower_bounds = lower["bounds"]
    z_gap = _vertical_gap(upper_bounds, lower_bounds)
    return (
        upper["centroid"][2] > lower["centroid"][2]
        and -0.15 <= z_gap <= max_gap
        and _overlap_xy_ratio(upper_bounds, lower_bounds) >= 0.05
    )


def _rests_on(upper: dict, lower: dict) -> bool:
    if lower.get("semantic_label") not in SUPPORTING_LABELS:
        return False

    upper_bounds = upper["bounds"]
    lower_bounds = lower["bounds"]

    z_gap = _vertical_gap(upper_bounds, lower_bounds)

    return (
        upper["centroid"][2] > lower["centroid"][2]
        and -0.15 <= z_gap <= 0.35
        and _overlap_xy_ratio(upper_bounds, lower_bounds) >= 0.10
    )

def _deduplicate(relationships: list[Relationship]) -> list[Relationship]:
    deduped = []
    seen = set()

    for relationship in relationships:
        if relationship not in seen:
            deduped.append(relationship)
            seen.add(relationship)

    return deduped

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
    threshold = distance_threshold or auto_threshold(objects)

    geometric = compute_spatial_relationships(objects, threshold)
    structural = compute_structural_relations(objects)
    mereological = compute_mereological_relations(
        objects,
        surface_contact_thresh=surface_contact_thresh,
    )

    print(f"L1 geometric    : {len(geometric):>4} relationships")
    print(f"L2 structural   : {len(structural):>4} relationships")
    print(f"L3 mereological : {len(mereological):>4} relationships")

    return _deduplicate(geometric + structural + mereological)


def compute_all_relations_stratified(
    objects: dict,
    distance_threshold: float | None = None,
    surface_contact_thresh: float = 0.10,
) -> dict[str, list[Relationship]]:
    threshold = distance_threshold or auto_threshold(objects)

    geometric = compute_spatial_relationships(objects, threshold)
    structural = compute_structural_relations(objects)
    mereological = compute_mereological_relations(
        objects,
        surface_contact_thresh=surface_contact_thresh,
    )
    all_relationships = _deduplicate(geometric + structural + mereological)

    print(f"L1 geometric    : {len(geometric):>4} relationships")
    print(f"L2 structural   : {len(structural):>4} relationships")
    print(f"L3 mereological : {len(mereological):>4} relationships")

    return {
        "L1": geometric,
        "L2": structural,
        "L3": mereological,
        "all": all_relationships,
    }


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

    if _is_above(obj1, obj2):
        relationships.append((name1, name2, "above", GEOMETRIC_LEVEL))
        relationships.append((name2, name1, "below", GEOMETRIC_LEVEL))
    elif _is_above(obj2, obj1):
        relationships.append((name2, name1, "above", GEOMETRIC_LEVEL))
        relationships.append((name1, name2, "below", GEOMETRIC_LEVEL))

    if _bounds_gap(obj1["bounds"], obj2["bounds"]) <= min(distance_threshold * 0.25, 0.75):
        relationships.append((name1, name2, "adjacent_to", GEOMETRIC_LEVEL))
        relationships.append((name2, name1, "adjacent_to", GEOMETRIC_LEVEL))

    return relationships

def compute_structural_relations(objects: dict) -> list[Relationship]:
    relationships: list[Relationship] = []
    names = list(objects.keys())

    floors = [name for name in names if objects[name]["semantic_label"] == "floor"]
    columns = [name for name in names if objects[name]["semantic_label"] == "column"]
    roofs = [name for name in names if objects[name]["semantic_label"] == "roof"]

    for column in columns:
        for roof in roofs:
            if _rests_on(objects[roof], objects[column]):
                relationships.append((column, roof, "supports", STRUCTURAL_LEVEL))
                relationships.append((roof, column, "rests_on", STRUCTURAL_LEVEL))

        for floor in floors:
            if _rests_on(objects[column], objects[floor]):
                relationships.append((floor, column, "supports", STRUCTURAL_LEVEL))
                relationships.append((column, floor, "rests_on", STRUCTURAL_LEVEL))

    for i, obj1 in enumerate(names):
        for obj2 in names[i + 1:]:
            if _rests_on(objects[obj1], objects[obj2]):
                relationships.extend([
                    (obj2, obj1, "supports", STRUCTURAL_LEVEL),
                    (obj1, obj2, "rests_on", STRUCTURAL_LEVEL),
                ])
            if _rests_on(objects[obj2], objects[obj1]):
                relationships.extend([
                    (obj1, obj2, "supports", STRUCTURAL_LEVEL),
                    (obj2, obj1, "rests_on", STRUCTURAL_LEVEL),
                ])

    return _deduplicate(relationships)


def compute_mereological_relations(
    objects: dict,
    surface_contact_thresh: float = 0.10,
) -> list[Relationship]:
    relationships: list[Relationship] = []
    names = list(objects.keys())

    for child_name in names:
        child = objects[child_name]
        child_label = child["semantic_label"]
        parent_labels = MEREOLOGICAL_RULES.get(child_label, [])
        if not parent_labels:
            continue

        for parent_name in names:
            if parent_name == child_name:
                continue

            parent = objects[parent_name]
            if parent["semantic_label"] not in parent_labels:
                continue

            contact_like = (
                _overlap_xy_ratio(child["bounds"], parent["bounds"]) >= surface_contact_thresh
                or _bounds_gap(child["bounds"], parent["bounds"]) <= 0.35
            )
            if not contact_like:
                continue

            rel = _mereological_relation(child_label, parent["semantic_label"])
            relationships.append((child_name, parent_name, rel, MEREOLOGICAL_LEVEL))
            relationships.append((parent_name, child_name, "has_part", MEREOLOGICAL_LEVEL))

    return _deduplicate(relationships)


def _mereological_relation(child_label: str, parent_label: str) -> str:
    if child_label == "door_window" and parent_label == "wall":
        return "is_opening_in"
    if child_label == "moldings":
        return "is_ornament_of"
    if child_label == "arch" and parent_label == "vault":
        return "is_rib_of"
    if child_label == "stairs" and parent_label == "floor":
        return "is_placed_on"
    if child_label == "stairs" and parent_label == "wall":
        return "is_connected_to"
    if child_label == "other":
        return "part_of"
    return "is_attached_to"
