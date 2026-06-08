import numpy as np

Relationship = tuple[str, str, str, str]

GEOMETRIC_LEVEL = "geometric"
STRUCTURAL_LEVEL = "structural"
MEREOLOGICAL_LEVEL = "mereological"

MEREOLOGICAL_RULES = {
    "door_window": ["wall"],
    "moldings": ["wall", "arch", "vault", "column"],
    "other": ["wall", "floor", "column", "arch"],
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


def _rests_on(upper: dict, lower: dict) -> bool:
    if lower.get("semantic_label") not in SUPPORTING_LABELS:
        return False

    upper_bounds = upper["bounds"]
    lower_bounds = lower["bounds"]

    upper_min_z = float(upper_bounds["min"][2])
    lower_max_z = float(lower_bounds["max"][2])
    z_gap = upper_min_z - lower_max_z

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


def compute_structural_relations(objects: dict) -> list[Relationship]:
    relationships: list[Relationship] = []
    names = list(objects.keys())

    floors = [name for name in names if objects[name]["semantic_label"] == "floor"]
    columns = [name for name in names if objects[name]["semantic_label"] == "column"]
    roofs = [name for name in names if objects[name]["semantic_label"] == "roof"]

    for column in columns:
        for roof in roofs:
            if _overlap_xy_ratio(objects[column]["bounds"], objects[roof]["bounds"]) >= 0.02:
                relationships.append((column, roof, "supports", STRUCTURAL_LEVEL))
                relationships.append((roof, column, "rests_on", STRUCTURAL_LEVEL))

        for floor in floors:
            if _overlap_xy_ratio(objects[column]["bounds"], objects[floor]["bounds"]) >= 0.02:
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