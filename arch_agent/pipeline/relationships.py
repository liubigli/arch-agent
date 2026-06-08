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