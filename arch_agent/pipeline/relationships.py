import numpy as np

Relationship = tuple[str, str, str]
StratifiedRelationships = dict[str, list[Relationship]]

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


def compute_all_relations_stratified(
    objects: dict,
    distance_threshold: float | None = None,
    surface_contact_thresh: float = 0.10,
) -> StratifiedRelationships:
    threshold = distance_threshold or auto_threshold(objects)

    L1 = compute_spatial_relationships(objects, threshold)
    L2 = compute_structural_relations(objects)
    L3 = compute_mereological_relations(
        objects,
        surface_contact_thresh=surface_contact_thresh,
    )

    print(
        f"L1 geometric    : {len(L1):>4} relationships "
        f"(adjacent_to, above, near, below)"
    )
    print(
        f"L2 structural   : {len(L2):>4} relationships "
        f"(supports, rests_on)"
    )
    print(
        f"L3 mereological : {len(L3):>4} relationships "
        f"(part_of, has_part, is_opening_in, is_attached_to, ...)"
    )

    return {
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "all": L1 + L2 + L3,
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


def compute_structural_relations(objects: dict) -> list[Relationship]:
    relationships: list[Relationship] = []
    names = list(objects.keys())
