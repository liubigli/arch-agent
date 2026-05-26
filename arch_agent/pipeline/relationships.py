import numpy as np


def compute_spatial_relationships(
    objects: dict,
    distance_threshold: float = 3.0,
) -> list[tuple[str, str, str]]:
    relationships = []
    names = list(objects.keys())

    for i, obj1 in enumerate(names):
        for obj2 in names[i + 1:]:
            rel = _determine_relationship(objects[obj1], objects[obj2], distance_threshold)
            if rel:
                relationships.append((obj1, obj2, rel))

    return relationships


def _determine_relationship(obj1: dict, obj2: dict, threshold: float) -> str | None:
    c1, c2 = obj1["centroid"], obj2["centroid"]
    b1, b2 = obj1["bounds"], obj2["bounds"]

    if np.linalg.norm(c1 - c2) > threshold:
        return None

    if _is_contained(b1, b2):
        return "inside"
    if _is_contained(b2, b1):
        return "contains"

    z_diff = float(c1[2] - c2[2])
    if abs(z_diff) > 0.5 and _overlap_xy(b1, b2):
        return "above" if z_diff > 0 else "below"

    if _are_adjacent(b1, b2, tolerance=0.3):
        return "adjacent"

    return "near"


def _is_contained(b1: dict, b2: dict, t: float = 0.02) -> bool:
    return (
        np.all(b1["min"] >= b2["min"] - t) and
        np.all(b1["max"] <= b2["max"] + t)
    )


def _overlap_xy(b1: dict, b2: dict) -> bool:
    x_ok = not (b1["max"][0] < b2["min"][0] or b2["max"][0] < b1["min"][0])
    y_ok = not (b1["max"][1] < b2["min"][1] or b2["max"][1] < b1["min"][1])
    return x_ok and y_ok


def _are_adjacent(b1: dict, b2: dict, tolerance: float = 0.1) -> bool:
    for axis in range(3):
        if (abs(b1["max"][axis] - b2["min"][axis]) < tolerance or
                abs(b2["max"][axis] - b1["min"][axis]) < tolerance):
            return True
    return False
