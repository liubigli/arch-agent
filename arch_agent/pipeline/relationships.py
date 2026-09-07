import re

import numpy as np

from ..semantic_schema import (
    ARCHITECTURAL_CLASS_RULES,
    SEMANTIC_CLASS_REGISTRY,
    canonical_semantic_label,
)

Relationship = tuple[str, str, str, str]

GEOMETRIC_LEVEL = "geometric"
ARCHITECTURAL_RULE_LEVEL = "architectural_rule"
CSV_METADATA_LEVEL = "csv_metadata"
STRUCTURAL_EVIDENCE_LEVEL = "csv_structural_evidence"
CIDOC_KG_LEVEL = "cidoc_knowledge_graph"

RELATIONSHIP_LAYER_ORDER = ("L1",)
RELATIONSHIP_LAYER_NAMES = {
    "L1": "spatial_graph",
    "L2": "csv_detail",
    "L3": CIDOC_KG_LEVEL,
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


def semantic_class_definition(label: str | None) -> dict:
    return dict(SEMANTIC_CLASS_REGISTRY.get(label or "", {}))


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


def _has_relationship_contact(child: dict, parent: dict, max_gap: float) -> bool:
    child_bounds = child["bounds"]
    parent_bounds = parent["bounds"]
    if _bounds_gap(child_bounds, parent_bounds) <= max_gap:
        return True
    if _is_adjacent_laterally(child, parent, max_gap):
        return True
    if _is_above(child, parent) or _is_above(parent, child):
        return True
    return (
        _overlap_xy_ratio(child_bounds, parent_bounds) >= 0.05
        and _axis_overlap_ratio(child_bounds, parent_bounds, axis=2) >= 0.05
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


def compute_csv_annotation_relationships(
    objects: dict,
    object_annotations: dict | None = None,
) -> list[Relationship]:
    if not object_annotations:
        return []

    relationships: list[Relationship] = []
    seen = set()

    def add(src: str, tgt: str, rel_type: str, level: str = CSV_METADATA_LEVEL) -> None:
        item = (src, tgt, rel_type, level)
        if item not in seen:
            relationships.append(item)
            seen.add(item)

    for object_name, annotations in object_annotations.items():
        source_label = objects.get(object_name, {}).get("semantic_label")
        if not source_label:
            continue

        for annotation in annotations:
            for target_label in _csv_support_target_labels(annotation, source_label):
                for target_name in _objects_with_semantic_label(objects, target_label):
                    add(object_name, target_name, "supports", STRUCTURAL_EVIDENCE_LEVEL)
                    add(target_name, object_name, "rests_on", STRUCTURAL_EVIDENCE_LEVEL)

            for source_support_label in _csv_supported_by_labels(annotation, source_label):
                for source_name in _objects_with_semantic_label(objects, source_support_label):
                    add(source_name, object_name, "supports", STRUCTURAL_EVIDENCE_LEVEL)
                    add(object_name, source_name, "rests_on", STRUCTURAL_EVIDENCE_LEVEL)

            for parent_label, relation_type in _csv_part_of_targets(annotation, source_label):
                for parent_name in _objects_with_semantic_label(objects, parent_label):
                    add(object_name, parent_name, relation_type)
                    add(parent_name, object_name, "has_part")

            for child_label in _csv_has_part_targets(annotation, source_label):
                for child_name in _objects_with_semantic_label(objects, child_label):
                    relation_type = mereological_relation_type(child_label, source_label) or "part_of"
                    add(child_name, object_name, relation_type)
                    add(object_name, child_name, "has_part")

    return relationships


def _objects_with_semantic_label(objects: dict, semantic_label: str) -> list[str]:
    return [
        name
        for name, obj in objects.items()
        if obj.get("semantic_label") == semantic_label
    ]


def _csv_support_target_labels(annotation: dict, source_label: str) -> list[str]:
    explicit_text = _annotation_first_value(
        annotation,
        (
            "supports",
            "supporta",
            "sostiene",
            "sorregge",
            "support_target",
            "supported_object",
            "supported_class",
            "structural_supports",
        ),
    )
    labels = _labels_mentioned_in_annotation_value(explicit_text, exclude={source_label})
    if labels:
        return labels

    descriptive_text = " ".join(
        str(annotation.get(key, "") or "")
        for key in (
            "function",
            "funzione",
            "description",
            "descrizione",
            "historical_description",
            "descrizione_storica",
            "notes",
            "note",
            "structural_evidence",
            "evidenza_strutturale",
        )
    )
    normalized = _normalize_annotation_text(descriptive_text)
    support_terms = (
        "support",
        "sostegn",
        "sosten",
        "sorregg",
        "regge",
        "portante",
        "load_bearing",
    )
    if not any(term in normalized for term in support_terms):
        return []
    return _labels_mentioned_in_annotation_value(descriptive_text, exclude={source_label})


def _csv_supported_by_labels(annotation: dict, source_label: str) -> list[str]:
    explicit_text = _annotation_first_value(
        annotation,
        (
            "supported_by",
            "supportato_da",
            "sostenuto_da",
            "sorretta_da",
            "sorretto_da",
            "rests_on",
            "resting_on",
            "appoggia_su",
            "appoggiato_su",
            "structural_supported_by",
        ),
    )
    return _labels_mentioned_in_annotation_value(explicit_text, exclude={source_label})


def _csv_part_of_targets(annotation: dict, source_label: str) -> list[tuple[str, str]]:
    labels = _labels_mentioned_in_annotation_value(
        _annotation_first_value(
            annotation,
            (
                "part_of",
                "parte_di",
                "belongs_to",
                "appartiene_a",
                "parent_class",
                "parent_object",
            ),
        ),
        exclude={source_label},
    )
    return [
        (label, mereological_relation_type(source_label, label) or "part_of")
        for label in labels
    ]


def _csv_has_part_targets(annotation: dict, source_label: str) -> list[str]:
    return _labels_mentioned_in_annotation_value(
        _annotation_first_value(
            annotation,
            (
                "has_part",
                "contains_part",
                "contiene",
                "comprende",
                "child_class",
                "child_object",
            ),
        ),
        exclude={source_label},
    )


def _annotation_first_value(annotation: dict, keys: tuple[str, ...]) -> object | None:
    for key in keys:
        value = annotation.get(key)
        if value not in (None, ""):
            return value
    return None


def _labels_mentioned_in_annotation_value(
    value: object | None,
    exclude: set[str] | None = None,
) -> list[str]:
    if value is None:
        return []
    normalized = _normalize_annotation_text(str(value))
    exclude = exclude or set()
    labels = []
    for token in re.split(r"[^a-z0-9_]+", normalized):
        label = canonical_semantic_label(token)
        if label and label not in exclude and label not in labels:
            labels.append(label)
    return labels


def _normalize_annotation_text(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", " ", value.lower().replace("-", "_")).strip()


def compute_all_relations(
    objects: dict,
    distance_threshold: float | None = None,
    surface_contact_thresh: float = 0.10,
    object_annotations: dict | None = None,
) -> list[Relationship]:
    relationship_layers = compute_all_relations_stratified(
        objects,
        distance_threshold=distance_threshold,
        surface_contact_thresh=surface_contact_thresh,
        object_annotations=object_annotations,
    )
    return relationship_layers["all"]


def compute_all_relations_stratified(
    objects: dict,
    distance_threshold: float | None = None,
    surface_contact_thresh: float = 0.10,
    object_annotations: dict | None = None,
) -> dict[str, list[Relationship]]:
    threshold = distance_threshold or auto_threshold(objects)

    geometric = compute_spatial_relationships(objects, threshold)
    structural = compute_structural_relations(objects)
    composition = compute_mereological_relations(
        objects,
        surface_contact_thresh=max(surface_contact_thresh, min(threshold * 0.25, 0.75)),
    )
    csv_relationships = compute_csv_annotation_relationships(objects, object_annotations)
    l1_relationships = _deduplicate(
        geometric
        + structural
        + composition
        + csv_relationships
    )
    relationship_layers = {
        "L1": l1_relationships,
    }
    all_relationships = flatten_relationship_layers(relationship_layers)

    print(f"Spatial graph : {len(l1_relationships):>4} relationships")
    print(f"  - geometric/spatial      : {len(geometric):>4}")
    print(f"  - spatial architectural  : {len(structural) + len(composition):>4}")
    print(f"  - CSV/user metadata      : {len(csv_relationships):>4}")
    print("CSV detail    : descriptive metadata, not a graph")
    print("CIDOC/KG      : built from CSV detail when requested")

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
    relationships: list[Relationship] = []
    names = list(objects.keys())

    for i, name1 in enumerate(names):
        for name2 in names[i + 1:]:
            obj1 = objects[name1]
            obj2 = objects[name2]
            if _rests_on(obj1, obj2):
                relationships.append((name2, name1, "supports", ARCHITECTURAL_RULE_LEVEL))
                relationships.append((name1, name2, "rests_on", ARCHITECTURAL_RULE_LEVEL))
            if _rests_on(obj2, obj1):
                relationships.append((name1, name2, "supports", ARCHITECTURAL_RULE_LEVEL))
                relationships.append((name2, name1, "rests_on", ARCHITECTURAL_RULE_LEVEL))

    return _deduplicate(relationships)


def compute_mereological_relations(
    objects: dict,
    surface_contact_thresh: float = 0.10,
) -> list[Relationship]:
    relationships: list[Relationship] = []
    names = list(objects.keys())

    for child_name in names:
        child = objects[child_name]
        child_label = child.get("semantic_label")
        for parent_name in names:
            if child_name == parent_name:
                continue
            parent = objects[parent_name]
            parent_label = parent.get("semantic_label")
            relation_type = mereological_relation_type(child_label, parent_label)
            if not relation_type:
                continue
            if not _has_relationship_contact(child, parent, surface_contact_thresh):
                continue
            relationships.append((child_name, parent_name, relation_type, ARCHITECTURAL_RULE_LEVEL))
            relationships.append((parent_name, child_name, "has_part", ARCHITECTURAL_RULE_LEVEL))

    return _deduplicate(relationships)


def _mereological_relation(child_label: str, parent_label: str) -> str:
    return mereological_relation_type(child_label, parent_label) or "is_attached_to"
