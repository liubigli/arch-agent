"""Helpers shared by more than one tool module.

Kept private to arch_agent.tools; the tool modules import from here rather
than from each other.
"""

from typing import Optional
import re
import unicodedata

from langchain_core.tools import tool
import networkx as nx

from ..pipeline.pipeline import SceneContext
from ..settings import get_config


_SEMANTIC_ALIASES = (
    ("porta finestra", "door_window"),
    ("porta-finestra", "door_window"),
    ("porte finestre", "door_window"),
    ("porte-finestre", "door_window"),
    ("openings", "door_window"),
    ("opening", "door_window"),
    ("aperture", "door_window"),
    ("apertura", "door_window"),
    ("porte", "door_window"),
    ("porta", "door_window"),
    ("finestre", "door_window"),
    ("finestra", "door_window"),
    ("doors", "door_window"),
    ("door", "door_window"),
    ("windows", "door_window"),
    ("window", "door_window"),
    ("archi", "arch"),
    ("arco", "arch"),
    ("arches", "arch"),
    ("arch", "arch"),
    ("colonne", "column"),
    ("colonna", "column"),
    ("columns", "column"),
    ("column", "column"),
    ("muri", "wall"),
    ("muro", "wall"),
    ("pareti", "wall"),
    ("parete", "wall"),
    ("walls", "wall"),
    ("wall", "wall"),
    ("pavimenti", "floor"),
    ("pavimento", "floor"),
    ("floors", "floor"),
    ("floor", "floor"),
    ("tetti", "roof"),
    ("tetto", "roof"),
    ("coperture", "roof"),
    ("copertura", "roof"),
    ("roofs", "roof"),
    ("roof", "roof"),
    ("volte", "vault"),
    ("volta", "vault"),
    ("vaults", "vault"),
    ("vault", "vault"),
    ("scale", "stairs"),
    ("scala", "stairs"),
    ("stairs", "stairs"),
    ("stair", "stairs"),
    ("modanatur", "moldings"),
    ("modanature", "moldings"),
    ("modanatura", "moldings"),
    ("moldings", "moldings"),
    ("molding", "moldings"),
    ("altro", "other"),
    ("other", "other"),
)


def _absent_semantic_labels_message(labels: list[str]) -> str:
    label_text = ", ".join(labels)
    lines = ["REQUESTED CLASS STATUS"]
    lines.extend(f"- {label}: absent (0 object(s))" for label in labels)
    lines.extend([
        "",
        "SCENE EVIDENCE",
        f"- No objects found for requested semantic_label(s): {label_text}.",
        "- Relationships involving absent classes cannot be observed in this scene.",
        "",
        "TOOL CONCLUSION",
        "- Count for the requested absent class set is 0.",
        "- No scene relationship can be reported for absent class(es).",
    ])
    return "\n".join(lines)


def _all_semantic_classes() -> list[str]:
    return list(get_config()["semantic_classes"]["names"])


def _canonical_semantic_label(value: Optional[str]) -> Optional[str]:
    value = _clean_optional(value)
    if value is None:
        return None
    normalized = _normalize_text(value)
    valid_labels = set(_all_semantic_classes())
    if normalized in valid_labels:
        return normalized
    for alias, label in _SEMANTIC_ALIASES:
        if _normalize_text(alias) == normalized:
            return label
    return value


def _canonical_semantic_label_list(values: list[str]) -> list[str]:
    labels: list[str] = []
    seen = set()
    for value in values:
        parts = re.split(r"[,;]|\s+e\s+|\s+and\s+", str(value))
        for part in parts:
            label = _canonical_semantic_label(part)
            if not label or label in seen:
                continue
            labels.append(label)
            seen.add(label)
    return labels


def _clean_optional(value: Optional[str]) -> Optional[str]:
    """Normalize an "unset" argument.

    Some local models (via Ollama tool-calling) emit the literal string
    "None"/"null" instead of omitting an optional argument or passing JSON
    null; treat those the same as not provided.
    """
    if value is None:
        return None
    if value.strip().lower() in ("none", "null", ""):
        return None
    return value


def _combined_graph(ctx: SceneContext) -> nx.DiGraph:
    graphs = list((ctx.scene_graphs or {}).values())
    if not graphs:
        return ctx.scene_graph or nx.DiGraph()
    return nx.compose_all(graphs)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text).strip().lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _object_box_center_text(ctx: SceneContext, object_name: str) -> str:
    obj = ctx.objects[object_name]
    box_center = (obj["bounds"]["min"] + obj["bounds"]["max"]) / 2.0
    return f"({box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f})"


def _object_not_found_message(object_name: str, objects: dict) -> str:
    matches = sorted(
        name for name, obj in objects.items()
        if obj["semantic_label"] == object_name or name.startswith(f"{object_name}_")
    )
    if matches:
        shown = ", ".join(matches[:8])
        more = f" (+{len(matches) - 8} more)" if len(matches) > 8 else ""
        return (
            f"'{object_name}' is a semantic class, not an object id. "
            f"Matching instances: {shown}{more}. Use one of these exact names."
        )

    sample: list[str] = []
    seen_labels: set[str] = set()
    for name, obj in objects.items():
        label = obj["semantic_label"]
        if label in seen_labels:
            continue
        seen_labels.add(label)
        sample.append(name)
        if len(sample) >= 8:
            break
    return f"Object '{object_name}' not found. Examples (one per class): {', '.join(sample)}"


def _objects_with_semantic_label(ctx: SceneContext, label: str) -> list[str]:
    return sorted(
        name
        for name, obj in ctx.objects.items()
        if obj.get("semantic_label") == label
    )


def _resolve_target_names(
    ctx,
    object_name: Optional[str],
    semantic_label: Optional[str],
    semantic_labels: Optional[list[str]] = None,
):
    """Resolve object/class filters into a list of exact object names.

    Returns a list[str] on success, or an error message string on failure —
    callers should check `isinstance(result, str)` and return it as-is.
    """
    object_name = _clean_optional(object_name)
    semantic_label = _canonical_semantic_label(semantic_label)
    requested_labels = _canonical_semantic_label_list(semantic_labels or [])
    target_filter_count = sum(
        1 for value in (object_name, semantic_label, requested_labels) if value
    )
    if target_filter_count == 0:
        return "Provide object_name, semantic_label, or semantic_labels."
    if target_filter_count > 1:
        return "Provide only one of object_name, semantic_label, or semantic_labels."
    if object_name:
        if object_name not in ctx.objects:
            object_as_label = _canonical_semantic_label(object_name)
            if object_as_label:
                names = sorted(
                    name for name, obj in ctx.objects.items()
                    if obj["semantic_label"] == object_as_label
                )
                if names:
                    return names
            return _object_not_found_message(object_name, ctx.objects)
        return [object_name]
    if requested_labels:
        names = []
        missing_labels = []
        for label in requested_labels:
            label_names = sorted(
                name for name, obj in ctx.objects.items()
                if obj["semantic_label"] == label
            )
            if label_names:
                names.extend(label_names)
            else:
                missing_labels.append(label)
        if not names:
            return _absent_semantic_labels_message(requested_labels)
        if missing_labels:
            return _TargetNameList(names, missing_labels)
        return names
    names = sorted(
        name for name, obj in ctx.objects.items()
        if obj["semantic_label"] == semantic_label
    )
    if not names:
        return _absent_semantic_labels_message([semantic_label])
    return names


def np_linalg_norm(values) -> float:
    return sum(float(value) ** 2 for value in values) ** 0.5


def _concat_frames(frames):
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


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
    normalized = _normalize_text(descriptive_text)
    support_terms = (
        "support",
        "sostegn",
        "sosten",
        "sorregg",
        "regge",
        "portante",
        "load-bearing",
        "load bearing",
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


def _labels_mentioned_in_annotation_value(
    value: object | None,
    exclude: set[str] | None = None,
) -> list[str]:
    if value is None:
        return []
    normalized = _normalize_text(str(value))
    exclude = exclude or set()
    labels: list[str] = []
    for alias, label in _SEMANTIC_ALIASES:
        if label in exclude or label in labels:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            labels.append(label)
    return labels


def _structural_elements() -> set[str]:
    return set(get_config()["semantic_classes"]["structural"])


def _scene_class_names(ctx: SceneContext) -> tuple[str, ...]:
    """Semantic classes actually present in the scene, else the full registry.

    Previously computed inline in create_scene_tools and closed over by the
    tools; it is shared state, so it lives here now.
    """
    return tuple(sorted({obj["semantic_label"] for obj in ctx.objects.values()})) or tuple(
        _all_semantic_classes()
    )
