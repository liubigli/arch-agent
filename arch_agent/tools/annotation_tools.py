"""Tools that expose the CSV metadata layer: material, typology, function and
description.

Every value served here comes from the scene annotation CSV. Nothing in this
module infers an attribute from geometry, colour or semantic class. Ablation
condition "graph" withholds this whole module from the agent."""


from typing import Optional
import unicodedata

from langchain_core.tools import tool

from ..pipeline.pipeline import SceneContext

from ._shared import (
    _canonical_semantic_label,
    _canonical_semantic_label_list,
    _clean_optional,
    _combined_graph,
    _normalize_text,
    _object_box_center_text,
    _object_not_found_message,
    _objects_with_semantic_label,
    _resolve_target_names,
    np_linalg_norm,
)


def create_annotation_tools(ctx: SceneContext) -> list:
    """Build the annotation tools bound to ``ctx``."""

    @tool
    def get_object_annotation(
        object_name: Optional[str] = None,
        semantic_label: Optional[str] = None,
        semantic_labels: Optional[list[str]] = None,
        position: Optional[str] = None,
    ) -> str:
        """Get user-provided CSV annotation/description for a matched object or class.

        Use this for material, typology, function, historical notes, or
        descriptive cards of a specific object/class. The result comes from
        matched CSV metadata only.

        Prefer get_object_semantic_details when the user asks for material,
        typology, and function of exact object ids. Use this tool when the
        question asks for raw annotation text, class-level annotation summaries,
        match uncertainty, or position-based selection.

        Prefer semantic_label plus position when the user does not know object
        ids, e.g. semantic_label='column', position='central'. If the user
        names multiple classes, pass them together as semantic_labels.

        Args:
            object_name: Optional matched object name, e.g. 'column_0'.
            semantic_label: Optional semantic class, e.g. 'column'.
            semantic_labels: Optional list of semantic classes to retrieve
                class-level annotations in one call.
            position: Optional spatial selector, e.g. 'central', 'left', 'north'.
        """
        semantic_label = _canonical_semantic_label(semantic_label)
        requested_labels = _canonical_semantic_label_list(semantic_labels or [])
        object_name = _clean_optional(object_name)
        if requested_labels and (object_name or semantic_label or position):
            return "Provide semantic_labels by itself, or use object_name/semantic_label/position for a single target."
        if requested_labels:
            blocks = []
            for label in requested_labels:
                candidates = [
                    name for name, obj in ctx.objects.items()
                    if obj.get("semantic_label") == label
                ]
                annotated = [
                    (name, annotation)
                    for name in sorted(candidates)
                    for annotation in getattr(ctx, "object_annotations", {}).get(name, [])
                ]
                if annotated:
                    blocks.append(_format_annotations_for_class(ctx, label, annotated))
                elif candidates:
                    blocks.append(
                        f"CSV annotations for class {label}: no matched annotation "
                        f"on {len(candidates)} detected object(s)."
                    )
                else:
                    blocks.append(f"CSV annotations for class {label}: 0 detected object(s).")
            return "\n\n".join(blocks)
        if object_name and object_name not in ctx.objects and semantic_label is None:
            object_as_label = _canonical_semantic_label(object_name)
            if object_as_label and _objects_with_semantic_label(ctx, object_as_label):
                semantic_label = object_as_label
                object_name = None

        if object_name is None and semantic_label and position is None:
            candidates = [
                name for name, obj in ctx.objects.items()
                if obj.get("semantic_label") == semantic_label
            ]
            annotated = [
                (name, annotation)
                for name in sorted(candidates)
                for annotation in getattr(ctx, "object_annotations", {}).get(name, [])
            ]
            if annotated:
                return _format_annotations_for_class(ctx, semantic_label, annotated)

        if object_name is None:
            object_name = _resolve_annotation_object(
                ctx,
                semantic_label=semantic_label,
                position=position,
            )
            if object_name is None:
                return _annotation_resolution_error(ctx, semantic_label=semantic_label)
        if object_name not in ctx.objects:
            return _object_not_found_message(object_name, ctx.objects)
        annotations = getattr(ctx, "object_annotations", {}).get(object_name, [])
        if not annotations:
            return f"No CSV annotation is associated with {object_name}."
        return _format_annotations_for_object(ctx, object_name, annotations)

    @tool
    def list_csv_annotation_matches(
        semantic_label: Optional[str] = None,
        limit: int = 80,
    ) -> str:
        """List CSV match status for detected objects and unmatched CSV rows.

        Use this when the user asks which objects have CSV correspondence,
        which objects do not have CSV correspondence, or whether a class/object
        is linked to the attached CSV. Matching is based on CSV
        global_box_center_x/y/z against each object's global AABB box center.

        Args:
            semantic_label: Optional semantic class filter, e.g. 'moldings'.
            limit: Maximum number of rows to show.
        """
        semantic_label = _clean_optional(semantic_label)
        semantic_label = _canonical_semantic_label(semantic_label)
        max_rows = max(1, min(int(limit), 300))
        object_annotations = getattr(ctx, "object_annotations", {})
        unmatched_annotations = getattr(ctx, "unmatched_annotations", [])

        object_names = [
            name for name, obj in sorted(ctx.objects.items())
            if semantic_label is None or obj.get("semantic_label") == semantic_label
        ]
        if semantic_label and not object_names:
            return f"No objects found for semantic_label={semantic_label!r}."

        matched_objects = [
            name for name in object_names
            if object_annotations.get(name)
        ]
        missing_objects = [
            name for name in object_names
            if not object_annotations.get(name)
        ]
        unmatched_rows = [
            annotation for annotation in unmatched_annotations
            if semantic_label is None or annotation.get("semantic_label") == semantic_label
        ]

        scope = f" for class {semantic_label}" if semantic_label else ""
        lines = [
            f"CSV annotation match status{scope}:",
            f"  Detected objects checked: {len(object_names)}",
            f"  Objects with matched CSV annotation: {len(matched_objects)}",
            f"  Objects without matched CSV annotation: {len(missing_objects)}",
            f"  Unmatched CSV rows in scope: {len(unmatched_rows)}",
            "  Match method: CSV global_box_center_x/y/z -> object AABB box_center.",
        ]

        rows_written = 0
        if matched_objects:
            lines.append("Matched objects:")
            for name in matched_objects:
                if rows_written >= max_rows:
                    break
                obj = ctx.objects[name]
                box_center = _object_box_center_text(ctx, name)
                annotations = object_annotations.get(name, [])
                lines.append(
                    f"  - {name} ({obj['semantic_label']}), "
                    f"box_center={box_center}: {len(annotations)} CSV row(s)"
                )
                rows_written += 1
                for annotation in annotations[:2]:
                    if rows_written >= max_rows:
                        break
                    match_text = _annotation_match_text(annotation)
                    summary = _annotation_summary_text(annotation)
                    if summary:
                        lines.append(
                            f"      row {annotation.get('source_row')}; {match_text}; {summary}"
                        )
                    else:
                        lines.append(
                            f"      row {annotation.get('source_row')}; {match_text}"
                        )
                    rows_written += 1

        if missing_objects:
            remaining = max_rows - rows_written
            shown_missing = missing_objects[:max(0, remaining)]
            lines.append(
                "Objects without matched CSV annotation: "
                + ", ".join(shown_missing)
            )
            rows_written += len(shown_missing)

        if unmatched_rows and rows_written < max_rows:
            lines.append("Unmatched CSV rows:")
            for annotation in unmatched_rows[: max_rows - rows_written]:
                match = annotation.get("match", {})
                reason = match.get("reason", "unknown reason")
                nearest = match.get("nearest_object")
                distance = match.get("distance_m")
                distance_text = (
                    f", nearest={nearest}, distance={distance:.3f} m"
                    if distance is not None
                    else (f", nearest={nearest}" if nearest else "")
                )
                summary = _annotation_summary_text(annotation)
                if summary:
                    lines.append(
                        f"  - row {annotation.get('source_row')}: {reason}{distance_text}; {summary}"
                    )
                else:
                    lines.append(
                        f"  - row {annotation.get('source_row')}: {reason}{distance_text}"
                    )
                rows_written += 1

        hidden = (
            len(matched_objects)
            + sum(min(len(object_annotations.get(name, [])), 2) for name in matched_objects)
            + len(missing_objects)
            + len(unmatched_rows)
            - rows_written
        )
        if hidden > 0:
            lines.append(f"  ... {hidden} more row(s) not shown.")

        if not object_annotations and not unmatched_annotations:
            lines.append(
                "No CSV annotations are loaded. Start main.py with --annotation-csv or place a matching CSV next to the LAZ file."
            )
        return "\n".join(lines)

    @tool
    def find_objects_by_material(
        material: str,
        semantic_label: Optional[str] = None,
        limit: int = 30,
    ) -> str:
        """Find objects whose CSV material fields mention a requested material.

        Use this for scene-wide material questions such as "ci sono oggetti
        in legno?" or "are there wooden objects?". The search uses only CSV
        material fields: material/materiale/material_description/
        descrizione_materica. It does not use RGB, roughness, semantic class,
        or free historical descriptions to infer materials.

        Args:
            material: Requested material, e.g. 'legno', 'wood', 'marmo'.
            semantic_label: Optional class filter, e.g. 'column'.
            limit: Maximum number of matched objects to list.
        """
        material = (material or "").strip()
        semantic_label = _canonical_semantic_label(semantic_label)
        if not material:
            return "Provide a material to search for, e.g. material='legno'."

        annotations = getattr(ctx, "object_annotations", {})
        if not annotations:
            return (
                "No CSV annotations are matched to this scene. Material search "
                "uses CSV material fields only."
            )

        aliases = _material_aliases_for_query(material)
        hits = []
        inspected = 0
        for object_name in sorted(ctx.objects):
            obj = ctx.objects[object_name]
            if semantic_label and obj.get("semantic_label") != semantic_label:
                continue
            for annotation in annotations.get(object_name, []):
                material_value = _annotation_material_value(annotation)
                if not material_value:
                    continue
                inspected += 1
                if _text_contains_any(material_value, aliases):
                    hits.append((object_name, obj["semantic_label"], material_value, annotation))

        scope = f" for class {semantic_label}" if semantic_label else ""
        if not hits:
            return (
                f"No CSV material matches{scope} for {material!r}. "
                "Searched only material/materiale/material_description/"
                "descrizione_materica. I do not infer material from descriptions, "
                "RGB, roughness, or semantic class. "
                f"CSV material entries checked: {inspected}."
            )

        max_rows = max(1, min(int(limit), 200))
        lines = [
            f"CSV material matches{scope} for {material!r}: {len(hits)}",
            "Source: CSV material fields only.",
        ]
        for object_name, label, value, annotation in hits[:max_rows]:
            match = annotation.get("match", {})
            distance = match.get("distance_m")
            match_text = (
                f"; global_box_center distance={distance:.3f} m"
                if distance is not None
                else ""
            )
            lines.append(f"  - {object_name} ({label}): {value}{match_text}")
        if len(hits) > max_rows:
            lines.append(f"  ... {len(hits) - max_rows} more matches not shown.")
        return "\n".join(lines)

    @tool
    def get_object_semantic_details(
        semantic_label: Optional[str] = None,
        object_name: Optional[str] = None,
        semantic_labels: Optional[list[str]] = None,
    ) -> str:
        """Get CSV-derived material, typology, function, and description.

        This is the preferred tool for material, typology, function, and
        description when exact object ids are known or when the user asks for
        every object in one or more classes. For multi-class questions, pass
        semantic_labels to avoid repeated calls.

        For one exact object, provide object_name and semantic_label.
        For all objects in one class, provide semantic_label only.
        For all objects in multiple classes, provide semantic_labels only.
        The response
        also includes the object's centroid and box-center position, so
        material/typology/function/description can be correlated with where
        the object sits in the scene.

        Args:
            semantic_label: Optional semantic class, e.g. 'column'.
            object_name: Optional exact object id, e.g. 'column_2'.
            semantic_labels: Optional list of semantic classes to retrieve
                together.
        """
        target_names = _resolve_target_names(
            ctx,
            object_name,
            semantic_label,
            semantic_labels,
        )
        if isinstance(target_names, str):
            return target_names

        graph = ctx.scene_graph if ctx.scene_graph is not None else _combined_graph(ctx)
        blocks = []
        missing_labels = getattr(target_names, "missing_labels", [])
        if missing_labels:
            blocks.append(
                "No objects found for requested semantic_label(s): "
                + ", ".join(missing_labels)
                + "."
            )
        for name in target_names:
            obj = ctx.objects[name]
            node_data = graph.nodes.get(name, {}) if graph is not None else {}
            c = obj["centroid"]
            box_center = (obj["bounds"]["min"] + obj["bounds"]["max"]) / 2.0

            lines = [
                f"Semantic details for {name} ({obj['semantic_label']}):",
                f"  Centroid: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})",
                f"  Box center: ({box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f})",
            ]
            found = False
            for field in ("material", "typology", "function", "description"):
                value = node_data.get(field)
                if value:
                    found = True
                    lines.append(f"  {field}: {value}")
            if not found:
                lines.append(
                    "  No material/typology/function/description annotation for this object."
                )
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    return [
        get_object_annotation,
        list_csv_annotation_matches,
        find_objects_by_material,
        get_object_semantic_details,
    ]



def _annotation_first_value(annotation: dict, keys: tuple[str, ...]) -> object | None:
    for key in keys:
        value = annotation.get(key)
        if value:
            return value
    return None


def _annotation_match_text(annotation: dict) -> str:
    match = annotation.get("match", {})
    method = match.get("method", "unknown")
    distance = match.get("distance_m")
    if distance is None:
        reason = match.get("reason")
        return f"match={method}" + (f", reason={reason}" if reason else "")
    return f"match={method}, distance={distance:.3f} m"


def _annotation_material_value(annotation: dict) -> object | None:
    return _annotation_first_value(
        annotation,
        (
            "material_description",
            "descrizione_materica",
            "material",
            "materiale",
            "materica",
        ),
    )


def _annotation_resolution_error(ctx: SceneContext, semantic_label: Optional[str] = None) -> str:
    candidates = [
        name for name, obj in ctx.objects.items()
        if semantic_label is None or obj.get("semantic_label") == semantic_label
    ]
    if not candidates:
        return f"No objects found for semantic_label={semantic_label!r}."
    lines = [
        "Could not resolve one object. Provide a position such as central, left, right, north, south, top, or bottom.",
        "Candidates:",
    ]
    for name in candidates[:12]:
        c = ctx.objects[name]["centroid"]
        lines.append(f"  - {name}: centroid=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
    if len(candidates) > 12:
        lines.append(f"  ... {len(candidates) - 12} more candidates not shown.")
    return "\n".join(lines)


def _annotation_summary_text(annotation: dict) -> str:
    values = []
    for key in (
        "material_description",
        "descrizione_materica",
        "material",
        "materiale",
        "typology",
        "tipologia",
        "function",
        "funzione",
        "description",
        "descrizione",
        "position",
        "posizione",
    ):
        value = annotation.get(key)
        if value:
            values.append(f"{key}: {value}")
        if len(values) >= 4:
            break
    return "; ".join(values)


def _format_annotations_for_class(
    ctx: SceneContext,
    semantic_label: str,
    annotated: list[tuple[str, dict]],
) -> str:
    lines = [
        f"CSV annotations for class {semantic_label}: {len(annotated)} matched entries.",
        "Matching method should be global_box_center when the CSV provides global_box_center_x/y/z.",
    ]
    for object_name, annotation in annotated[:30]:
        box_center = _object_box_center_text(ctx, object_name)
        values = []
        for key in (
            "material_description",
            "descrizione_materica",
            "material",
            "materiale",
            "typology",
            "tipologia",
            "function",
            "funzione",
            "description",
            "descrizione",
        ):
            value = annotation.get(key)
            if value:
                values.append(f"{key}: {value}")
        match = annotation.get("match", {})
        distance = match.get("distance_m")
        match_text = match.get("method", "unknown")
        if distance is not None:
            match_text += f", distance={distance:.3f} m"
        lines.append(
            f"  - {object_name}: "
            + "; ".join(values + [f"box_center: {box_center}", f"match: {match_text}"])
        )
    if len(annotated) > 30:
        lines.append(f"  ... {len(annotated) - 30} annotations not shown.")
    return "\n".join(lines)


def _format_annotations_for_object(ctx: SceneContext, object_name: str, annotations: list[dict]) -> str:
    obj = ctx.objects[object_name]
    c = obj["centroid"]
    box_center = (obj["bounds"]["min"] + obj["bounds"]["max"]) / 2.0
    lines = [
        f"CSV annotations for {object_name} ({obj['semantic_label']}):",
        f"  Centroid: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})",
        f"  Box center: ({box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f})",
    ]
    for index, annotation in enumerate(annotations, start=1):
        lines.append(f"  Entry {index}:")
        for key in (
            "description",
            "descrizione",
            "historical_description",
            "descrizione_storica",
            "material_description",
            "descrizione_materica",
            "material",
            "materiale",
            "typology",
            "tipologia",
            "function",
            "funzione",
            "notes",
            "note",
            "position",
            "posizione",
        ):
            value = annotation.get(key)
            if value:
                lines.append(f"    {key}: {value}")
        match = annotation.get("match", {})
        method = match.get("method", "unknown")
        distance = match.get("distance_m")
        if distance is None:
            lines.append(f"    match: {method}")
        else:
            lines.append(f"    match: {method}, distance={distance:.3f} m")
        lines.append(f"    source row: {annotation.get('source_row')}")
    return "\n".join(lines)


def _material_aliases_for_query(material: str) -> tuple[str, ...]:
    normalized = _normalize_text(material)
    groups = (
        ("marmo", "marble"),
        ("calcare", "limestone"),
        ("pietra", "stone"),
        ("laterizio", "brick", "mattoni", "brick masonry"),
        ("intonaco", "plaster"),
        ("stucco",),
        ("legno", "wood"),
        ("metallo", "metal"),
        ("vetro", "glass"),
        ("terracotta", "cotto", "tile"),
    )
    for aliases in groups:
        if any(_normalize_text(alias) in normalized for alias in aliases):
            return aliases
    return (normalized,)


def _normalize_position_text(text: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", str(text).strip().lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _resolve_annotation_object(
    ctx: SceneContext,
    semantic_label: Optional[str] = None,
    position: Optional[str] = None,
) -> str | None:
    candidates = [
        name for name, obj in ctx.objects.items()
        if semantic_label is None or obj.get("semantic_label") == semantic_label
    ]
    if not candidates:
        return None
    if position:
        selected = _select_annotation_candidate_by_position(ctx, candidates, position)
        if selected:
            return selected
    if len(candidates) == 1:
        return candidates[0]
    annotated = [
        name for name in candidates
        if getattr(ctx, "object_annotations", {}).get(name)
    ]
    if len(annotated) == 1:
        return annotated[0]
    return None


def _select_annotation_candidate_by_position(ctx: SceneContext, candidates: list[str], position: str) -> str | None:
    normalized = _normalize_position_text(position)
    centroids = {name: ctx.objects[name]["centroid"] for name in candidates}
    if any(term in normalized for term in ("central", "center", "centro", "centrale", "middle")):
        mean = sum((centroids[name] for name in candidates)) / len(candidates)
        return min(candidates, key=lambda name: np_linalg_norm(centroids[name] - mean))
    if any(term in normalized for term in ("left", "sinistra", "west", "ovest")):
        return min(candidates, key=lambda name: float(centroids[name][0]))
    if any(term in normalized for term in ("right", "destra", "east", "est")):
        return max(candidates, key=lambda name: float(centroids[name][0]))
    if any(term in normalized for term in ("south", "sud", "front", "davanti")):
        return min(candidates, key=lambda name: float(centroids[name][1]))
    if any(term in normalized for term in ("north", "nord", "back", "dietro")):
        return max(candidates, key=lambda name: float(centroids[name][1]))
    if any(term in normalized for term in ("bottom", "lower", "basso", "bassa", "inferiore")):
        return min(candidates, key=lambda name: float(centroids[name][2]))
    if any(term in normalized for term in ("top", "upper", "alto", "alta", "superiore")):
        return max(candidates, key=lambda name: float(centroids[name][2]))
    return None


def _text_contains_any(value: object, aliases: tuple[str, ...]) -> bool:
    normalized_value = _normalize_text(str(value))
    return any(_normalize_text(alias) in normalized_value for alias in aliases)
