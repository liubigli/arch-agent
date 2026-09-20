"""Tools that query the spatial graph: relationships between objects and their
derivation level."""


from collections import Counter
from typing import Optional

from langchain_core.tools import tool

from ..pipeline.pipeline import SceneContext
from ..pipeline.relationships import RELATIONSHIP_LAYER_NAMES
from ..pipeline.relationships import RELATIONSHIP_LAYER_ORDER
from ..pipeline.relationships import compute_csv_annotation_relationships

from ._shared import (
    _canonical_semantic_label,
    _canonical_semantic_label_list,
    _clean_optional,
    _object_not_found_message,
    _objects_with_semantic_label,
    _resolve_target_names,
)


def create_relationship_tools(ctx: SceneContext) -> list:
    """Build the relationship tools bound to ``ctx``."""

    @tool
    def find_relationships(
        object_name: Optional[str] = None,
        semantic_label: Optional[str] = None,
        semantic_labels: Optional[list[str]] = None,
        limit: int = 40,
        offset: int = 0,
    ) -> str:
        """Find all relationships/evidence involving object(s).

        Use this for relationship questions focused on one object, one
        semantic class, e.g. "che relazioni hanno le colonne?", "what
        relationships involve wall?", or "cosa supportano le colonne?".
        If the user names multiple semantic classes, pass all of them together
        as semantic_labels, especially for support questions such as
        "roof supported by columns" -> ['roof', 'column']. Never drop one side
        of the relation. For global relationship inventories not focused on one
        class/object,
        use list_relationships.

        Provide exactly one of these:
        - object_name: one exact object id (e.g. 'column_2') for a single instance.
        - semantic_label: a semantic class (e.g. 'column') to aggregate the
          relationships of every instance of that class.
        - semantic_labels: multiple semantic classes to query together. Use
          this whenever the user names two or more classes in the same
          relationship question.

        Results are paginated to avoid flooding the chat with a large scene's
        full spatial graph. If the response says rows were not shown,
        call again with the suggested offset to see the next batch; rows are
        always returned in the same order for the same object(s), so no row
        is skipped or repeated across calls.

        Args:
            object_name: Exact object id for a single instance.
            semantic_label: Semantic class to query across all its instances.
            semantic_labels: Optional list of semantic classes to query
                together. Include every class named in the user question.
            limit: Maximum number of relationship rows to return in this call.
            offset: Number of relationship rows to skip before collecting up
                to `limit` rows, to page through a result already seen.
        """
        object_name = _clean_optional(object_name)
        semantic_label = _canonical_semantic_label(semantic_label)
        target_names = _resolve_target_names(
            ctx,
            object_name,
            semantic_label,
            semantic_labels,
        )
        if isinstance(target_names, str):
            return target_names
        target_set = set(target_names)

        max_rows = max(1, min(int(limit), 200))
        skip = max(0, int(offset))

        lines = [
            f"Relationships/evidence for {len(target_names)} object(s): {', '.join(target_names)}",
            "Cascade: spatial graph -> CSV/user metadata -> CIDOC/KG. CSV detail is not a graph.",
        ]
        missing_labels = getattr(target_names, "missing_labels", [])
        if missing_labels:
            present_labels = sorted({
                ctx.objects[name]["semantic_label"]
                for name in target_names
            })
            lines = [
                "REQUESTED CLASS STATUS",
            ]
            for label in missing_labels:
                lines.append(f"- {label}: absent (0 object(s))")
            if present_labels:
                for label in present_labels:
                    count = sum(
                        1 for obj in ctx.objects.values()
                        if obj["semantic_label"] == label
                    )
                    lines.append(f"- {label}: present ({count} object(s))")
            lines.extend([
                "",
                "SCENE EVIDENCE",
                "- No objects found for requested semantic_label(s): "
                + ", ".join(missing_labels)
                + ".",
                "- Relationships involving an absent class cannot be observed in this scene.",
                "- Relationships of present classes alone are not evidence for an absent class.",
                "",
                "TOOL CONCLUSION",
                "- No scene relationship can be reported for the full requested class set because at least one requested class is absent.",
            ])
            if present_labels:
                lines.append(
                    "- To inspect the present class by itself, call the tool again with only: "
                    + ", ".join(present_labels)
                    + "."
                )
            return "\n".join(lines)
        total = 0
        shown = 0
        remaining = max_rows

        for level, relationships in _relationship_layers_with_csv_evidence(ctx):
            seen = set()
            filtered = []
            for rel in relationships:
                if (rel[0] in target_set or rel[1] in target_set) and rel not in seen:
                    seen.add(rel)
                    filtered.append(rel)
            total += len(filtered)
            lines.append(f"  {_relationship_layer_display_name(level)}: {len(filtered)}")
            for src, tgt, rel_type, rel_level in filtered:
                if skip > 0:
                    skip -= 1
                    continue
                if remaining <= 0:
                    continue
                lines.append(f"    {src} --[{rel_level}:{rel_type}]--> {tgt}")
                remaining -= 1
                shown += 1

        if total == 0:
            lines.append("  No relationships found.")
        elif offset >= total:
            lines.append(
                f"  offset={offset} is beyond the last row; "
                f"this object set has {total} relationship(s) total."
            )
        else:
            hidden = total - offset - shown
            if hidden > 0:
                next_offset = offset + shown
                lines.append(
                    f"  Showing rows {offset + 1}-{offset + shown} of {total}; "
                    f"{hidden} more not shown. Call again with offset={next_offset} "
                    "to see the next batch."
                )

        return "\n".join(lines)

    @tool
    def list_relationships(
        level: str = "all",
        relationship_type: Optional[str] = None,
        object_name: Optional[str] = None,
        semantic_label: Optional[str] = None,
        semantic_labels: Optional[list[str]] = None,
        limit: int = 30,
        offset: int = 0,
    ) -> str:
        """List relationships from the scene graph.

        Use this for global relationship inventories, relationship-type
        summaries, or questions not focused on a single object/class. For
        relationships involving one specific object or class, prefer
        find_relationships.

        object_name, semantic_label, and semantic_labels are optional and
        exclusive: use object_name for one exact instance (e.g. 'column_2'),
        or semantic_label to match relationships for every instance of a
        class (e.g. 'column'), or semantic_labels to match multiple classes
        in one call. For any relationship/support question naming two or more
        classes, semantic_labels must include every class named by the user,
        even if one class may be absent. Omit all target filters to list all
        relationships.

        Results are paginated: if the response says rows were not shown,
        call again with the suggested offset to see the next batch; rows are
        always returned in the same order for the same filters, so no row is
        skipped or repeated across calls.

        Args:
            level: Relationship/evidence selector to list: 'spatial',
                'geometric', 'structural_evidence', 'structural', 'csv',
                'cidoc', 'kg', or 'all'. Legacy aliases are still accepted
                for compatibility. 'structural' filters support/appoggio
                relations. 'csv' returns CSV detail metadata. 'cidoc' returns
                CIDOC/KG metadata summary.
            relationship_type: Optional relationship type, e.g. 'above',
                'supports', 'is_opening_in'.
            object_name: Optional exact object id. If provided, only
                relationships where this object is source or target are listed.
            semantic_label: Optional semantic class. If provided, only
                relationships where an instance of this class is source or
                target are listed.
            semantic_labels: Optional list of semantic classes. If provided,
                only relationships where an instance of any requested class is
                source or target are listed. Include every class named in a
                multi-class relationship question.
            limit: Maximum number of relationship rows to return in this
                call. Default is intentionally small to avoid flooding the chat.
            offset: Number of relationship rows to skip before collecting up
                to `limit` rows, to page through a result already seen.
        """
        layer_key = _relationship_layer_key(level)
        if layer_key is None:
            valid = "all, spatial graph, structural support filter, CSV detail, CIDOC/KG"
            return f"Unknown relationship level '{level}'. Valid values: {valid}."
        object_name = _clean_optional(object_name)
        semantic_label = _canonical_semantic_label(semantic_label)
        requested_labels = _canonical_semantic_label_list(semantic_labels or [])
        target_filter_count = sum(
            1 for value in (object_name, semantic_label, requested_labels) if value
        )
        if target_filter_count > 1:
            return "Provide only one of object_name, semantic_label, or semantic_labels."
        if object_name and object_name not in ctx.objects:
            object_as_label = _canonical_semantic_label(object_name)
            if object_as_label and _objects_with_semantic_label(ctx, object_as_label):
                semantic_label = object_as_label
                object_name = None
        if layer_key == "L2_DETAIL":
            return _l2_detail_tool_summary(ctx)
        if layer_key == "L3_KG_DETAIL":
            return _l3_kg_tool_summary(ctx)
        if object_name and object_name not in ctx.objects:
            return _object_not_found_message(object_name, ctx.objects)

        target_set = None
        if requested_labels:
            present_requested_labels = sorted({
                obj["semantic_label"]
                for obj in ctx.objects.values()
                if obj["semantic_label"] in set(requested_labels)
            })
            missing_labels = [
                label for label in requested_labels
                if label not in set(present_requested_labels)
            ]
            if missing_labels:
                present_counts = Counter(obj["semantic_label"] for obj in ctx.objects.values())
                lines = [
                    "REQUESTED CLASS STATUS",
                ]
                for label in missing_labels:
                    lines.append(f"- {label}: absent (0 object(s))")
                if present_requested_labels:
                    for label in present_requested_labels:
                        lines.append(
                            f"- {label}: present ({present_counts.get(label, 0)} object(s))"
                        )
                lines.extend([
                    "",
                    "SCENE EVIDENCE",
                    "- No objects found for requested semantic_label(s): "
                    + ", ".join(missing_labels)
                    + ".",
                    "- Relationships involving an absent class cannot be observed in this scene.",
                    "- Relationships of present classes alone are not evidence for an absent class.",
                    "",
                    "TOOL CONCLUSION",
                    "- No scene relationship can be reported for the full requested class set because at least one requested class is absent.",
                ])
                if present_requested_labels:
                    lines.append(
                        "- To inspect the present class by itself, call the tool again with only: "
                        + ", ".join(present_requested_labels)
                        + "."
                    )
                return "\n".join(lines)
            target_set = {
                name for name, obj in ctx.objects.items()
                if obj["semantic_label"] in set(requested_labels)
            }
        elif semantic_label:
            target_set = {
                name for name, obj in ctx.objects.items()
                if obj["semantic_label"] == semantic_label
            }

        layers = (
            _relationship_layers_with_csv_evidence(ctx)
            if layer_key == "all"
            else [(layer_key, _relationships_for_layer(ctx, layer_key))]
        )
        filtered_by_layer = [
            (
                layer,
                [
                    rel for rel in relationships
                    if (not relationship_type or rel[2] == relationship_type)
                    and (not object_name or rel[0] == object_name or rel[1] == object_name)
                    and (target_set is None or rel[0] in target_set or rel[1] in target_set)
                ],
            )
            for layer, relationships in layers
        ]
        filtered = [
            rel
            for _, layer_relationships in filtered_by_layer
            for rel in layer_relationships
        ]

        title = f"Relationships ({_relationship_layer_display_name(layer_key)}): {len(filtered)}"
        if layer_key == "all":
            title += " | cascade=spatial graph->CSV/user metadata->CIDOC/KG"
        if relationship_type:
            title += f" | type={relationship_type}"
        if object_name:
            title += f" | object={object_name}"
        if semantic_label:
            title += f" | class={semantic_label}"
        if requested_labels:
            title += f" | classes={', '.join(requested_labels)}"

        type_counts = Counter(rel[2] for rel in filtered)
        if type_counts:
            title += " | " + ", ".join(
                f"{rel_type}={count}" for rel_type, count in sorted(type_counts.items())
            )

        max_rows = max(1, min(int(limit), 200))
        skip = max(0, int(offset))
        lines = [title]
        remaining = max_rows
        shown = 0
        for layer, layer_relationships in filtered_by_layer:
            lines.append(f"  {_relationship_layer_display_name(layer)}: {len(layer_relationships)}")
            for src, tgt, rel_type, rel_level in layer_relationships:
                if skip > 0:
                    skip -= 1
                    continue
                if remaining <= 0:
                    continue
                lines.append(f"    - {src} --[{rel_level}:{rel_type}]--> {tgt}")
                remaining -= 1
                shown += 1

        total = len(filtered)
        if total == 0:
            lines.append("  No matching relationships found.")
        elif offset >= total:
            lines.append(
                f"  offset={offset} is beyond the last row; "
                f"this filter has {total} relationship(s) total."
            )
        else:
            hidden = total - offset - shown
            if hidden > 0:
                next_offset = offset + shown
                lines.append(
                    f"  Showing rows {offset + 1}-{offset + shown} of {total}; "
                    f"{hidden} more not shown. Call again with offset={next_offset} "
                    "to see the next batch."
                )
        return "\n".join(lines)

    @tool
    def find_relationship_anomalies(limit: int = 200) -> str:
        """Find direct logical or semantic anomalies in the computed spatial graph.

        Args:
            limit: Maximum number of anomaly rows to return.
        """
        issues = _relationship_anomalies(ctx)
        max_rows = max(1, min(int(limit), 1000))
        if not issues:
            return (
                "No direct relationship anomalies found "
                "(reciprocal above/below/support loops, invalid contains/inside, or invalid openings)."
            )

        lines = [f"Relationship anomalies: {len(issues)}"]
        lines.extend(f"  - {issue}" for issue in issues[:max_rows])
        if len(issues) > max_rows:
            lines.append(f"  ... {len(issues) - max_rows} more anomalies not shown.")
        return "\n".join(lines)

    return [
        find_relationships,
        list_relationships,
        find_relationship_anomalies,
    ]



def _l2_detail_tool_summary(ctx: SceneContext) -> str:
    matched = sum(
        len(entries)
        for entries in getattr(ctx, "object_annotations", {}).values()
    )
    annotated_objects = len(getattr(ctx, "object_annotations", {}))
    unmatched = len(getattr(ctx, "unmatched_annotations", []))
    return (
        "CSV detail is descriptive metadata, not a graph. "
        f"Matched annotations: {matched} on {annotated_objects} objects; "
        f"unmatched CSV rows: {unmatched}. "
        "Use get_object_annotation for scene/object descriptions, material, "
        "typology, function, and notes."
    )


def _l3_kg_tool_summary(ctx: SceneContext) -> str:
    matched = sum(
        len(entries)
        for entries in getattr(ctx, "object_annotations", {}).values()
    )
    annotated_objects = len(getattr(ctx, "object_annotations", {}))
    return (
        "CIDOC/KG is the semantic knowledge graph. "
        f"It can be built from CSV/user metadata: {matched} matched annotations "
        f"on {annotated_objects} objects. Use the CIDOC/KG viewer/export tools for semantic edges."
    )


def _relationship_anomalies(ctx: SceneContext) -> list[str]:
    pair_relations: dict[frozenset[str], list[tuple[str, str, str, str]]] = {}
    for src, tgt, rel_type, rel_level in ctx.relationships:
        pair_relations.setdefault(frozenset((src, tgt)), []).append(
            (src, tgt, rel_type, rel_level)
        )

    issues: list[str] = []
    for pair, rels in pair_relations.items():
        if len(pair) != 2:
            continue
        a, b = list(pair)
        rel_set = {(src, tgt, rel_type, rel_level) for src, tgt, rel_type, rel_level in rels}
        rel_type_set = {(src, tgt, rel_type) for src, tgt, rel_type, _ in rels}

        if (
            (a, b, "above", "geometric") in rel_set
            and (b, a, "above", "geometric") in rel_set
        ):
            issues.append(f"{a} and {b}: reciprocal 'above' relation.")
        if (
            (a, b, "below", "geometric") in rel_set
            and (b, a, "below", "geometric") in rel_set
        ):
            issues.append(f"{a} and {b}: reciprocal 'below' relation.")
        if (
            (a, b, "supports") in rel_type_set
            and (b, a, "supports") in rel_type_set
        ):
            issues.append(f"{a} and {b}: reciprocal 'supports' relation.")
        if (
            (a, b, "rests_on") in rel_type_set
            and (b, a, "rests_on") in rel_type_set
        ):
            issues.append(f"{a} and {b}: reciprocal 'rests_on' relation.")

        for src, tgt, rel_type, rel_level in rels:
            src_label = ctx.objects.get(src, {}).get("semantic_label")
            tgt_label = ctx.objects.get(tgt, {}).get("semantic_label")
            if rel_type in {"contains", "inside"}:
                issues.append(f"{src} -> {tgt}: unsupported relation '{rel_type}'.")
            if rel_type == "is_opening_in" and not (src_label == "door_window" and tgt_label == "wall"):
                issues.append(
                    f"{src} -> {tgt}: invalid is_opening_in for {src_label}->{tgt_label}."
                )
            if rel_type == "has_part" and src_label == "floor" and tgt_label == "door_window":
                issues.append(f"{src} -> {tgt}: floor should not contain a door_window.")

    return issues


def _relationship_layer_display_name(layer_key: str) -> str:
    if layer_key == "all":
        return "spatial graph"
    if layer_key == "L1":
        return "spatial graph"
    if layer_key == "structural_evidence":
        return "support view"
    if layer_key == "L2_DETAIL":
        return "CSV detail"
    if layer_key == "L3_KG_DETAIL":
        return "CIDOC/KG"
    return RELATIONSHIP_LAYER_NAMES.get(layer_key, layer_key)


def _relationship_layer_key(level: str) -> str | None:
    normalized = (level or "all").strip().lower()
    aliases = {
        "all": "all",
        "l1": "L1",
        "spatial": "L1",
        "spatial_graph": "L1",
        "geometric": "L1",
        "geometry": "L1",
        "l2": "L2_DETAIL",
        "csv": "L2_DETAIL",
        "metadata": "L2_DETAIL",
        "detail": "L2_DETAIL",
        "details": "L2_DETAIL",
        "structural": "structural_evidence",
        "structure": "structural_evidence",
        "structural_evidence": "structural_evidence",
        "support_evidence": "structural_evidence",
        "l3": "L3_KG_DETAIL",
        "cidoc": "L3_KG_DETAIL",
        "kg": "L3_KG_DETAIL",
        "knowledge_graph": "L3_KG_DETAIL",
        "mereological": "L3_KG_DETAIL",
        "composition": "L3_KG_DETAIL",
    }
    return aliases.get(normalized)


def _relationship_layers_in_order(ctx: SceneContext) -> list[tuple[str, list]]:
    if not ctx.relationship_layers:
        return [("all", ctx.relationships)]
    return [
        (level, ctx.relationship_layers.get(level, []))
        for level in RELATIONSHIP_LAYER_ORDER
    ]


def _relationship_layers_with_csv_evidence(ctx: SceneContext) -> list[tuple[str, list]]:
    return _relationship_layers_in_order(ctx)


def _relationships_for_layer(ctx: SceneContext, layer_key: str) -> list:
    if layer_key == "structural_evidence":
        return _structural_evidence_relationships(ctx)
    if layer_key == "L3_KG_DETAIL":
        return []
    return ctx.relationship_layers.get(layer_key, [])


def _structural_evidence_relationships(ctx: SceneContext) -> list[tuple[str, str, str, str]]:
    """Return supports/rests_on relations from the spatial graph."""
    relationships: list[tuple[str, str, str, str]] = []
    seen = set()

    for rel in ctx.relationship_layers.get("L1", []) + list(ctx.relationships or []):
        if len(rel) >= 3 and rel[2] in {"supports", "rests_on"} and rel not in seen:
            relationships.append(rel)
            seen.add(rel)

    for rel in compute_csv_annotation_relationships(
        ctx.objects,
        getattr(ctx, "object_annotations", {}),
    ):
        if len(rel) >= 3 and rel[2] in {"supports", "rests_on"} and rel not in seen:
            relationships.append(rel)
            seen.add(rel)

    return relationships
