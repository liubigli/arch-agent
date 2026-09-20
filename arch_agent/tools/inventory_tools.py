"""Tools that enumerate the scene: semantic classes, object counts and object lists."""


from collections import Counter
from typing import Optional

from langchain_core.tools import tool

from ..pipeline.pipeline import SceneContext

from ._shared import (
    _all_semantic_classes,
    _canonical_semantic_label,
    _canonical_semantic_label_list,
    _scene_class_names,
)


def create_inventory_tools(ctx: SceneContext) -> list:
    """Build the inventory tools bound to ``ctx``."""

    _class_names = _scene_class_names(ctx)

    @tool
    def list_semantic_labels(semantic_labels: Optional[list[str]] = None) -> str:
        """List semantic labels/classes present and absent in this scene graph.

        Use this to discover the valid semantic_label values for this scene
        before calling other tools that accept a semantic_label parameter.
        Use this for present/absent class questions. For object counts across
        multiple classes, prefer count_objects_by_class instead.

        Args:
            semantic_labels: Optional semantic classes or aliases to check
                explicitly, e.g. ['colonna', 'tetto'].
        """
        counts = Counter(obj["semantic_label"] for obj in ctx.objects.values())
        if not counts:
            return "No semantic labels found in the scene graph."

        if semantic_labels:
            requested_labels = _canonical_semantic_label_list(semantic_labels)
            lines = [
                "REQUESTED CLASS STATUS",
                f"- Semantic classes present in the scene: {len(_class_names)}",
                "- Requested classes after alias normalization:",
            ]
            for label in requested_labels:
                count = counts.get(label, 0)
                status = "present" if count else "absent"
                lines.append(f"  - {label}: {status} ({count} object(s))")
            absent_requested = [
                label for label in requested_labels
                if counts.get(label, 0) == 0
            ]
            if absent_requested:
                lines.extend([
                    "",
                    "SCENE EVIDENCE",
                    "- These requested classes are absent from the extracted scene objects: "
                    + ", ".join(absent_requested)
                    + ".",
                    "",
                    "TOOL CONCLUSION",
                    "- Questions requiring an absent class have no supporting scene evidence for that class.",
                ])
            else:
                lines.extend([
                    "",
                    "TOOL CONCLUSION",
                    "- All requested classes are present in the extracted scene objects.",
                ])
            return "\n".join(lines)

        expected_labels = [
            label
            for label in _all_semantic_classes()
            if label != "other" or counts.get(label, 0) > 0
        ]
        absent_labels = [
            label for label in expected_labels
            if counts.get(label, 0) == 0
        ]
        lines = [f"Semantic labels present in the scene graph: {len(_class_names)}"]
        lines.extend(
            f"  - {label}: {counts.get(label, 0)} instance(s)"
            for label in _class_names
        )
        if absent_labels:
            lines.append("Semantic labels absent in this scene:")
            lines.extend(f"  - {label}: 0 instance(s)" for label in absent_labels)
        else:
            lines.append("Semantic labels absent in this scene: none.")
        return "\n".join(lines)

    @tool
    def count_objects(semantic_label: Optional[str] = None) -> str:
        """Count detected objects, optionally filtered by semantic class.

        Use this for a single-class count, e.g. "quante colonne ci sono?",
        or for the total object count when semantic_label is omitted.
        Do not use this repeatedly for multi-class count questions; prefer
        count_objects_by_class for grouped counts or object distribution.

        Args:
            semantic_label: Optional semantic class to count, e.g. 'wall',
                'column', or 'floor'. If omitted, returns the total object count.
        """
        semantic_label = _canonical_semantic_label(semantic_label)
        if semantic_label:
            count = sum(
                1 for obj in ctx.objects.values()
                if obj["semantic_label"] == semantic_label
            )
            if count == 0:
                return (
                    f"DIRECT COUNT ANSWER\n"
                    f"- {semantic_label}: 0 object(s)\n\n"
                    "REQUESTED CLASS STATUS\n"
                    f"- {semantic_label}: absent (0 object(s))\n\n"
                    "SCENE EVIDENCE\n"
                    f"- No objects with semantic label '{semantic_label}' were extracted from this scene.\n\n"
                    "TOOL CONCLUSION\n"
                    f"- Count for '{semantic_label}' is 0."
                )
            return f"Objects with semantic label '{semantic_label}': {count}"

        return f"Total detected objects: {len(ctx.objects)}"

    @tool
    def count_objects_by_class(semantic_labels: Optional[list[str]] = None) -> str:
        """Count detected objects grouped by semantic class.

        Use this tool when the user asks for:
        - object distribution by class;
        - counts for requested semantic classes;
        - counts for multiple semantic classes;
        - counts grouped by class;
        - a complete class inventory with counts;
        - counts of column, wall, floor, roof, vault, arch, stairs,
          door_window, moldings together.

        Prefer this tool over multiple count_objects calls for multi-class
        count questions. It reports expected semantic classes with count 0
        when they are absent, so absent classes are explicit.
        If the user asks for all classes, all objects by class, the complete
        inventory, or the full class distribution without naming specific
        classes, omit semantic_labels. Do not invent or copy example labels.

        Args:
            semantic_labels: Optional list of semantic classes to count. If
                omitted, returns counts for all expected semantic classes
                except absent 'other'. Pass this only when the user explicitly
                names the requested classes.
        """
        counts = Counter(obj["semantic_label"] for obj in ctx.objects.values())
        if not counts:
            return "No objects in the scene."

        if semantic_labels:
            expected_labels = _canonical_semantic_label_list(semantic_labels)
            title = "Objects per requested class:"
        else:
            expected_labels = [
                label
                for label in _all_semantic_classes()
                if label != "other" or counts.get(label, 0) > 0
            ]
            title = "Objects per class:"

        lines = [
            f"Total detected objects: {len(ctx.objects)}",
            f"Semantic classes present: {len(counts)}",
            title,
        ]
        lines.extend(
            f"  - {label}: {counts.get(label, 0)}"
            for label in expected_labels
        )
        absent_requested = [
            label for label in expected_labels
            if counts.get(label, 0) == 0
        ]
        if absent_requested:
            lines.extend([
                "",
                "REQUESTED CLASS STATUS",
                "- Absent/zero-count classes: " + ", ".join(absent_requested) + ".",
                "",
                "TOOL CONCLUSION",
                "- A zero count means the class is not present among extracted scene objects.",
            ])
        return "\n".join(lines)

    @tool
    def list_objects(semantic_labels: Optional[list[str]] = None) -> str:
        """List detected objects, grouped by semantic class.

        Use this when the user asks which exact objects are present, object ids
        by class, or a full object inventory. If the user names multiple
        semantic classes, pass them together as semantic_labels to avoid
        repeated tool calls. For counts only, prefer count_objects or
        count_objects_by_class.
        If the user asks for all objects, every object, or a complete object
        inventory without naming specific classes, omit semantic_labels. Do
        not invent or copy example labels.

        Args:
            semantic_labels: Optional list of semantic classes to list. If
                omitted, all detected objects are listed. Pass this only when
                the user explicitly names requested classes.
        """
        if not ctx.objects:
            return "No objects in the scene."
        requested_labels = _canonical_semantic_label_list(semantic_labels or [])
        requested_set = set(requested_labels)
        by_class: dict[str, list] = {}
        for name, obj in ctx.objects.items():
            if requested_set and obj["semantic_label"] not in requested_set:
                continue
            by_class.setdefault(obj["semantic_label"], []).append(
                (name, obj["point_count"])
            )
        listed_count = sum(len(items) for items in by_class.values())
        if requested_labels:
            lines = [
                f"Scene contains {listed_count} object(s) in requested classes: "
                f"{', '.join(requested_labels)}.\n"
            ]
            labels_to_show = requested_labels
        else:
            lines = [f"Scene contains {len(ctx.objects)} objects:\n"]
            labels_to_show = sorted(by_class)
        for lbl in labels_to_show:
            if lbl not in by_class:
                lines.append(f"  {lbl.upper()} (0 instances):")
                continue
            lines.append(f"  {lbl.upper()} ({len(by_class[lbl])} instances):")
            for name, count in sorted(by_class[lbl]):
                lines.append(f"    - {name}: {count:,} points")
        return "\n".join(lines)

    return [
        list_semantic_labels,
        count_objects,
        count_objects_by_class,
        list_objects,
    ]
