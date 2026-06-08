from collections import Counter
from typing import Optional

import networkx as nx
from langchain_core.tools import tool

from ..pipeline.pipeline import SceneContext, run_pipeline
from ..pipeline.graph import analyze_scene_graph
from ..settings import get_config


def _structural_elements() -> set[str]:
    return set(get_config()["semantic_classes"]["structural"])


def _classify_area(label_set: set) -> str:
    structural = _structural_elements()
    if {"vault", "arch", "column"} & label_set:
        return "vaulted_space"
    if "stairs" in label_set:
        return "vertical_circulation"
    if {"door_window", "wall"} & label_set:
        return "facade_zone"
    if {"floor", "wall"} & label_set:
        return "floor_zone"
    if label_set <= structural:
        return "structural_zone"
    return "general_area"


def create_scene_tools(ctx: SceneContext) -> list:

    @tool
    def list_objects() -> str:
        """List all objects detected in the scene, grouped by semantic class."""
        if not ctx.objects:
            return "No objects in the scene."
        by_class: dict[str, list] = {}
        for name, obj in ctx.objects.items():
            by_class.setdefault(obj["semantic_label"], []).append(
                (name, obj["point_count"])
            )
        lines = [f"Scene contains {len(ctx.objects)} objects:\n"]
        for lbl in sorted(by_class):
            lines.append(f"  {lbl.upper()} ({len(by_class[lbl])} instances):")
            for name, count in sorted(by_class[lbl]):
                lines.append(f"    - {name}: {count:,} points")
        return "\n".join(lines)

    @tool
    def get_object_info(object_name: str) -> str:
        """Get detailed geometric and semantic information about a specific object.

        Args:
            object_name: Name of the object, e.g. 'wall_0' or 'column_2'.
        """
        if object_name not in ctx.objects:
            sample = ", ".join(list(ctx.objects.keys())[:8])
            return f"Object '{object_name}' not found. Examples: {sample}"

        obj = ctx.objects[object_name]
        feat = ctx.features.get(object_name, {})
        c = obj["centroid"]
        dims = obj["bounds"]["max"] - obj["bounds"]["min"]

        return "\n".join([
            f"Object: {object_name}",
            f"  Semantic class  : {obj['semantic_label']}",
            f"  Element type    : {feat.get('element_type', 'unknown')}",
            f"  Point count     : {obj['point_count']:,}",
            f"  Centroid (x,y,z): ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})",
            f"  Dimensions (m)  : {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f}",
            f"  Volume (AABB)   : {feat.get('volume', 0):.3f} m3",
            f"  Surface area    : {feat.get('surface_area', 0):.3f} m2",
            f"  Height          : {feat.get('height', 0):.2f} m",
            f"  Compactness     : {feat.get('compactness', 0):.4f}",
        ])

    @tool
    def find_relationships(object_name: str) -> str:
        """Find all spatial relationships involving a given object.

        Args:
            object_name: Name of the object to query.
        """
        G = ctx.scene_graph
        if object_name not in G:
            return f"Object '{object_name}' not found in the scene graph."

        out_edges = list(G.out_edges(object_name, data=True))
        in_edges = list(G.in_edges(object_name, data=True))
        lines = [f"Relationships for '{object_name}':"]

        if out_edges:
            lines.append("  As source:")
            for _, tgt, d in out_edges:
                for rel in d.get("relations", []):
                    lines.append(f"    {object_name} --[{rel['level']}:{rel['type']}]--> {tgt}")
        if in_edges:
            lines.append("  As target:")