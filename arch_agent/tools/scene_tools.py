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
            for src, _, d in in_edges:
                for rel in d.get("relations", []):
                    lines.append(f"    {src} --[{rel['level']}:{rel['type']}]--> {object_name}")

        if len(lines) == 1:
            lines.append("  No relationships found.")

        return "\n".join(lines)

    @tool
    def get_scene_statistics() -> str:
        """Get summary statistics for the current scene graph."""
        graphs = ctx.scene_graphs or {"L1": ctx.scene_graph}
        lines = [
            f"Objects: {len(ctx.objects)}",
            f"Relationships: {len(ctx.relationships)}",
        ]

        class_counts = Counter(obj["semantic_label"] for obj in ctx.objects.values())
        if class_counts:
            lines.append("Semantic classes:")
            for label, count in sorted(class_counts.items()):
                lines.append(f"  - {label}: {count}")

        lines.append("Graphs:")
        for level, graph in graphs.items():
            analysis = analyze_scene_graph(graph)
            lines.append(
                f"  - {level}: {analysis['node_count']} nodes, "
                f"{analysis['edge_count']} edges, avg degree {analysis['avg_degree']:.2f}"
            )

        return "\n".join(lines)

    @tool
    def find_focal_points(limit: int = 5) -> str:
        """Find the most central objects in the scene graph.

        Args:
            limit: Maximum number of objects to return.
        """
        graph = _combined_graph(ctx)
        if graph.number_of_nodes() == 0:
            return "No objects in the scene graph."

        scores = nx.degree_centrality(graph)
        top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        lines = ["Most central objects:"]
        for name, score in top:
            label = ctx.objects.get(name, {}).get("semantic_label", "unknown")
            lines.append(f"  - {name} ({label}): {score:.3f}")
        return "\n".join(lines)

    @tool
    def find_pattern(
        semantic_label: Optional[str] = None,
        relationship_type: Optional[str] = None,
    ) -> str:
        """Find objects or relationships matching a semantic label or relationship type.

        Args:
            semantic_label: Optional semantic class to search, e.g. 'wall'.
            relationship_type: Optional relationship type to search, e.g. 'supports'.
        """
        lines = []

        if semantic_label:
            matches = [
                name for name, obj in ctx.objects.items()
                if obj["semantic_label"] == semantic_label
            ]
            lines.append(f"Objects with semantic label '{semantic_label}': {len(matches)}")
            lines.extend(f"  - {name}" for name in matches[:50])

        if relationship_type:
            matches = [
                rel for rel in ctx.relationships
                if len(rel) >= 3 and rel[2] == relationship_type
            ]
            lines.append(f"Relationships of type '{relationship_type}': {len(matches)}")
            for src, tgt, rel_type, level in matches[:50]:
                lines.append(f"  - {src} --[{level}:{rel_type}]--> {tgt}")

        if not lines:
            return "Provide a semantic_label, a relationship_type, or both."
        return "\n".join(lines)

    @tool
    def discover_functional_areas() -> str:
        """Group connected scene components into coarse functional areas."""
        graph = _combined_graph(ctx)
        if graph.number_of_nodes() == 0:
            return "No functional areas found."

        lines = ["Functional areas:"]
        for idx, component in enumerate(nx.weakly_connected_components(graph), start=1):
            labels = {
                ctx.objects.get(name, {}).get("semantic_label", "unknown")
                for name in component
            }
            area_type = _classify_area(labels)
            lines.append(
                f"  - area_{idx}: {area_type} | "
                f"{len(component)} objects | classes: {', '.join(sorted(labels))}"
            )
        return "\n".join(lines)

    @tool
    def reload_scene(
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        sample_n: Optional[int] = None,
    ) -> str:
        """Reload the current scene with updated pipeline parameters.

        Args:
            eps: Optional DBSCAN epsilon.
            min_samples: Optional DBSCAN min_samples.
            distance_threshold: Optional relationship distance threshold.
            sample_n: Optional maximum number of points to load.
        """
        params = ctx.params
        if eps is not None:
            params.eps = eps
        if min_samples is not None:
            params.min_samples = min_samples
        if distance_threshold is not None:
            params.distance_threshold = distance_threshold
        if sample_n is not None:
            params.sample_n = sample_n

        new_ctx = run_pipeline(params)
        ctx.df = new_ctx.df
        ctx.objects = new_ctx.objects
        ctx.features = new_ctx.features
        ctx.relationships = new_ctx.relationships
        ctx.relationship_layers = new_ctx.relationship_layers
        ctx.scene_graph = new_ctx.scene_graph
        ctx.scene_graphs = new_ctx.scene_graphs

        return (
            "Scene reloaded. "
            f"Objects: {len(ctx.objects)} | Relationships: {len(ctx.relationships)}"
        )

    return [
        list_objects,
        get_object_info,
        find_relationships,
        get_scene_statistics,
        find_focal_points,
        find_pattern,
        discover_functional_areas,
        reload_scene,
    ]


def _combined_graph(ctx: SceneContext) -> nx.DiGraph:
    graphs = list((ctx.scene_graphs or {}).values())
    if not graphs:
        return ctx.scene_graph or nx.DiGraph()
    return nx.compose_all(graphs)
