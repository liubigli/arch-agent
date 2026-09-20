"""Tools that report or change the state of the loaded scene."""


from collections import Counter
from typing import Optional

from langchain_core.tools import tool

from ..pipeline.graph import analyze_scene_graph
from ..pipeline.pipeline import SceneContext
from ..pipeline.pipeline import run_pipeline
from ..pipeline.relationships import architectural_role

from ._shared import (
    _all_semantic_classes,
)


def create_scene_state_tools(ctx: SceneContext) -> list:
    """Build the scene_state tools bound to ``ctx``."""

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
            absent = sorted(set(_all_semantic_classes()) - set(class_counts))
            lines.append(
                "Absent semantic classes: "
                + (", ".join(absent) if absent else "none")
            )

        role_counts = Counter()
        for label, count in class_counts.items():
            role_counts[architectural_role(label)] += count
        if role_counts:
            lines.append("Element roles:")
            for role, count in sorted(role_counts.items()):
                lines.append(f"  - {role}: {count}")

        room_volume = ctx.scene_features.get("room_volume", {})
        if room_volume:
            lines.append(
                "Room volume feature: "
                f"{room_volume['volume']:.3f} m3 "
                f"({room_volume['method']})"
            )

        lines.append("Graphs:")
        for level, graph in graphs.items():
            analysis = analyze_scene_graph(graph)
            lines.append(
                f"  - {level}: {analysis['node_count']} nodes, "
                f"{analysis['edge_count']} edges, avg degree {analysis['avg_degree']:.2f}"
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
        ctx.scene_features = new_ctx.scene_features
        ctx.relationships = new_ctx.relationships
        ctx.relationship_layers = new_ctx.relationship_layers
        ctx.scene_graph = new_ctx.scene_graph
        ctx.scene_graphs = new_ctx.scene_graphs
        ctx.object_annotations = new_ctx.object_annotations
        ctx.unmatched_annotations = new_ctx.unmatched_annotations

        return (
            "Scene reloaded. "
            f"Objects: {len(ctx.objects)} | Relationships: {len(ctx.relationships)}"
        )

    return [
        get_scene_statistics,
        reload_scene,
    ]
