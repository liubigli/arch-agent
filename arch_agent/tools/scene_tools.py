from collections import Counter
from typing import Optional

import networkx as nx
from langchain_core.tools import tool

from ..pipeline.pipeline import SceneContext, run_pipeline
from ..pipeline.graph import analyze_scene_graph
from ..settings import get_config


def _structural_elements() -> set[str]:
    return set(get_config()["semantic_classes"]["structural"])


def _all_semantic_classes() -> list[str]:
    return list(get_config()["semantic_classes"]["names"])


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
        lines = [
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
        ]
        color = _mean_rgb(obj["points"])
        if color is not None:
            raw, rgb8 = color
            lines.append(
                "  Mean RGB        : "
                f"raw ({raw[0]:.1f}, {raw[1]:.1f}, {raw[2]:.1f}); "
                f"8-bit ({rgb8[0]}, {rgb8[1]}, {rgb8[2]})"
            )

        return "\n".join(lines)

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
    def list_relationships(
        level: str = "all",
        relationship_type: Optional[str] = None,
        object_name: Optional[str] = None,
        limit: int = 200,
    ) -> str:
        """List relationships from the scene graph.

        Args:
            level: Relationship level to list: 'L1', 'L2', 'L3', 'geometric',
                'structural', 'mereological', or 'all'.
            relationship_type: Optional relationship type, e.g. 'above',
                'supports', 'is_opening_in'.
            object_name: Optional object name. If provided, only relationships
                where this object is source or target are listed.
            limit: Maximum number of relationship rows to return.
        """
        layer_key = _relationship_layer_key(level)
        if layer_key is None:
            valid = "all, L1/geometric, L2/structural, L3/mereological"
            return f"Unknown relationship level '{level}'. Valid values: {valid}."
        if object_name and object_name not in ctx.objects:
            sample = ", ".join(list(ctx.objects.keys())[:8])
            return f"Object '{object_name}' not found. Examples: {sample}"

        relationships = (
            ctx.relationships
            if layer_key == "all"
            else ctx.relationship_layers.get(layer_key, [])
        )
        filtered = [
            rel for rel in relationships
            if (not relationship_type or rel[2] == relationship_type)
            and (not object_name or rel[0] == object_name or rel[1] == object_name)
        ]

        title = f"Relationships ({level}): {len(filtered)}"
        if relationship_type:
            title += f" | type={relationship_type}"
        if object_name:
            title += f" | object={object_name}"

        type_counts = Counter(rel[2] for rel in filtered)
        if type_counts:
            title += " | " + ", ".join(
                f"{rel_type}={count}" for rel_type, count in sorted(type_counts.items())
            )

        max_rows = max(1, min(int(limit), 1000))
        lines = [title]
        for src, tgt, rel_type, rel_level in filtered[:max_rows]:
            lines.append(f"  - {src} --[{rel_level}:{rel_type}]--> {tgt}")
        if len(filtered) > max_rows:
            lines.append(
                f"  ... {len(filtered) - max_rows} more relationships not shown; "
                "increase limit if needed, but avoid pasting thousands of rows into chat."
            )
        if len(lines) == 1:
            lines.append("  No matching relationships found.")
        return "\n".join(lines)

    @tool
    def find_relationship_anomalies(limit: int = 200) -> str:
        """Find direct logical or semantic anomalies in the computed relationship graph.

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

        structural = _structural_elements()
        structural_count = sum(
            count for label, count in class_counts.items() if label in structural
        )
        finishing_count = len(ctx.objects) - structural_count
        lines.append(
            f"Element types: structural={structural_count}, "
            f"finishing={finishing_count}"
        )

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
    def get_point_cloud_info() -> str:
        """Get point-cloud level metrics: point count, classes, bounding box, and RGB availability."""
        if ctx.df is None or ctx.df.empty:
            return "No point-cloud dataframe is available."

        mins = ctx.df[["x", "y", "z"]].min()
        maxs = ctx.df[["x", "y", "z"]].max()
        dims = maxs - mins
        volume = float(dims["x"] * dims["y"] * dims["z"])
        class_counts = ctx.df["semantic_label"].value_counts().sort_index()

        lines = [
            f"Point count: {len(ctx.df):,}",
            "Bounding box:",
            f"  Min (x,y,z): ({mins['x']:.2f}, {mins['y']:.2f}, {mins['z']:.2f})",
            f"  Max (x,y,z): ({maxs['x']:.2f}, {maxs['y']:.2f}, {maxs['z']:.2f})",
            f"  Size (x,y,z): ({dims['x']:.2f}, {dims['y']:.2f}, {dims['z']:.2f}) m",
            f"  AABB volume: {volume:.3f} m3",
            "Point classes:",
        ]
        lines.extend(f"  - {label}: {count:,}" for label, count in class_counts.items())
        lines.append(
            "RGB channels: "
            + ("available" if _has_rgb(ctx.df) else "not available")
        )
        return "\n".join(lines)

    @tool
    def get_color_summary(
        semantic_label: Optional[str] = None,
        object_name: Optional[str] = None,
    ) -> str:
        """Summarize mean RGB values for the whole scene, a semantic class, or one object.

        Args:
            semantic_label: Optional semantic class to summarize, e.g. 'column'.
            object_name: Optional object name to summarize, e.g. 'column_0'.
        """
        if object_name:
            if object_name not in ctx.objects:
                return f"Object '{object_name}' not found."
            return _format_rgb_summary(object_name, ctx.objects[object_name]["points"])

        if semantic_label:
            parts = [
                obj["points"] for obj in ctx.objects.values()
                if obj["semantic_label"] == semantic_label
            ]
            if not parts:
                return f"No objects with semantic label '{semantic_label}' found."
            return _format_rgb_summary(semantic_label, _concat_frames(parts))

        if ctx.df is None or ctx.df.empty:
            return "No point-cloud dataframe is available."
        return _format_rgb_summary("scene", ctx.df)

    @tool
    def estimate_room_volume() -> str:
        """Estimate room volume as a containing box: floor footprint times envelope height."""
        room_volume = ctx.scene_features.get("room_volume", {})
        if not room_volume:
            return (
                "Cannot estimate room volume: no room_volume scene feature is available. "
                "A floor plus at least one wall, column, roof, or vault is required."
            )
        floor_dims = room_volume["floor_base_dimensions"]

        return "\n".join([
            "Room volume estimate as containing box:",
            f"  Floor footprint object: {room_volume['floor_object']}",
            f"  Floor base dimensions (AABB XY): {floor_dims[0]:.3f} x {floor_dims[1]:.3f} m",
            f"  Floor base area: {room_volume['floor_base_area']:.3f} m2",
            f"  Lower Z: floor top = {room_volume['lower_z']:.3f} m",
            f"  Upper Z: max wall/column/roof/vault Z = {room_volume['upper_z']:.3f} m",
            f"  Box height: {room_volume['height']:.3f} m",
            f"  Estimated volume: {room_volume['volume']:.3f} m3",
            f"  Feature: scene_features['room_volume']",
            "  Formula: floor base area multiplied by box height.",
        ])

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
        ctx.scene_features = new_ctx.scene_features
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
        list_relationships,
        find_relationship_anomalies,
        get_scene_statistics,
        get_point_cloud_info,
        get_color_summary,
        estimate_room_volume,
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


def _relationship_layer_key(level: str) -> str | None:
    normalized = (level or "all").strip().lower()
    aliases = {
        "all": "all",
        "l1": "L1",
        "geometric": "L1",
        "geometry": "L1",
        "l2": "L2",
        "structural": "L2",
        "structure": "L2",
        "l3": "L3",
        "mereological": "L3",
        "composition": "L3",
    }
    return aliases.get(normalized)


def _has_rgb(df) -> bool:
    return all(column in df.columns for column in ["R", "G", "B"])


def _mean_rgb(df) -> tuple[tuple[float, float, float], tuple[int, int, int]] | None:
    if not _has_rgb(df) or df.empty:
        return None
    raw = tuple(float(value) for value in df[["R", "G", "B"]].mean().to_numpy())
    max_channel = max(float(df[["R", "G", "B"]].max().max()), 1.0)
    divisor = 257.0 if max_channel > 255 else 1.0
    rgb8 = tuple(int(round(min(max(value / divisor, 0), 255))) for value in raw)
    return raw, rgb8


def _format_rgb_summary(name: str, df) -> str:
    color = _mean_rgb(df)
    if color is None:
        return f"RGB values are not available for {name}."
    raw, rgb8 = color
    return "\n".join([
        f"Color summary for {name}:",
        f"  Mean RGB raw: ({raw[0]:.1f}, {raw[1]:.1f}, {raw[2]:.1f})",
        f"  Mean RGB 8-bit: ({rgb8[0]}, {rgb8[1]}, {rgb8[2]})",
    ])


def _concat_frames(frames):
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


def _xy_area(bounds: dict) -> float:
    dims = bounds["max"][:2] - bounds["min"][:2]
    return float(max(dims[0], 0.0) * max(dims[1], 0.0))


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
            (a, b, "supports", "structural") in rel_set
            and (b, a, "supports", "structural") in rel_set
        ):
            issues.append(f"{a} and {b}: reciprocal 'supports' relation.")
        if (
            (a, b, "rests_on", "structural") in rel_set
            and (b, a, "rests_on", "structural") in rel_set
        ):
            issues.append(f"{a} and {b}: reciprocal 'rests_on' relation.")

        for src, tgt, rel_type, _ in rels:
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
