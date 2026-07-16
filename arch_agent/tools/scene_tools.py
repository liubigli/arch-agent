from collections import Counter
from typing import Optional

import networkx as nx
from langchain_core.tools import tool

from ..pipeline.point_metrics import (
    format_material_summary,
    format_rgb_summary,
    format_roughness_summary,
    has_rgb,
    rgb_statistics,
)
from ..pipeline.pipeline import SceneContext, run_pipeline
from ..pipeline.graph import analyze_scene_graph
from ..pipeline.relationships import (
    RELATIONSHIP_LAYER_NAMES,
    RELATIONSHIP_LAYER_ORDER,
    architectural_role,
    mereological_relation_type,
    supports_label_pair,
)
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
        color = rgb_statistics(obj["points"])
        if color is not None:
            raw = color["mean_raw"]
            rgb8 = color["mean_rgb8"]
            lines.append(
                "  Mean RGB        : "
                f"raw ({raw[0]:.1f}, {raw[1]:.1f}, {raw[2]:.1f}); "
                f"8-bit ({rgb8[0]}, {rgb8[1]}, {rgb8[2]})"
            )

        return "\n".join(lines)

    @tool
    def find_relationships(object_name: str) -> str:
        """Find all relationships involving a given object, using the L1->L2->L3 cascade.

        Args:
            object_name: Name of the object to query.
        """
        if object_name not in ctx.objects:
            sample = ", ".join(list(ctx.objects.keys())[:8])
            return f"Object '{object_name}' not found. Examples: {sample}"

        lines = [
            f"Relationships for '{object_name}':",
            "Cascade: L1/geometric -> L2/structural -> L3/mereological",
        ]
        total = 0

        for level, relationships in _relationship_layers_in_order(ctx):
            filtered = [
                rel for rel in relationships
                if rel[0] == object_name or rel[1] == object_name
            ]
            total += len(filtered)
            layer_name = RELATIONSHIP_LAYER_NAMES.get(level, level)
            lines.append(f"  {level}/{layer_name}: {len(filtered)}")
            for src, tgt, rel_type, rel_level in filtered:
                lines.append(f"    {src} --[{rel_level}:{rel_type}]--> {tgt}")

        if total == 0:
            lines.append("  No relationships found.")

        return "\n".join(lines)

    @tool
    def list_relationships(
        level: str = "all",
        relationship_type: Optional[str] = None,
        object_name: Optional[str] = None,
        limit: int = 30,
    ) -> str:
        """List relationships from the scene graph.

        Args:
            level: Relationship level to list: 'L1', 'L2', 'L3', 'geometric',
                'structural', 'mereological', or 'all'.
            relationship_type: Optional relationship type, e.g. 'above',
                'supports', 'is_opening_in'.
            object_name: Optional object name. If provided, only relationships
                where this object is source or target are listed.
            limit: Maximum number of relationship rows to return. Default is
                intentionally small to avoid flooding the chat.
        """
        layer_key = _relationship_layer_key(level)
        if layer_key is None:
            valid = "all, L1/geometric, L2/structural, L3/mereological"
            return f"Unknown relationship level '{level}'. Valid values: {valid}."
        if object_name and object_name not in ctx.objects:
            sample = ", ".join(list(ctx.objects.keys())[:8])
            return f"Object '{object_name}' not found. Examples: {sample}"

        layers = (
            _relationship_layers_in_order(ctx)
            if layer_key == "all"
            else [(layer_key, ctx.relationship_layers.get(layer_key, []))]
        )
        filtered_by_layer = [
            (
                layer,
                [
                    rel for rel in relationships
                    if (not relationship_type or rel[2] == relationship_type)
                    and (not object_name or rel[0] == object_name or rel[1] == object_name)
                ],
            )
            for layer, relationships in layers
        ]
        filtered = [
            rel
            for _, layer_relationships in filtered_by_layer
            for rel in layer_relationships
        ]

        title = f"Relationships ({level}): {len(filtered)}"
        if layer_key == "all":
            title += " | cascade=L1/geometric->L2/structural->L3/mereological"
        if relationship_type:
            title += f" | type={relationship_type}"
        if object_name:
            title += f" | object={object_name}"

        type_counts = Counter(rel[2] for rel in filtered)
        if type_counts:
            title += " | " + ", ".join(
                f"{rel_type}={count}" for rel_type, count in sorted(type_counts.items())
            )

        max_rows = max(1, min(int(limit), 200))
        lines = [title]
        remaining = max_rows
        hidden = 0
        for layer, layer_relationships in filtered_by_layer:
            layer_name = RELATIONSHIP_LAYER_NAMES.get(layer, layer)
            lines.append(f"  {layer}/{layer_name}: {len(layer_relationships)}")
            shown = layer_relationships[:remaining] if remaining > 0 else []
            for src, tgt, rel_type, rel_level in shown:
                lines.append(f"    - {src} --[{rel_level}:{rel_type}]--> {tgt}")
            hidden += max(0, len(layer_relationships) - len(shown))
            remaining -= len(shown)
        if hidden:
            lines.append(
                f"  ... {hidden} more relationships not shown; "
                "increase limit if needed, but avoid pasting thousands of rows into chat."
            )
        if not filtered:
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
    def get_point_cloud_info() -> str:
        """Get point-cloud level metrics: point count, classes, bounding box, footprint, and RGB availability."""
        if ctx.df is None or ctx.df.empty:
            return "No point-cloud dataframe is available."

        mins = ctx.df[["x", "y", "z"]].min()
        maxs = ctx.df[["x", "y", "z"]].max()
        dims = maxs - mins
        footprint_area = float(dims["x"] * dims["y"])
        volume = float(dims["x"] * dims["y"] * dims["z"])
        class_counts = ctx.df["semantic_label"].value_counts().sort_index()

        lines = [
            f"Point count: {len(ctx.df):,}",
            "Bounding box:",
            f"  Min (x,y,z): ({mins['x']:.2f}, {mins['y']:.2f}, {mins['z']:.2f})",
            f"  Max (x,y,z): ({maxs['x']:.2f}, {maxs['y']:.2f}, {maxs['z']:.2f})",
            f"  Size (x,y,z): ({dims['x']:.2f}, {dims['y']:.2f}, {dims['z']:.2f}) m",
            f"  XY footprint area: {footprint_area:.3f} m2",
            f"  AABB volume: {volume:.3f} m3",
            "Point classes:",
        ]
        lines.extend(f"  - {label}: {count:,}" for label, count in class_counts.items())
        lines.append(
            "RGB channels: "
            + ("available" if has_rgb(ctx.df) else "not available")
        )
        return "\n".join(lines)

    @tool
    def measure_occupied_area(
        semantic_label: Optional[str] = None,
        object_name: Optional[str] = None,
    ) -> str:
        """Measure occupied area/footprint in square meters.

        Use this for questions about area, occupied area, footprint, area della
        scena, superficie occupata, or impronta. This returns XY AABB footprint
        area in m2, not room volume. Do not use estimate_room_volume for area.

        Args:
            semantic_label: Optional semantic class, e.g. 'floor' or 'column'.
            object_name: Optional object name, e.g. 'floor_0'.
        """
        if object_name:
            if object_name not in ctx.objects:
                return _object_not_found_message(object_name, ctx.objects)
            area = _xy_area(ctx.objects[object_name]["bounds"])
            label = ctx.objects[object_name]["semantic_label"]
            return (
                f"Occupied area for {object_name} ({label}): {area:.3f} m2 "
                "(XY AABB footprint, not volume)."
            )

        if semantic_label:
            matching = [
                (name, obj)
                for name, obj in ctx.objects.items()
                if obj["semantic_label"] == semantic_label
            ]
            if not matching:
                return f"No objects with semantic label '{semantic_label}' found."
            rows = [
                (name, _xy_area(obj["bounds"]))
                for name, obj in matching
            ]
            total = sum(area for _, area in rows)
            largest_name, largest_area = max(rows, key=lambda row: row[1])
            return (
                f"Occupied area for class {semantic_label}: {total:.3f} m2 "
                f"summing {len(rows)} XY AABB footprints. "
                f"Largest object: {largest_name} = {largest_area:.3f} m2."
            )

        if ctx.df is None or ctx.df.empty:
            return "No point-cloud dataframe is available."
        mins = ctx.df[["x", "y"]].min()
        maxs = ctx.df[["x", "y"]].max()
        dx = float(maxs["x"] - mins["x"])
        dy = float(maxs["y"] - mins["y"])
        area = dx * dy
        return (
            f"Scene occupied area: {area:.3f} m2 "
            f"(XY AABB footprint: {dx:.3f} x {dy:.3f} m; not volume)."
        )

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
            return format_rgb_summary(object_name, ctx.objects[object_name]["points"])

        if semantic_label:
            parts = [
                obj["points"] for obj in ctx.objects.values()
                if obj["semantic_label"] == semantic_label
            ]
            if not parts:
                return f"No objects with semantic label '{semantic_label}' found."
            return format_rgb_summary(semantic_label, _concat_frames(parts))

        if ctx.df is None or ctx.df.empty:
            return "No point-cloud dataframe is available."
        return format_rgb_summary("scene", ctx.df)

    @tool
    def analyze_surface_roughness(
        semantic_label: Optional[str] = None,
        object_name: Optional[str] = None,
        sample_size: int = 5000,
        k_neighbors: int = 24,
    ) -> str:
        """Estimate surface roughness from point-cloud geometry.

        Roughness is estimated as the local residual from a best-fit plane
        computed with PCA over k-nearest neighbors. It can reflect material
        roughness, scan noise, curvature, or segmentation artifacts.

        Args:
            semantic_label: Optional semantic class to analyze, e.g. 'wall'.
            object_name: Optional object name to analyze, e.g. 'wall_0'.
            sample_size: Maximum number of points sampled for the estimate.
            k_neighbors: Number of neighbors used for local plane fitting.
        """
        if object_name:
            if object_name not in ctx.objects:
                return f"Object '{object_name}' not found."
            return format_roughness_summary(
                object_name,
                ctx.objects[object_name]["points"],
                sample_size=sample_size,
                k_neighbors=k_neighbors,
            )

        if semantic_label:
            parts = [
                obj["points"] for obj in ctx.objects.values()
                if obj["semantic_label"] == semantic_label
            ]
            if not parts:
                return f"No objects with semantic label '{semantic_label}' found."
            return format_roughness_summary(
                semantic_label,
                _concat_frames(parts),
                sample_size=sample_size,
                k_neighbors=k_neighbors,
            )

        if ctx.df is None or ctx.df.empty:
            return "No point-cloud dataframe is available."
        return format_roughness_summary(
            "scene",
            ctx.df,
            sample_size=sample_size,
            k_neighbors=k_neighbors,
        )

    @tool
    def infer_material_from_color(
        semantic_label: Optional[str] = None,
        object_name: Optional[str] = None,
        sample_size: int = 3000,
        k_neighbors: int = 24,
    ) -> str:
        """Infer candidate materials from semantic class, RGB color, and surface roughness.

        This is an architectural/material hypothesis, not a direct material
        measurement. RGB can be affected by lighting, scanner calibration,
        texture, shadows, and post-processing.

        Args:
            semantic_label: Optional semantic class to analyze, e.g. 'wall'.
            object_name: Optional object name to analyze, e.g. 'wall_0'.
            sample_size: Maximum number of points sampled for roughness.
            k_neighbors: Number of neighbors used for local roughness.
        """
        if object_name:
            if object_name not in ctx.objects:
                return f"Object '{object_name}' not found."
            obj = ctx.objects[object_name]
            return format_material_summary(
                object_name,
                obj["points"],
                semantic_label=obj["semantic_label"],
                sample_size=sample_size,
                k_neighbors=k_neighbors,
            )

        if semantic_label:
            parts = [
                obj["points"] for obj in ctx.objects.values()
                if obj["semantic_label"] == semantic_label
            ]
            if not parts:
                return f"No objects with semantic label '{semantic_label}' found."
            return format_material_summary(
                semantic_label,
                _concat_frames(parts),
                semantic_label=semantic_label,
                sample_size=sample_size,
                k_neighbors=k_neighbors,
            )

        if ctx.df is None or ctx.df.empty:
            return "No point-cloud dataframe is available."
        return format_material_summary(
            "scene",
            ctx.df,
            sample_size=sample_size,
            k_neighbors=k_neighbors,
        )

    @tool
    def estimate_room_volume() -> str:
        """Estimate room volume in cubic meters.

        Use only for room-volume questions. Do not use this for area, occupied
        area, footprint, area della scena, superficie occupata, or impronta:
        those require measure_occupied_area and must be reported in m2.
        """
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
    def measure_distance(object_a: str, object_b: str) -> str:
        """Measure geometric distances between two detected objects.

        Args:
            object_a: First object name, e.g. 'column_0'.
            object_b: Second object name, e.g. 'roof_0'.
        """
        if object_a not in ctx.objects:
            return _object_not_found_message(object_a, ctx.objects)
        if object_b not in ctx.objects:
            return _object_not_found_message(object_b, ctx.objects)

        metrics = _distance_metrics(ctx.objects[object_a], ctx.objects[object_b])
        return _format_distance_metrics(object_a, object_b, metrics)

    @tool
    def find_nearest_objects(
        object_name: str,
        limit: int = 10,
        semantic_label: Optional[str] = None,
    ) -> str:
        """Find nearest objects to a detected object by bounding-box gap.

        Args:
            object_name: Reference object name, e.g. 'roof_0'.
            limit: Maximum number of nearest objects to return.
            semantic_label: Optional semantic class filter, e.g. 'column'.
        """
        if object_name not in ctx.objects:
            return _object_not_found_message(object_name, ctx.objects)

        rows = []
        for candidate_name, candidate in ctx.objects.items():
            if candidate_name == object_name:
                continue
            if semantic_label and candidate["semantic_label"] != semantic_label:
                continue
            metrics = _distance_metrics(ctx.objects[object_name], candidate)
            rows.append((candidate_name, candidate["semantic_label"], metrics))

        rows.sort(key=lambda row: (row[2]["bbox_gap"], row[2]["centroid_distance"]))
        max_rows = max(1, min(int(limit), 100))
        lines = [
            f"Nearest objects to {object_name}"
            + (f" with class {semantic_label}" if semantic_label else "")
            + f": {len(rows)} candidates"
        ]
        for candidate_name, label, metrics in rows[:max_rows]:
            lines.append(
                f"  - {candidate_name} ({label}): "
                f"bbox_gap={metrics['bbox_gap']:.3f} m, "
                f"centroid={metrics['centroid_distance']:.3f} m, "
                f"vertical_gap={metrics['vertical_gap']:.3f} m, "
                f"xy_overlap={metrics['xy_overlap_ratio']:.3f}"
            )
        if len(rows) > max_rows:
            lines.append(f"  ... {len(rows) - max_rows} more candidates not shown.")
        if not rows:
            lines.append("  No matching objects found.")
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
        measure_occupied_area,
        get_color_summary,
        analyze_surface_roughness,
        infer_material_from_color,
        estimate_room_volume,
        measure_distance,
        find_nearest_objects,
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


def _relationship_layers_in_order(ctx: SceneContext) -> list[tuple[str, list]]:
    if not ctx.relationship_layers:
        return [("all", ctx.relationships)]
    return [
        (level, ctx.relationship_layers.get(level, []))
        for level in RELATIONSHIP_LAYER_ORDER
    ]


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


def _concat_frames(frames):
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


def _xy_area(bounds: dict) -> float:
    dims = bounds["max"][:2] - bounds["min"][:2]
    return float(max(dims[0], 0.0) * max(dims[1], 0.0))


def _object_not_found_message(object_name: str, objects: dict) -> str:
    sample = ", ".join(list(objects.keys())[:8])
    return f"Object '{object_name}' not found. Examples: {sample}"


def _distance_metrics(obj_a: dict, obj_b: dict) -> dict:
    c_a = obj_a["centroid"]
    c_b = obj_b["centroid"]
    centroid_distance = float(np_linalg_norm(c_a - c_b))
    gaps = _axis_gaps(obj_a["bounds"], obj_b["bounds"])
    bbox_gap = float(np_linalg_norm(gaps))
    vertical_gap = _signed_vertical_gap(obj_a["bounds"], obj_b["bounds"])
    xy_overlap_ratio = _overlap_xy_ratio(obj_a["bounds"], obj_b["bounds"])

    return {
        "centroid_distance": centroid_distance,
        "bbox_gap": bbox_gap,
        "gap_x": gaps[0],
        "gap_y": gaps[1],
        "gap_z": gaps[2],
        "vertical_gap": vertical_gap,
        "xy_overlap_ratio": xy_overlap_ratio,
        "touching_or_overlapping": bbox_gap == 0.0,
    }


def _format_distance_metrics(object_a: str, object_b: str, metrics: dict) -> str:
    return "\n".join([
        f"Distance between {object_a} and {object_b}:",
        f"  Centroid distance: {metrics['centroid_distance']:.3f} m",
        f"  Bounding-box gap: {metrics['bbox_gap']:.3f} m",
        f"  Axis gaps (x,y,z): ({metrics['gap_x']:.3f}, {metrics['gap_y']:.3f}, {metrics['gap_z']:.3f}) m",
        f"  Signed vertical gap: {metrics['vertical_gap']:.3f} m",
        f"  XY overlap ratio: {metrics['xy_overlap_ratio']:.3f}",
        "  Bounding boxes touch/overlap: "
        + ("yes" if metrics["touching_or_overlapping"] else "no"),
    ])


def _axis_gaps(bounds_a: dict, bounds_b: dict) -> list[float]:
    return [
        _axis_gap(
            float(bounds_a["min"][axis]),
            float(bounds_a["max"][axis]),
            float(bounds_b["min"][axis]),
            float(bounds_b["max"][axis]),
        )
        for axis in range(3)
    ]


def _axis_gap(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    if max_a < min_b:
        return float(min_b - max_a)
    if max_b < min_a:
        return float(min_a - max_b)
    return 0.0


def _signed_vertical_gap(bounds_a: dict, bounds_b: dict) -> float:
    if bounds_a["max"][2] < bounds_b["min"][2]:
        return float(bounds_b["min"][2] - bounds_a["max"][2])
    if bounds_b["max"][2] < bounds_a["min"][2]:
        return float(bounds_a["min"][2] - bounds_b["max"][2])
    return 0.0


def _overlap_xy_ratio(bounds_a: dict, bounds_b: dict) -> float:
    x_overlap = max(
        0.0,
        min(bounds_a["max"][0], bounds_b["max"][0])
        - max(bounds_a["min"][0], bounds_b["min"][0]),
    )
    y_overlap = max(
        0.0,
        min(bounds_a["max"][1], bounds_b["max"][1])
        - max(bounds_a["min"][1], bounds_b["min"][1]),
    )
    reference_area = min(_xy_area(bounds_a), _xy_area(bounds_b))
    if reference_area <= 0:
        return 0.0
    return float((x_overlap * y_overlap) / reference_area)


def np_linalg_norm(values) -> float:
    return sum(float(value) ** 2 for value in values) ** 0.5


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
            if rel_type == "supports" and not supports_label_pair(src_label, tgt_label):
                issues.append(
                    f"{src} -> {tgt}: invalid supports for {src_label}->{tgt_label}."
                )
            if rel_type == "rests_on" and not supports_label_pair(tgt_label, src_label):
                issues.append(
                    f"{src} -> {tgt}: invalid rests_on for {src_label}->{tgt_label}."
                )
            if rel_level == "mereological":
                if rel_type == "has_part":
                    expected = mereological_relation_type(tgt_label, src_label)
                    if expected is None:
                        issues.append(
                            f"{src} -> {tgt}: has_part has no inverse class rule "
                            f"for {src_label}->{tgt_label}."
                        )
                else:
                    expected = mereological_relation_type(src_label, tgt_label)
                    if expected is None:
                        issues.append(
                            f"{src} -> {tgt}: invalid mereological relation '{rel_type}' "
                            f"for {src_label}->{tgt_label}."
                        )
                    elif rel_type != expected:
                        issues.append(
                            f"{src} -> {tgt}: expected '{expected}', got '{rel_type}' "
                            f"for {src_label}->{tgt_label}."
                        )

    return issues
