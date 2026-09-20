"""Tools that report geometry only: dimensions, areas, volumes and distances.

These read the point cloud and the computed features. They never interpret
material or historical meaning."""


from typing import Optional

from langchain_core.tools import tool

from ..pipeline.graph import ANNOTATION_NODE_FIELDS
from ..pipeline.pipeline import SceneContext

from ._shared import (
    _canonical_semantic_label,
    _combined_graph,
    _object_box_center_text,
    _object_not_found_message,
    _objects_with_semantic_label,
    _resolve_target_names,
    np_linalg_norm,
)


def create_geometry_tools(ctx: SceneContext) -> list:
    """Build the geometry tools bound to ``ctx``."""

    @tool
    def list_object_geometry(
        semantic_label: Optional[str] = None,
        object_name: Optional[str] = None,
    ) -> str:
        """List global centroid, AABB box center, bounds, and dimensions.

        Use this for coordinates, global coordinates, centroid, global box
        center, bounding-box center, or AABB center questions.
        This tool returns geometry only. For CSV descriptions, material,
        typology, function, or annotation match status, use
        get_object_annotation or list_csv_annotation_matches instead.

        Args:
            semantic_label: Optional semantic class, e.g. 'moldings'.
            object_name: Optional object name, e.g. 'moldings_0'.
        """
        semantic_label = _canonical_semantic_label(semantic_label)
        if object_name:
            if object_name not in ctx.objects:
                object_as_label = _canonical_semantic_label(object_name)
                if object_as_label and _objects_with_semantic_label(ctx, object_as_label):
                    semantic_label = object_as_label
                    object_name = None
                else:
                    return _object_not_found_message(object_name, ctx.objects)

        if object_name:
            names = [object_name]
            target = object_name
        elif semantic_label:
            names = sorted(
                name for name, obj in ctx.objects.items()
                if obj["semantic_label"] == semantic_label
            )
            target = semantic_label
        else:
            names = sorted(ctx.objects)
            target = "all objects"

        if not names:
            return f"No object found for {target!r}."

        lines = [
            f"Geometry for {target}: {len(names)} object(s).",
            "Values are global scene coordinates. box_center is the AABB center; centroid is the point mean.",
            "Geometry only: no CSV material, typology, function, or historical description is returned by this tool.",
        ]
        for name in names:
            obj = ctx.objects[name]
            bounds = obj["bounds"]
            centroid = obj["centroid"]
            box_center = (bounds["min"] + bounds["max"]) / 2.0
            dims = bounds["max"] - bounds["min"]
            lines.append(
                f"  - {name}: "
                f"centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}); "
                f"box_center=({box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f}); "
                f"dims=({dims[0]:.3f}, {dims[1]:.3f}, {dims[2]:.3f}) m; "
                f"bounds_min=({bounds['min'][0]:.3f}, {bounds['min'][1]:.3f}, {bounds['min'][2]:.3f}); "
                f"bounds_max=({bounds['max'][0]:.3f}, {bounds['max'][1]:.3f}, {bounds['max'][2]:.3f})"
            )
        return "\n".join(lines)

    @tool
    def get_object_info(
        object_name: Optional[str] = None,
        semantic_label: Optional[str] = None,
        semantic_labels: Optional[list[str]] = None,
    ) -> str:
        """Get geometric and semantic-class information about object(s).

        Use this for centroid, dimensions, point count, AABB volume, surface
        area, height, compactness, and architectural role. For material,
        typology, function, or descriptive CSV metadata, use
        get_object_semantic_details or get_object_annotation instead.

        Provide exactly one of these:
        - object_name: one exact object id (e.g. 'column_2') for a single instance.
        - semantic_label: a semantic class (e.g. 'column') to get info for
          every instance of that class.
        - semantic_labels: multiple semantic classes to get info for in one
          call, e.g. ['column', 'wall', 'floor'].

        Args:
            object_name: Exact object id for a single instance.
            semantic_label: Semantic class to report across all its instances.
            semantic_labels: Optional list of semantic classes to report
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
            feat = ctx.features.get(name, {})
            c = obj["centroid"]
            dims = obj["bounds"]["max"] - obj["bounds"]["min"]
            lines = [
                f"Object: {name}",
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
            annotations = getattr(ctx, "object_annotations", {}).get(name, [])
            if annotations:
                lines.append(f"  CSV annotations : {len(annotations)}")
            node_data = graph.nodes.get(name, {}) if graph is not None else {}
            for field in ANNOTATION_NODE_FIELDS:
                value = node_data.get(field)
                if value:
                    lines.append(f"  {field.capitalize():<16}: {value}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @tool
    def find_sparse_objects(
        min_points: Optional[int] = None,
        limit: int = 20,
    ) -> str:
        """List objects with few points as possible segmentation-quality warnings.

        Use this for questions about sparse objects, objects with few points,
        noisy/incomplete objects, possible segmentation errors, or "pochi
        punti". This does not inspect relationship contradictions; for
        relationship anomalies use find_relationship_anomalies.

        Args:
            min_points: Optional point-count threshold. If omitted, an
                automatic threshold is selected from the scene distribution.
            limit: Maximum number of low-point objects to list.
        """
        if not ctx.objects:
            return "No objects in the scene."

        point_counts = sorted(int(obj.get("point_count", 0)) for obj in ctx.objects.values())
        if min_points is None:
            low_decile_index = max(0, int(len(point_counts) * 0.10) - 1)
            threshold = max(100, point_counts[low_decile_index])
        else:
            threshold = max(0, int(min_points))

        sparse_rows = [
            (name, obj["semantic_label"], int(obj.get("point_count", 0)))
            for name, obj in ctx.objects.items()
            if int(obj.get("point_count", 0)) <= threshold
        ]
        sparse_rows.sort(key=lambda row: (row[2], row[0]))

        max_rows = max(1, min(int(limit), 200))
        lines = [
            "Sparse-object check:",
            f"  Threshold: <= {threshold} points",
            f"  Objects flagged: {len(sparse_rows)}",
            "  Meaning: low point count is a segmentation-quality warning, not a confirmed error.",
        ]
        for name, label, point_count in sparse_rows[:max_rows]:
            lines.append(
                f"  - {name} ({label}): {point_count:,} points; "
                f"box_center={_object_box_center_text(ctx, name)}"
            )
        if len(sparse_rows) > max_rows:
            lines.append(f"  ... {len(sparse_rows) - max_rows} more objects not shown.")
        if not sparse_rows:
            lines.append("  No objects below the selected threshold.")
        return "\n".join(lines)

    @tool
    def get_point_cloud_info() -> str:
        """Get point-cloud level metrics: point count, classes, bounding box, and footprint."""
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
        return "\n".join(lines)

    @tool
    def measure_scene_occupied_area() -> str:
        """Measure the whole scene occupied area/footprint in square meters.

        Use this for scene-wide questions such as "area occupata dalla scena",
        "superficie della scena", "scene occupied area", or "scene footprint".
        This returns the XY AABB footprint of all loaded points in m2.
        """
        if ctx.df is None or ctx.df.empty:
            return "No point-cloud dataframe is available."
        mins = ctx.df[["x", "y"]].min()
        maxs = ctx.df[["x", "y"]].max()
        dx = float(maxs["x"] - mins["x"])
        dy = float(maxs["y"] - mins["y"])
        area = dx * dy
        return (
            f"Scene occupied area: {area:.3f} m2 "
            f"(XY AABB footprint of all loaded points: {dx:.3f} x {dy:.3f} m; not volume)."
        )

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
            + f": {len(rows)} candidates",
            "Sorted by bbox_gap: minimum distance between bounding boxes, not centroid distance.",
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

    return [
        list_object_geometry,
        get_object_info,
        find_sparse_objects,
        get_point_cloud_info,
        measure_scene_occupied_area,
        measure_occupied_area,
        estimate_room_volume,
        measure_distance,
        find_nearest_objects,
    ]



def _axis_gap(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    if max_a < min_b:
        return float(min_b - max_a)
    if max_b < min_a:
        return float(min_a - max_b)
    return 0.0


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


def _signed_vertical_gap(bounds_a: dict, bounds_b: dict) -> float:
    if bounds_a["max"][2] < bounds_b["min"][2]:
        return float(bounds_b["min"][2] - bounds_a["max"][2])
    if bounds_b["max"][2] < bounds_a["min"][2]:
        return float(bounds_a["min"][2] - bounds_b["max"][2])
    return 0.0


def _xy_area(bounds: dict) -> float:
    dims = bounds["max"][:2] - bounds["min"][:2]
    return float(max(dims[0], 0.0) * max(dims[1], 0.0))
