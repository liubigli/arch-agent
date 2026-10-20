"""Portable JSON export of a computed scene, including its loaded points."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _graph_data(graph):
    if graph is None:
        return None
    return {
        "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph(),
        "attributes": dict(graph.graph),
        "nodes": [{"id": node, "attributes": attrs} for node, attrs in graph.nodes(data=True)],
        "edges": (
            [{"source": s, "target": t, "key": key, "attributes": attrs}
             for s, t, key, attrs in graph.edges(data=True, keys=True)]
            if graph.is_multigraph() else
            [{"source": s, "target": t, "attributes": attrs}
             for s, t, attrs in graph.edges(data=True)]
        ),
    }


def scene_to_json_data(ctx, *, include_cidoc=True):
    """Create a versioned JSON-compatible snapshot without changing the scene.

    Point IDs are zero-based positions in the exported points array, not LAZ
    source indices. Unclustered points are retained in that array too.
    """
    if ctx.df is None or not ctx.df.index.is_unique:
        raise ValueError("Export requires a loaded point cloud with unique row indices.")
    point_ids = {index: position for position, index in enumerate(ctx.df.index)}
    objects = {}
    for name, obj in ctx.objects.items():
        objects[name] = {key: value for key, value in obj.items() if key != "points"}
        try:
            objects[name]["point_ids"] = [point_ids[index] for index in obj["points"].index]
        except KeyError as exc:
            raise ValueError(f"Object {name!r} references points outside the loaded cloud.") from exc
    cidoc = None
    if include_cidoc and any(ctx.object_annotations.values()):
        from .pipeline.cidoc_context import build_cidoc_graph_from_context
        cidoc = asdict(build_cidoc_graph_from_context(ctx))
    data = {
        "schema_version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "point_cloud_path": ctx.params.point_cloud_path,
            "pipeline_parameters": asdict(ctx.params),
            "point_scope": "loaded_semantically_labelled_points",
            "point_count": len(ctx.df),
            "point_id_definition": "zero-based index in point_cloud.points",
            "coordinate_reference_system": None,
        },
        "point_cloud": {"columns": list(ctx.df.columns), "points": ctx.df.to_dict(orient="records")},
        "objects": objects,
        "object_annotations": ctx.object_annotations,
        "unmatched_annotations": ctx.unmatched_annotations,
        "features": ctx.features,
        "scene_features": ctx.scene_features,
        "relationships": ctx.relationships,
        "relationship_layers": ctx.relationship_layers,
        "scene_graph": _graph_data(ctx.scene_graph),
        "scene_graphs": {level: _graph_data(graph) for level, graph in ctx.scene_graphs.items()},
        "cidoc_graph": cidoc,
        "cidoc_status": "included" if cidoc is not None else (
            "disabled" if not include_cidoc else "no_matched_annotations"
        ),
    }
    return _json_value(data)


def export_scene_json(ctx, output_path, *, include_cidoc=True):
    """Write strict UTF-8 JSON. Existing files are never overwritten."""
    data = scene_to_json_data(ctx, include_cidoc=include_cidoc)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = output.open("x", encoding="utf-8")
    try:
        with handle:
            json.dump(data, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            handle.write("\n")
    except BaseException:
        output.unlink()
        raise
    return output
