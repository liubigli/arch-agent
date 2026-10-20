"""OpenUSD scene export using the optional official ``usd-core`` bindings."""

import json
import math
import os
from pathlib import Path
import re
import tempfile

from .scene_export import scene_to_json_data


def _json(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def export_scene_usd(ctx, output_path, *, include_cidoc=True, point_width=0.02,
                     color_max=65535):
    """Export a self-contained .usda/.usdc/.usd without replacing existing files.

    A single visible Points prim contains the entire loaded cloud. Object Xforms
    reference its point IDs, avoiding duplicate rendering of segmented points.
    Geometry is rebased around the first point to limit float precision loss;
    original double coordinates and the root translation retain the source frame.
    """
    output = Path(output_path)
    if output.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("OpenUSD output must end in .usd, .usda or .usdc.")
    if output.exists():
        raise FileExistsError(output)
    if not math.isfinite(point_width) or point_width <= 0:
        raise ValueError("point_width must be finite and positive.")
    if color_max not in (255, 65535):
        raise ValueError("color_max must be 255 or 65535.")
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    except ImportError as exc:
        raise ImportError("OpenUSD export requires usd-core: python -m pip install usd-core") from exc

    data = scene_to_json_data(ctx, include_cidoc=include_cidoc)
    records = data["point_cloud"].pop("points")
    coordinates = []
    for row in records:
        xyz = tuple(row.get(axis) for axis in ("x", "y", "z"))
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in xyz):
            raise ValueError("OpenUSD geometry requires finite x, y, z for every point.")
        coordinates.append(xyz)
    origin = coordinates[0] if coordinates else (0., 0., 0.)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Scene")
    root.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*origin))
    stage.SetDefaultPrim(root.GetPrim())

    def attr(prim, name, value, kind=Sdf.ValueTypeNames.String):
        prim.CreateAttribute(name, kind, custom=True).Set(value)

    def metadata(prim, value):
        attr(prim, "arch:dataJson", _json(value))

    metadata(root.GetPrim(), data)
    attr(root.GetPrim(), "arch:schemaVersion", "1.0")
    cloud = UsdGeom.Points.Define(stage, "/Scene/PointCloud")
    positions = Vt.Vec3fArray([Gf.Vec3f(*(v-o for v, o in zip(xyz, origin)))
                              for xyz in coordinates])
    cloud.CreatePointsAttr(positions)
    cloud.CreateIdsAttr(Vt.Int64Array(range(len(records))))
    cloud.CreateWidthsAttr([point_width])
    cloud.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    if positions:
        radius = point_width / 2
        cloud.CreateExtentAttr(Vt.Vec3fArray([
            Gf.Vec3f(*(min(p[i] for p in positions) - radius for i in range(3))),
            Gf.Vec3f(*(max(p[i] for p in positions) + radius for i in range(3))),
        ]))
    attr(cloud.GetPrim(), "arch:sourceCoordinates", Vt.Vec3dArray([Gf.Vec3d(*p) for p in coordinates]),
         Sdf.ValueTypeNames.Double3Array)
    attr(cloud.GetPrim(), "arch:pointAttributesJson",
         _json([{k: v for k, v in row.items() if k not in {"x", "y", "z"}} for row in records]))
    pv = UsdGeom.PrimvarsAPI(cloud)
    pv.CreatePrimvar("semanticLabel", Sdf.ValueTypeNames.StringArray, UsdGeom.Tokens.vertex).Set(
        [str(row.get("semantic_label") or "") for row in records])
    if records and all(all(row.get(k) is not None for k in ("R", "G", "B")) for row in records):
        # USD displayColor is linear; LAZ RGB is interpreted as sRGB.
        def linear(channel):
            v = min(1., max(0., channel / color_max))
            return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
        cloud.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
            Vt.Vec3fArray([Gf.Vec3f(*(linear(row[k]) for k in ("R", "G", "B"))) for row in records]))
    if records and all(all(row.get(k) is not None for k in ("nx", "ny", "nz")) for row in records):
        cloud.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*(row[k] for k in ("nx", "ny", "nz")))
                                            for row in records]))
        cloud.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    def safe_name(index, name):
        return f"item_{index}_" + re.sub(r"[^A-Za-z0-9_]", "_", str(name))[:80]

    UsdGeom.Scope.Define(stage, "/Scene/Objects")
    object_paths = {}
    for i, (name, obj) in enumerate(data["objects"].items()):
        prim = UsdGeom.Xform.Define(stage, "/Scene/Objects/" + safe_name(i, name)).GetPrim()
        object_paths[name] = prim.GetPath()
        attr(prim, "arch:objectId", name)
        attr(prim, "arch:pointIds", Vt.Int64Array(obj["point_ids"]), Sdf.ValueTypeNames.Int64Array)
        prim.CreateRelationship("arch:pointCloud", custom=True).SetTargets([cloud.GetPath()])
        metadata(prim, {"object": obj, "features": data["features"].get(name, {}),
                        "annotations": data["object_annotations"].get(name, [])})

    def edges(scope, rows, paths):
        UsdGeom.Scope.Define(stage, scope)
        for i, (source, target, predicate, details) in enumerate(rows):
            prim = UsdGeom.Scope.Define(stage, scope + "/edge_" + str(i)).GetPrim()
            attr(prim, "arch:predicate", str(predicate))
            metadata(prim, details)
            for role, node in (("source", source), ("target", target)):
                if node not in paths:
                    raise ValueError(f"Relationship endpoint {node!r} has no exported prim.")
                prim.CreateRelationship("arch:" + role, custom=True).SetTargets([paths[node]])

    edges("/Scene/SpatialRelations",
          [(r[0], r[1], r[2], r) for r in data["relationships"]], object_paths)
    cidoc = data["cidoc_graph"]
    if cidoc is not None:
        UsdGeom.Scope.Define(stage, "/Scene/CIDOC")
        UsdGeom.Scope.Define(stage, "/Scene/CIDOC/Nodes")
        node_paths = {}
        for i, node in enumerate(cidoc["nodes"]):
            prim = UsdGeom.Scope.Define(stage, "/Scene/CIDOC/Nodes/" + safe_name(i, node["node_id"])).GetPrim()
            node_paths[node["node_id"]] = prim.GetPath()
            metadata(prim, node)
            obj = node["properties"].get("object_name")
            if obj in object_paths:
                prim.CreateRelationship("arch:sceneObject", custom=True).SetTargets([object_paths[obj]])
        edges("/Scene/CIDOC/Edges", [(e["source"], e["target"], e["predicate"], e)
                                    for e in cidoc["edges"]], node_paths)

    output.parent.mkdir(parents=True, exist_ok=True)
    # Export through the SDK, then publish without an overwrite race. Keeping the
    # temporary file on the same filesystem permits an atomic hard link.
    fd, temporary = tempfile.mkstemp(suffix=output.suffix.lower(), dir=output.parent)
    os.close(fd)
    try:
        if not stage.GetRootLayer().Export(temporary):
            raise RuntimeError("OpenUSD failed to export the scene.")
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output
