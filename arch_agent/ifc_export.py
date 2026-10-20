"""IFC4 export of segmented objects as explicitly approximate bounding boxes."""

import json
import math
import os
from pathlib import Path
import tempfile

from .scene_export import scene_to_json_data


IFC_CLASSES = {
    "wall": "IfcWall", "column": "IfcColumn", "floor": "IfcSlab",
    "roof": "IfcRoof", "stairs": "IfcStair",
}


def export_scene_ifc(ctx, output_path, *, include_cidoc=True):
    """Write IFC4 with meter units, source links and application property sets.

    Unknown classes (including ambiguous door_window) use IfcBuildingElementProxy.
    Boxes are approximations, not reconstructed or parametric building geometry.
    Raw points stay in the external source; point IDs refer to the loaded cloud.
    """
    output = Path(output_path)
    if output.suffix.lower() != ".ifc":
        raise ValueError("IFC output must end in .ifc.")
    if output.exists():
        raise FileExistsError(output)
    try:
        import ifcopenshell
        import ifcopenshell.api
        import numpy as np
    except ImportError as exc:
        raise ImportError("IFC export requires ifcopenshell: python -m pip install -r requirements-ifc.txt") from exc

    data = scene_to_json_data(ctx, include_cidoc=include_cidoc)
    records = data["point_cloud"].pop("points")
    model = ifcopenshell.api.run("project.create_file", version="IFC4")

    def create(ifc_class, name):
        return ifcopenshell.api.run("root.create_entity", model, ifc_class=ifc_class, name=name)

    def props(product, name, values):
        pset = ifcopenshell.api.run("pset.add_pset", model, product=product, name=name)
        ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties=values)

    def encoded(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    project = create("IfcProject", Path(ctx.params.point_cloud_path).stem)
    units = [ifcopenshell.api.run("unit.add_si_unit", model, unit_type=kind)
             for kind in ("LENGTHUNIT", "AREAUNIT", "VOLUMEUNIT")]
    ifcopenshell.api.run("unit.assign_unit", model, units=units)
    parent = project
    for kind, name in (("IfcSite", "Site"), ("IfcBuilding", "Building"),
                       ("IfcBuildingStorey", "Scene")):
        child = create(kind, name)
        ifcopenshell.api.run("aggregate.assign_object", model, products=[child], relating_object=parent)
        ifcopenshell.api.run("geometry.edit_object_placement", model, product=child)
        parent = child
    storey = parent
    context = ifcopenshell.api.run("context.add_context", model, context_type="Model")
    body = ifcopenshell.api.run("context.add_context", model, context_type="Model",
                               context_identifier="Body", target_view="MODEL_VIEW", parent=context)
    products = {}
    for name, obj in data["objects"].items():
        product = create(IFC_CLASSES.get(obj.get("semantic_label"), "IfcBuildingElementProxy"), name)
        product.Description = "Approximate axis-aligned bounding box of segmented point-cloud object."
        products[name] = product
        ifcopenshell.api.run("spatial.assign_container", model, products=[product], relating_structure=storey)
        points = [records[i] for i in obj["point_ids"]]
        bounds = obj.get("bounds")
        if bounds is None and points:
            bounds = {"min": [min(p[a] for p in points) for a in ("x", "y", "z")],
                      "max": [max(p[a] for p in points) for a in ("x", "y", "z")]}
        status = "no_points"
        if bounds is not None:
            lo, hi = bounds["min"], bounds["max"]
            if len(lo) != 3 or len(hi) != 3 or any(
                    v is None or not math.isfinite(v) for v in (*lo, *hi)):
                raise ValueError(f"Object {name!r} has invalid bounding box coordinates.")
            dims = [b-a for a, b in zip(lo, hi)]
            if any(d < 0 for d in dims):
                raise ValueError(f"Object {name!r} has inverted bounds.")
            matrix = np.eye(4); matrix[:3, 3] = lo
            ifcopenshell.api.run("geometry.edit_object_placement", model, product=product, matrix=matrix)
            if all(d > 0 for d in dims):
                x, y, z = dims
                vertices = [(0.,0.,0.),(x,0.,0.),(x,y,0.),(0.,y,0.),
                            (0.,0.,z),(x,0.,z),(x,y,z),(0.,y,z)]
                faces = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
                representation = ifcopenshell.api.run("geometry.add_mesh_representation", model,
                    context=body, vertices=[vertices], faces=[faces])
                ifcopenshell.api.run("geometry.assign_representation", model, product=product,
                                     representation=representation)
                status = "axis_aligned_bounding_box"
            else:
                status = "degenerate_bounds_no_solid"
        props(product, "ArchAgent_Object", {
            "ObjectId": name, "SemanticLabel": obj.get("semantic_label", ""),
            "GeometryStatus": status, "IsApproximation": True,
            "PointIdsJson": encoded(obj["point_ids"]),
            "ObjectJson": encoded(obj), "FeaturesJson": encoded(data["features"].get(name, {})),
            "AnnotationsJson": encoded(data["object_annotations"].get(name, [])),
        })

    # Generic named assignments preserve graph predicates without asserting IFC
    # physical connectivity, containment, or composition from mere proximity.
    for source, target, predicate, *details in data["relationships"]:
        if source not in products or target not in products:
            raise ValueError("Spatial relationship references an unexported object.")
        model.create_entity("IfcRelAssignsToProduct", GlobalId=ifcopenshell.guid.new(),
            Name=str(predicate), Description=encoded(details),
            RelatedObjects=[products[target]], RelatingProduct=products[source])

    data["ifc_object_guids"] = {name: p.GlobalId for name, p in products.items()}
    props(project, "ArchAgent_Scene", {
        "SchemaVersion": "1.0", "GeometryMethod": "axis_aligned_bounding_boxes",
        "CoordinateAssumption": "meters, Z up; CRS unavailable",
        "SceneJson": encoded(data),
        "PointCloudEmbedded": False,
    })
    reference = model.create_entity("IfcDocumentReference", Location=str(ctx.params.point_cloud_path),
                                    Identification="point_cloud_source", Name="Source point cloud (external)")
    model.create_entity("IfcRelAssociatesDocument", GlobalId=ifcopenshell.guid.new(),
                        RelatedObjects=[project, *products.values()], RelatingDocument=reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix=".ifc", dir=output.parent)
    os.close(fd)
    try:
        model.write(temporary)
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output
