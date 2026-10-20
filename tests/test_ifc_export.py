"""IFC4 round-trip, schema and geometry checks with IfcOpenShell."""

import json
from pathlib import Path
import tempfile
import unittest

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import ifcopenshell.validate

from arch_agent.ifc_export import export_scene_ifc
from test_usd_export import context


class IfcExportTests(unittest.TestCase):
    def test_round_trip_entities_properties_geometry_and_schema(self):
        ctx = context()
        for obj in ctx.objects.values():
            obj["bounds"] = {"min": [1000., 2000., 3.], "max": [1001., 2002., 6.]}
        with tempfile.TemporaryDirectory() as directory:
            path = export_scene_ifc(ctx, Path(directory) / "scene.ifc")
            model = ifcopenshell.open(str(path))
            self.assertEqual(model.schema, "IFC4")
            self.assertEqual(len(model.by_type("IfcWall")), 1)
            self.assertEqual(len(model.by_type("IfcColumn")), 1)
            self.assertEqual(len(model.by_type("IfcBuildingStorey")), 1)
            for product in model.by_type("IfcElement"):
                pset = ifcopenshell.util.element.get_psets(product)["ArchAgent_Object"]
                self.assertTrue(pset["IsApproximation"])
                self.assertEqual(pset["GeometryStatus"], "axis_aligned_bounding_box")
                self.assertEqual(len(json.loads(pset["PointIdsJson"])), 1)
                shape = ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), product)
                self.assertTrue(shape.geometry.verts)
            project = model.by_type("IfcProject")[0]
            pset = ifcopenshell.util.element.get_psets(project)["ArchAgent_Scene"]
            self.assertFalse(pset["PointCloudEmbedded"])
            scene = json.loads(pset["SceneJson"])
            self.assertTrue(scene["cidoc_graph"]["edges"])
            self.assertEqual(len(scene["ifc_object_guids"]), 2)
            self.assertEqual(model.by_type("IfcRelAssignsToProduct")[0].Name, "near")
            self.assertEqual(model.by_type("IfcDocumentReference")[0].Location, "scene.laz")
            units = model.by_type("IfcUnitAssignment")[0].Units
            self.assertTrue(any(u.UnitType == "LENGTHUNIT" and u.Name == "METRE" and u.Prefix is None
                                for u in units))
            logger = ifcopenshell.validate.json_logger()
            ifcopenshell.validate.validate(model, logger, express_rules=True)
            self.assertEqual(logger.statements, [], logger.statements)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_degenerate_bounds_are_not_invented_and_existing_files_survive(self):
        ctx = context()  # single-point objects: no solid can be inferred
        ctx.objects["wall_0"]["semantic_label"] = "door_window"
        with tempfile.TemporaryDirectory() as directory:
            path = export_scene_ifc(ctx, Path(directory) / "scene.ifc")
            model = ifcopenshell.open(str(path))
            self.assertEqual(len(model.by_type("IfcBuildingElementProxy")), 1)
            self.assertFalse(model.by_type("IfcElement")[0].Representation)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                export_scene_ifc(ctx, path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
