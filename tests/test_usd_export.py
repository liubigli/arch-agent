"""Round-trip tests with the real OpenUSD parser and pipeline data types."""

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import networkx as nx
import numpy as np
import pandas as pd
from pxr import Usd, UsdGeom

from arch_agent.usd_export import export_scene_usd


@dataclass
class Params:
    point_cloud_path: str = "scene.laz"
    sample_n: int = 0


def context():
    df = pd.DataFrame({
        "x": [1_000_000.001, 1_000_000.002, 1_000_000.003],
        "y": [2., 3., 4.], "z": [5., 6., 7.],
        "semantic_label": ["column", "wall", "wall"],
        "R": [65535, 0, 100], "G": [0, 65535, 100], "B": [0, 0, 100],
        "nx": [0., 0., 0.], "ny": [0., 0., 0.], "nz": [1., 1., 1.],
    }, index=[17, 42, 99])
    objects = {name: {"points": df.loc[[index]], "semantic_label": label,
                      "point_count": 1, "centroid": np.array([1., 2., 3.])}
               for name, index, label in [("colonna è/1", 17, "column"), ("wall_0", 42, "wall")]}
    graph = nx.MultiDiGraph()
    graph.add_edge("colonna è/1", "wall_0", relationship="near")
    graph.add_edge("colonna è/1", "wall_0", relationship="adjacent_to")
    return SimpleNamespace(
        params=Params(), df=df, objects=objects,
        object_annotations={"colonna è/1": [{"material": "Pietra", "typology": "Colonna"}]},
        unmatched_annotations=[{"description": "non associata"}],
        features={"wall_0": {"surface_area": float("nan")}}, scene_features={},
        relationships=[("colonna è/1", "wall_0", "near", "geometric")],
        relationship_layers={"L1": [("colonna è/1", "wall_0", "near", "geometric")]},
        scene_graph=graph, scene_graphs={"L1": graph},
    )


class UsdExportTests(unittest.TestCase):
    def test_round_trip_all_formats_geometry_metadata_and_relations(self):
        for suffix in (".usda", ".usdc", ".usd"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                ctx = context()
                path = export_scene_usd(ctx, Path(directory) / ("scene" + suffix))
                stage = Usd.Stage.Open(str(path))
                self.assertTrue(stage)
                self.assertEqual(stage.GetDefaultPrim().GetPath().pathString, "/Scene")
                self.assertEqual(UsdGeom.GetStageUpAxis(stage), "Z")
                self.assertEqual(UsdGeom.GetStageMetersPerUnit(stage), 1.)
                cloud = UsdGeom.Points(stage.GetPrimAtPath("/Scene/PointCloud"))
                self.assertEqual(len(cloud.GetPointsAttr().Get()), 3)  # includes unclustered point
                self.assertEqual(list(cloud.GetIdsAttr().Get()), [0, 1, 2])
                self.assertAlmostEqual(cloud.GetPointsAttr().Get()[1][0], .001, places=8)
                self.assertEqual(cloud.GetPrim().GetAttribute("arch:sourceCoordinates").Get()[0][0],
                                 ctx.df.iloc[0]["x"])
                self.assertEqual(list(cloud.GetDisplayColorAttr().Get()[0]), [1., 0., 0.])
                self.assertEqual(cloud.GetNormalsInterpolation(), "vertex")
                self.assertEqual(cloud.GetWidthsInterpolation(), "constant")
                self.assertTrue(cloud.GetExtentAttr().Get())
                obj = list(stage.GetPrimAtPath("/Scene/Objects").GetChildren())[0]
                self.assertEqual(obj.GetAttribute("arch:objectId").Get(), "colonna è/1")
                self.assertEqual(list(obj.GetAttribute("arch:pointIds").Get()), [0])
                payload = json.loads(stage.GetDefaultPrim().GetAttribute("arch:dataJson").Get())
                self.assertIsNone(payload["features"]["wall_0"]["surface_area"])
                self.assertEqual(len(payload["scene_graph"]["edges"]), 2)
                for scope in ("/Scene/SpatialRelations", "/Scene/CIDOC/Edges"):
                    edges = list(stage.GetPrimAtPath(scope).GetChildren())
                    self.assertTrue(edges)
                    for edge in edges:
                        for role in ("source", "target"):
                            target = edge.GetRelationship("arch:" + role).GetTargets()[0]
                            self.assertTrue(stage.GetPrimAtPath(target))
                self.assertEqual(len(ctx.df), 3)
                self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_existing_file_and_invalid_geometry_do_not_damage_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.usda"
            path.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_scene_usd(context(), path)
            self.assertEqual(path.read_text(encoding="utf-8"), "original")
            invalid = Path(directory) / "invalid.usda"
            ctx = context()
            ctx.df.loc[17, "x"] = float("nan")
            with self.assertRaises(ValueError):
                export_scene_usd(ctx, invalid)
            self.assertFalse(invalid.exists())

    def test_empty_cloud_without_cidoc_or_optional_channels(self):
        ctx = context()
        ctx.df = pd.DataFrame(columns=["x", "y", "z", "semantic_label"])
        ctx.objects = {}; ctx.object_annotations = {}; ctx.relationships = []
        with tempfile.TemporaryDirectory() as directory:
            stage = Usd.Stage.Open(str(export_scene_usd(ctx, Path(directory) / "empty.usda")))
            self.assertEqual(len(UsdGeom.Points(stage.GetPrimAtPath("/Scene/PointCloud")).GetPointsAttr().Get()), 0)
            self.assertFalse(stage.GetPrimAtPath("/Scene/CIDOC"))


if __name__ == "__main__":
    unittest.main()
