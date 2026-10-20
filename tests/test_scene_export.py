"""JSON export contract tests, runnable with unittest."""

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from arch_agent.scene_export import export_scene_json, scene_to_json_data


class Index(list):
    @property
    def is_unique(self):
        return len(set(self)) == len(self)


class Frame:
    """Minimal DataFrame protocol fixture; indices deliberately are not positions."""
    def __init__(self, records, indices):
        self.records = records
        self.index = Index(indices)
        self.columns = list(records[0]) if records else []

    def __len__(self):
        return len(self.records)

    def to_dict(self, orient):
        assert orient == "records"
        return self.records


@dataclass
class Params:
    point_cloud_path: str = "scene.laz"
    sample_n: int = 0


def context():
    points = [{"x": 1., "y": 2., "z": 3., "semantic_label": "column", "R": 255},
              {"x": 4., "y": 5., "z": 6., "semantic_label": "wall", "R": 0}]
    return SimpleNamespace(
        params=Params(), df=Frame(points, [17, 42]),
        objects={"column_0": {"points": Frame(points[:1], [17]),
                              "semantic_label": "column", "point_count": 1}},
        object_annotations={}, unmatched_annotations=[{"description": "non associata"}],
        features={"column_0": {"surface_area": float("nan")}}, scene_features={},
        relationships=[], relationship_layers={"L1": []}, scene_graph=None, scene_graphs={},
    )


class SceneExportTests(unittest.TestCase):
    def test_points_links_noise_and_nulls_round_trip(self):
        ctx = context()
        with tempfile.TemporaryDirectory() as directory:
            path = export_scene_json(ctx, Path(directory) / "nested" / "scene.json")
            data = json.loads(path.read_text(encoding="utf-8"),
                              parse_constant=lambda value: self.fail(value))
        self.assertEqual(len(data["point_cloud"]["points"]), 2)
        self.assertEqual(data["objects"]["column_0"]["point_ids"], [0])
        self.assertEqual(data["point_cloud"]["points"][0]["R"], 255)
        self.assertIsNone(data["features"]["column_0"]["surface_area"])
        self.assertEqual(data["unmatched_annotations"], ctx.unmatched_annotations)
        self.assertEqual(data["cidoc_status"], "no_matched_annotations")
        self.assertNotIn("point_ids", ctx.objects["column_0"])

    def test_cidoc_is_exported_with_object_links(self):
        ctx = context()
        ctx.object_annotations = {"column_0": [{"material": "Pietra", "typology": "Colonna"}]}
        data = scene_to_json_data(ctx)
        self.assertEqual(data["cidoc_status"], "included")
        self.assertTrue(data["cidoc_graph"]["edges"])
        self.assertTrue(any(n["properties"].get("object_name") == "column_0"
                            for n in data["cidoc_graph"]["nodes"]))

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            path.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_scene_json(context(), path)
            self.assertEqual(path.read_text(encoding="utf-8"), "original")

    def test_invalid_point_membership_and_duplicate_indices_are_rejected(self):
        ctx = context()
        ctx.objects["column_0"]["points"].index = Index([999])
        with self.assertRaises(ValueError):
            scene_to_json_data(ctx)
        ctx = context()
        ctx.df.index = Index([17, 17])
        with self.assertRaises(ValueError):
            scene_to_json_data(ctx)

    def test_graph_attributes_and_multiple_edges_survive(self):
        class Graph:
            graph = {"name": "spatial"}
            def is_directed(self): return True
            def is_multigraph(self): return True
            def nodes(self, data): return [("column_0", {"material": "pietra"})]
            def edges(self, data, keys):
                return [("column_0", "column_0", 0, {"predicate": "near"}),
                        ("column_0", "column_0", 1, {"predicate": "above"})]
        ctx = context()
        ctx.scene_graphs = {"L1": Graph()}
        graph = scene_to_json_data(ctx)["scene_graphs"]["L1"]
        self.assertEqual(len(graph["edges"]), 2)
        self.assertEqual(graph["nodes"][0]["attributes"]["material"], "pietra")


if __name__ == "__main__":
    unittest.main()
