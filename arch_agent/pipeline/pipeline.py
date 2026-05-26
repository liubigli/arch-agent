from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from .loader import load_semantic_point_cloud
from .segmentation import extract_semantic_objects
from .features import compute_object_features
from .relationships import compute_spatial_relationships
from .graph import build_scene_graph


@dataclass
class PipelineParams:
    csv_path: str
    sample_n: int = 150_000
    eps: float = 0.5
    min_samples: int = 15
    distance_threshold: float = 3.0
    use_normals: bool = False


@dataclass
class SceneContext:
    params: PipelineParams
    df: pd.DataFrame = field(default=None)
    objects: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    relationships: list = field(default_factory=list)
    scene_graph: nx.DiGraph = field(default=None)


def run_pipeline(params: PipelineParams) -> SceneContext:
    print(f"\n[1/5] Loading point cloud: {params.csv_path}")
    df = load_semantic_point_cloud(params.csv_path, sample_n=params.sample_n)

    print(f"[2/5] Segmenting objects  (eps={params.eps}, min_samples={params.min_samples})")
    objects = extract_semantic_objects(df, eps=params.eps, min_samples=params.min_samples)
    print(f"      → {len(objects)} objects found")

    print(f"[3/5] Computing features  (use_normals={params.use_normals})")
    features = compute_object_features(objects, use_normals=params.use_normals)

    print(f"[4/5] Computing relationships (threshold={params.distance_threshold} m)")
    relationships = compute_spatial_relationships(objects, distance_threshold=params.distance_threshold)
    print(f"      → {len(relationships)} relationships found")

    print("[5/5] Building scene graph")
    scene_graph = build_scene_graph(objects, relationships, features)
    print(f"      → {scene_graph.number_of_nodes()} nodes, {scene_graph.number_of_edges()} edges\n")

    return SceneContext(
        params=params,
        df=df,
        objects=objects,
        features=features,
        relationships=relationships,
        scene_graph=scene_graph,
    )
