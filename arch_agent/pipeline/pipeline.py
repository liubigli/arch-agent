from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from .loader import load_semantic_point_cloud
from .segmentation import extract_semantic_objects
from .features import compute_object_features
from .relationships import compute_all_relations_stratified
from .graph import build_scene_graphs


@dataclass
class PipelineParams:
    point_cloud_path: str
    sample_n: int = 150_000
    eps: float = 0.5
    min_samples: int = 15
    distance_threshold: float = 3.0
    use_normals: bool = False

    @property
    def csv_path(self) -> str:
        return self.point_cloud_path


@dataclass
class SceneContext:
    params: PipelineParams
    df: pd.DataFrame = field(default=None)
    objects: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    relationships: list = field(default_factory=list)
    relationship_layers: dict = field(default_factory=dict)
    scene_graph: nx.DiGraph = field(default=None)
    scene_graphs: dict[str, nx.DiGraph] = field(default_factory=dict)


def run_pipeline(params: PipelineParams) -> SceneContext:
    print(f"\n[1/5] Loading point cloud: {params.point_cloud_path}")
    df = load_semantic_point_cloud(
        params.point_cloud_path,
        sample_n=params.sample_n,
        include_normals=params.use_normals,
    )

    print(f"[2/5] Segmenting objects  (eps={params.eps}, min_samples={params.min_samples})")
    objects = extract_semantic_objects(df, eps=params.eps, min_samples=params.min_samples)
    print(f"      -> {len(objects)} objects found")

    print(f"[3/5] Computing features  (use_normals={params.use_normals})")
    features = compute_object_features(objects, use_normals=params.use_normals)

    print(f"[4/5] Computing stratified relationships (threshold={params.distance_threshold} m)")
    relationship_layers = compute_all_relations_stratified(
        objects,
        distance_threshold=params.distance_threshold,
    )
    relationships = relationship_layers["all"]
    print(f"      -> {len(relationships)} relationships found")

    print("[5/5] Building stratified scene graphs")
    scene_graphs = build_scene_graphs(objects, relationship_layers, features)
    scene_graph = scene_graphs.get("L1")
    graph_summary = " | ".join(
        f"{level}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        for level, G in scene_graphs.items()
    )
    print(f"      -> {graph_summary}\n")

    return SceneContext(
        params=params,
        df=df,
        objects=objects,
        features=features,
        relationships=relationships,
        relationship_layers=relationship_layers,
        scene_graph=scene_graph,
        scene_graphs=scene_graphs,
    )
