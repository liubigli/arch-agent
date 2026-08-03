"""NetworkX/Matplotlib viewer for arch-agent scene graphs."""

from __future__ import annotations

import argparse
from pathlib import Path, PureWindowsPath

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from arch_agent.pipeline.pipeline import PipelineParams, run_pipeline

try:
    from arch_agent.pipeline.l3_cidoc_graph_builder import (
        SceneGraph as CidocSceneGraph,
        build_scene_graph as build_l3_cidoc_scene_graph,
    )
except ImportError:  # pragma: no cover - depends on repo version
    CidocSceneGraph = None
    build_l3_cidoc_scene_graph = None


LEVEL_NAMES = {
    "L1": "geometric",
    "L3": "CIDOC ontology",
}


def resolve_local_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.exists():
        return path

    windows_path = PureWindowsPath(path_value)
    if windows_path.drive:
        drive = windows_path.drive.rstrip(":").lower()
        wsl_path = Path("/mnt") / drive / Path(*windows_path.parts[1:])
        if wsl_path.exists():
            return wsl_path

    return path


def _stable_color_map(values: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.cm.get_cmap("tab20", max(1, len(values)))
    return {value: cmap(index) for index, value in enumerate(values)}


def _edge_relation_types(edge_data: dict) -> list[str]:
    if edge_data.get("predicate"):
        return [edge_data["predicate"]]
    return [
        rel.get("type", "unknown")
        for rel in edge_data.get("relations", [])
    ]


def _relationship_label(edge_data: dict) -> str:
    relation_types = [
        rel_type.replace("crm:", "")
        for rel_type in _edge_relation_types(edge_data)
    ]
    return ", ".join(relation_types)


def _node_category(data: dict) -> str:
    return (
        data.get("semantic_label")
        or data.get("local_class")
        or data.get("cidoc_class")
        or "unknown"
    )


def _matches_class_filter(data: dict, classes: list[str]) -> bool:
    requested = {item.lower() for item in classes}
    candidates = {
        str(data.get("semantic_label", "")).lower(),
        str(data.get("local_class", "")).lower(),
        str(data.get("cidoc_class", "")).lower(),
    }
    return bool(requested & candidates)


def _matches_relation_filter(edge_data: dict, relation_types: list[str]) -> bool:
    requested = {item.lower() for item in relation_types}
    candidates = set()
    for rel_type in _edge_relation_types(edge_data):
        rel_type = str(rel_type).lower()
        candidates.add(rel_type)
        candidates.add(rel_type.replace("crm:", ""))
    return bool(requested & candidates)


def _filter_graph(
    graph: nx.DiGraph,
    classes: list[str] | None,
    relation_types: list[str] | None,
    include_neighbors: bool,
) -> nx.DiGraph:
    filtered = graph.copy()

    if classes:
        selected_nodes = {
            node
            for node, data in graph.nodes(data=True)
            if _matches_class_filter(data, classes)
        }

        if include_neighbors:
            neighbor_nodes = set(selected_nodes)
            for node in selected_nodes:
                neighbor_nodes.update(graph.predecessors(node))
                neighbor_nodes.update(graph.successors(node))
            selected_nodes = neighbor_nodes

        filtered = filtered.subgraph(selected_nodes).copy()

    if relation_types:
        selected_edges = []
        for source, target, data in filtered.edges(data=True):
            if _matches_relation_filter(data, relation_types):
                selected_edges.append((source, target))

        edge_graph = nx.DiGraph()
        edge_graph.add_nodes_from(filtered.nodes(data=True))
        edge_graph.add_edges_from(
            (source, target, filtered[source][target])
            for source, target in selected_edges
        )
        filtered = edge_graph

    return filtered


def _as_draw_graph(graph: nx.DiGraph, directed: bool) -> nx.Graph | nx.DiGraph:
    if directed:
        return graph

    draw_graph = nx.Graph()
    draw_graph.add_nodes_from(graph.nodes(data=True))
    for source, target, data in graph.edges(data=True):
        if draw_graph.has_edge(source, target):
            existing = draw_graph[source][target].setdefault("relations", [])
            for relation in data.get("relations", []):
                if relation not in existing:
                    existing.append(relation)
            draw_graph[source][target]["relationship"] = _relationship_label(
                draw_graph[source][target]
            )
        else:
            draw_graph.add_edge(source, target, **data)
    return draw_graph


def _node_positions(graph: nx.Graph | nx.DiGraph, layout: str) -> dict:
    if layout == "spatial":
        positions = {}
        for node, data in graph.nodes(data=True):
            centroid = data.get("centroid")
            if centroid and len(centroid) >= 2:
                positions[node] = np.asarray(centroid[:2], dtype=float)
        if len(positions) == graph.number_of_nodes():
            return positions

    return nx.spring_layout(graph, seed=42)


def _annotation_rows_from_context(ctx) -> list[dict]:
    rows = []
    for object_name, annotations in getattr(ctx, "object_annotations", {}).items():
        for annotation in annotations:
            row = dict(annotation)
            row.setdefault("object_name", object_name)
            obj = ctx.objects.get(object_name, {})
            row.setdefault("semantic_label", obj.get("semantic_label", ""))
            rows.append(row)
    return rows


def _cidoc_scene_graph_to_networkx(scene_graph: CidocSceneGraph) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in scene_graph.nodes:
        graph.add_node(
            node.node_id,
            label=node.label,
            cidoc_class=node.cidoc_class,
            local_class=node.local_class,
            **node.properties,
        )

    for edge in scene_graph.edges:
        graph.add_edge(
            edge.source,
            edge.target,
            predicate=edge.predicate,
            relationship=edge.predicate,
        )
    return graph


def build_l3_graph_from_context(ctx) -> nx.DiGraph:
    if build_l3_cidoc_scene_graph is None:
        raise RuntimeError(
            "L3 CIDOC graph builder is not available in this repository version."
        )

    annotation_rows = _annotation_rows_from_context(ctx)
    if not annotation_rows:
        raise RuntimeError(
            "L3 CIDOC needs matched CSV annotations. Run with --annotation-csv "
            "or place a matching annotation CSV next to the LAZ file."
        )

    scene_id = Path(ctx.params.point_cloud_path).stem
    cidoc_scene_graph = build_l3_cidoc_scene_graph(scene_id, annotation_rows)
    return _cidoc_scene_graph_to_networkx(cidoc_scene_graph)


def visualize_scene_graph(
    graph: nx.DiGraph,
    level: str,
    classes: list[str] | None = None,
    relation_types: list[str] | None = None,
    include_neighbors: bool = False,
    directed: bool = False,
    layout: str = "spring",
    edge_labels: bool = False,
    node_size: int = 650,
    font_size: int = 8,
    output: str | None = None,
    show: bool = True,
) -> None:
    graph = _filter_graph(
        graph,
        classes=classes,
        relation_types=relation_types,
        include_neighbors=include_neighbors,
    )
    draw_graph = _as_draw_graph(graph, directed=directed)

    if draw_graph.number_of_nodes() == 0:
        print("No graph nodes to visualize after filtering.")
        return

    labels = {
        node: _node_category(data)
        for node, data in draw_graph.nodes(data=True)
    }
    unique_labels = sorted(set(labels.values()))
    color_map = _stable_color_map(unique_labels)
    node_colors = [
        color_map.get(labels.get(node, "unknown"), "salmon")
        for node in draw_graph.nodes()
    ]

    positions = _node_positions(draw_graph, layout=layout)

    plt.figure(figsize=(13, 9))
    nx.draw(
        draw_graph,
        positions,
        with_labels=True,
        node_color=node_colors,
        node_size=node_size,
        font_size=font_size,
        font_weight="bold",
        edge_color="#6b7280",
        arrows=directed,
        arrowsize=14,
        width=1.2,
    )

    if edge_labels:
        nx.draw_networkx_edge_labels(
            draw_graph,
            positions,
            edge_labels={
                (source, target): _relationship_label(data)
                for source, target, data in draw_graph.edges(data=True)
            },
            font_size=max(6, font_size - 1),
        )

    title = (
        f"Scene Graph {level}/{LEVEL_NAMES.get(level, 'all')} - "
        f"{draw_graph.number_of_nodes()} nodes, {draw_graph.number_of_edges()} edges"
    )
    plt.title(title)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[label],
            markersize=9,
            label=label,
        )
        for label in unique_labels
    ]
    if handles:
        plt.legend(handles=handles, loc="best", fontsize=8)

    plt.tight_layout()

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=180)
        print(f"Graph saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize arch-agent scene graphs with NetworkX and Matplotlib.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("point_cloud_path", help="Input .laz/.las point cloud.")
    parser.add_argument(
        "--level",
        choices=("L1", "L3"),
        default="L1",
        help="Graph level. L2 is CSV/descriptive metadata, not a graph.",
    )
    parser.add_argument("--classes", nargs="+", default=None, help="Show only selected semantic classes.")
    parser.add_argument("--include-neighbors", action="store_true", help="Include neighbors of selected classes.")
    parser.add_argument(
        "--relation-types",
        nargs="+",
        default=None,
        help="Show only selected relation types, for example: near adjacent_to supports.",
    )
    parser.add_argument("--layout", choices=("spring", "spatial"), default="spring", help="Graph layout.")
    parser.add_argument("--directed", action="store_true", help="Draw directional arrows.")
    parser.add_argument("--edge-labels", action="store_true", help="Draw relationship names on edges.")
    parser.add_argument("--node-size", type=int, default=650, help="Node size.")
    parser.add_argument("--font-size", type=int, default=8, help="Node label font size.")
    parser.add_argument("--output", default=None, help="Optional PNG output path.")
    parser.add_argument("--no-show", action="store_true", help="Save/prepare graph without opening a window.")
    parser.add_argument("--sample-n", type=int, default=150_000, help="Max points to load; use 0 for all points.")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN epsilon.")
    parser.add_argument("--min-samples", type=int, default=15, help="DBSCAN min_samples.")
    parser.add_argument("--distance-threshold", type=float, default=3.0, help="Relationship distance threshold.")
    parser.add_argument("--annotation-csv", default=None, help="Optional annotation CSV.")
    parser.add_argument(
        "--annotation-match-threshold",
        type=float,
        default=2.0,
        help="Max distance for matching CSV global_box_center to object AABB box_center.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    point_cloud_path = resolve_local_path(args.point_cloud_path)
    if not point_cloud_path.exists():
        raise FileNotFoundError(f"Point cloud path not found: {point_cloud_path}")

    annotation_csv = (
        str(resolve_local_path(args.annotation_csv))
        if args.annotation_csv
        else None
    )

    params = PipelineParams(
        point_cloud_path=str(point_cloud_path),
        sample_n=args.sample_n if args.sample_n > 0 else None,
        eps=args.eps,
        min_samples=args.min_samples,
        distance_threshold=args.distance_threshold,
        annotation_csv_path=annotation_csv,
        annotation_match_threshold=args.annotation_match_threshold,
    )
    ctx = run_pipeline(params)
    if args.level == "L1":
        graph = ctx.scene_graphs["L1"]
    else:
        graph = build_l3_graph_from_context(ctx)

    visualize_scene_graph(
        graph,
        level=args.level,
        classes=args.classes,
        relation_types=args.relation_types,
        include_neighbors=args.include_neighbors,
        directed=args.directed,
        layout=args.layout,
        edge_labels=args.edge_labels,
        node_size=args.node_size,
        font_size=args.font_size,
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
