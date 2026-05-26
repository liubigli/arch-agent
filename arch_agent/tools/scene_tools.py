from collections import Counter
from typing import Optional

import networkx as nx
from langchain_core.tools import tool

from ..pipeline.pipeline import SceneContext, run_pipeline
from ..pipeline.graph import analyze_scene_graph
from ..settings import get_config


def _structural_elements() -> set[str]:
    return set(get_config()["semantic_classes"]["structural"])


def _classify_area(label_set: set) -> str:
    structural = _structural_elements()
    if {"vault", "arch", "column"} & label_set:
        return "vaulted_space"
    if "stairs" in label_set:
        return "vertical_circulation"
    if {"door_window", "wall"} & label_set:
        return "facade_zone"
    if {"floor", "wall"} & label_set:
        return "floor_zone"
    if label_set <= structural:
        return "structural_zone"
    return "general_area"


def create_scene_tools(ctx: SceneContext) -> list:

    @tool
    def list_objects() -> str:
        """List all objects detected in the scene, grouped by semantic class."""
        if not ctx.objects:
            return "No objects in the scene."
        by_class: dict[str, list] = {}
        for name, obj in ctx.objects.items():
            by_class.setdefault(obj["semantic_label"], []).append(
                (name, obj["point_count"])
            )
        lines = [f"Scene contains {len(ctx.objects)} objects:\n"]
        for lbl in sorted(by_class):
            lines.append(f"  {lbl.upper()} ({len(by_class[lbl])} instances):")
            for name, count in sorted(by_class[lbl]):
                lines.append(f"    - {name}: {count:,} points")
        return "\n".join(lines)

    @tool
    def get_object_info(object_name: str) -> str:
        """Get detailed geometric and semantic information about a specific object.

        Args:
            object_name: Name of the object, e.g. 'wall_0' or 'column_2'.
        """
        if object_name not in ctx.objects:
            sample = ", ".join(list(ctx.objects.keys())[:8])
            return f"Object '{object_name}' not found. Examples: {sample}"

        obj = ctx.objects[object_name]
        feat = ctx.features.get(object_name, {})
        c = obj["centroid"]
        dims = obj["bounds"]["max"] - obj["bounds"]["min"]

        return "\n".join([
            f"Object: {object_name}",
            f"  Semantic class  : {obj['semantic_label']}",
            f"  Element type    : {feat.get('element_type', 'unknown')}",
            f"  Point count     : {obj['point_count']:,}",
            f"  Centroid (x,y,z): ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})",
            f"  Dimensions (m)  : {dims[0]:.2f} × {dims[1]:.2f} × {dims[2]:.2f}",
            f"  Volume (AABB)   : {feat.get('volume', 0):.3f} m³",
            f"  Surface area    : {feat.get('surface_area', 0):.3f} m²",
            f"  Height          : {feat.get('height', 0):.2f} m",
            f"  Compactness     : {feat.get('compactness', 0):.4f}",
        ])

    @tool
    def find_relationships(object_name: str) -> str:
        """Find all spatial relationships involving a given object.

        Args:
            object_name: Name of the object to query.
        """
        G = ctx.scene_graph
        if object_name not in G:
            return f"Object '{object_name}' not found in the scene graph."

        out_edges = list(G.out_edges(object_name, data=True))
        in_edges = list(G.in_edges(object_name, data=True))
        lines = [f"Relationships for '{object_name}':"]

        if out_edges:
            lines.append("  As source:")
            for _, tgt, d in out_edges:
                lines.append(f"    {object_name} --[{d['relationship']}]--> {tgt}")
        if in_edges:
            lines.append("  As target:")
            for src, _, d in in_edges:
                lines.append(f"    {src} --[{d['relationship']}]--> {object_name}")
        if not out_edges and not in_edges:
            lines.append("  No relationships found (isolated node).")

        return "\n".join(lines)

    @tool
    def get_scene_statistics() -> str:
        """Get an overall statistical summary of the current scene."""
        G = ctx.scene_graph
        if G is None:
            return "Scene graph not available."

        structural = _structural_elements()
        analysis = analyze_scene_graph(G)
        struct = sum(1 for o in ctx.objects.values()
                     if o["semantic_label"] in structural)

        lines = [
            "Scene Statistics",
            "─" * 40,
            f"  Objects total      : {len(ctx.objects)}",
            f"    Structural        : {struct}",
            f"    Finishing         : {len(ctx.objects) - struct}",
            f"  Relationships      : {len(ctx.relationships)}",
            f"  Graph nodes/edges  : {analysis['node_count']} / {analysis['edge_count']}",
            f"  Connected components: {analysis['connected_components']}",
            f"  Avg node degree    : {analysis['avg_degree']:.2f}",
            "",
            "  Semantic distribution:",
        ]
        for lbl, cnt in sorted(analysis["semantic_distribution"].items()):
            lines.append(f"    {lbl:<15}: {cnt}")
        lines.append("  Relationship types:")
        for rel, cnt in sorted(analysis["relationship_types"].items()):
            lines.append(f"    {rel:<15}: {cnt}")
        lines.append("")
        lines.append("  Active pipeline params:")
        p = ctx.params
        lines.append(f"    csv            : {p.csv_path}")
        lines.append(f"    eps            : {p.eps}")
        lines.append(f"    min_samples    : {p.min_samples}")
        lines.append(f"    dist_threshold : {p.distance_threshold}")
        return "\n".join(lines)

    @tool
    def find_focal_points(top_n: int = 5) -> str:
        """Find the most spatially central elements in the scene graph.

        Args:
            top_n: Number of top elements to return (default: 5).
        """
        G = ctx.scene_graph
        nodes = list(G.nodes())
        if len(nodes) < 2:
            return "Not enough nodes to compute centrality."

        degree_c = nx.degree_centrality(G)
        between_c = nx.betweenness_centrality(G)
        scores = {n: degree_c[n] * 0.6 + between_c[n] * 0.4 for n in nodes}
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        lines = [f"Top {top_n} focal points:"]
        for rank, (node, score) in enumerate(top, 1):
            lbl = G.nodes[node].get("semantic_label", "?")
            lines.append(f"  {rank}. {node:<30} [{lbl}]  score={score:.4f}")
        return "\n".join(lines)

    @tool
    def find_pattern(element_a: str, element_b: str, relationship: str) -> str:
        """Search for pairs of architectural elements with a given spatial relationship.

        Args:
            element_a: Semantic class of the first element (e.g. 'arch', 'column').
            element_b: Semantic class of the second element (e.g. 'wall', 'floor').
            relationship: Relationship type to search for: near, adjacent, above, below, contains, inside.
        """
        G = ctx.scene_graph
        proximity = {"adjacent", "near"}
        vertical = {"above", "below"}
        matches = []

        for u, v, data in G.edges(data=True):
            rel = data.get("relationship", "")
            u_lbl = G.nodes[u].get("semantic_label", "")
            v_lbl = G.nodes[v].get("semantic_label", "")

            pair_match = (u_lbl == element_a and v_lbl == element_b) or \
                         (u_lbl == element_b and v_lbl == element_a)
            rel_match = (rel == relationship) or \
                        (rel in proximity and relationship in proximity) or \
                        (rel in vertical and relationship in vertical)

            if pair_match and rel_match:
                matches.append(f"  {u} [{u_lbl}] --[{rel}]--> {v} [{v_lbl}]")

        if not matches:
            return f"No matches for: {element_a} --[{relationship}]--> {element_b}"
        header = f"Found {len(matches)} match(es) for {element_a} --[{relationship}]--> {element_b}:"
        return header + "\n" + "\n".join(matches)

    @tool
    def discover_functional_areas() -> str:
        """Discover functional areas in the scene via Louvain community detection."""
        import networkx.algorithms.community as nx_comm

        G = ctx.scene_graph
        undirected = G.to_undirected()
        edges = [(u, v) for u, v, d in undirected.edges(data=True)
                 if d.get("relationship") != "contains"]

        if not edges:
            return "Not enough relationships for community detection."

        subgraph = undirected.edge_subgraph(edges)
        communities = nx_comm.louvain_communities(subgraph, seed=42)

        lines = [f"Discovered {len(communities)} functional areas:"]
        for i, community in enumerate(communities):
            objs = list(community)
            labels = [G.nodes[o]["semantic_label"] for o in objs]
            label_counts = dict(Counter(labels))
            function = _classify_area(set(labels))
            preview = ", ".join(sorted(objs)[:5]) + ("…" if len(objs) > 5 else "")
            lines.append(f"\n  Area {i}: {function}  ({len(objs)} objects)")
            lines.append(f"    Composition : {label_counts}")
            lines.append(f"    Objects     : {preview}")
        return "\n".join(lines)

    @tool
    def reload_scene(
        csv_path: Optional[str] = None,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
        distance_threshold: Optional[float] = None,
    ) -> str:
        """Reload and reprocess the scene, optionally changing pipeline parameters.

        All arguments are optional — omit any to keep the current value.

        Args:
            csv_path: Path to a new CSV point cloud file.
            eps: DBSCAN epsilon for object clustering.
            min_samples: DBSCAN min_samples for object clustering.
            distance_threshold: Max centroid distance (m) for spatial relationships.
        """
        if csv_path is not None:
            ctx.params.csv_path = csv_path
        if eps is not None:
            ctx.params.eps = eps
        if min_samples is not None:
            ctx.params.min_samples = min_samples
        if distance_threshold is not None:
            ctx.params.distance_threshold = distance_threshold

        new = run_pipeline(ctx.params)
        ctx.df = new.df
        ctx.objects = new.objects
        ctx.features = new.features
        ctx.relationships = new.relationships
        ctx.scene_graph = new.scene_graph

        return (
            f"Scene reloaded.\n"
            f"  Objects       : {len(ctx.objects)}\n"
            f"  Relationships : {len(ctx.relationships)}\n"
            f"  Graph         : {ctx.scene_graph.number_of_nodes()} nodes, "
            f"{ctx.scene_graph.number_of_edges()} edges\n"
            f"  Params        : eps={ctx.params.eps}, min_samples={ctx.params.min_samples}, "
            f"distance_threshold={ctx.params.distance_threshold}"
        )

    return [
        list_objects,
        get_object_info,
        find_relationships,
        get_scene_statistics,
        find_focal_points,
        find_pattern,
        discover_functional_areas,
        reload_scene,
    ]
