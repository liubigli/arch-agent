import networkx as nx

GRAPH_LEVELS = {
    "L1": "geometric",
    "L2": "structural",
    "L3": "mereological",
}


def build_scene_graphs(
    objects: dict,
    stratified_relationships: dict,
    features: dict,
) -> dict[str, nx.DiGraph]:
    return {
        level_name: build_scene_graph(
            objects,
            relationships,
            features,
            graph_level=GRAPH_LEVELS[level_name],
        )
        for level_name, relationships in stratified_relationships.items()
        if level_name in GRAPH_LEVELS
    }


def build_scene_graph(
    objects: dict,
    relationships: list,
    features: dict,
    graph_level: str | None = None,
) -> nx.DiGraph:
    G = nx.DiGraph()
    if graph_level is not None:
        G.graph["level"] = graph_level

    for obj_name, obj_data in objects.items():
        node_attrs = features.get(obj_name, {}).copy()
        node_attrs.pop("semantic_label", None)
        node_attrs.pop("centroid", None)
        G.add_node(
            obj_name,
            semantic_label=obj_data["semantic_label"],
            centroid=obj_data["centroid"].tolist(),
            point_count=obj_data["point_count"],
            **node_attrs,
        )

    for src, tgt, rel in relationships:
        add_relation(G, src, tgt, rel)

    return G


def add_relation(G: nx.DiGraph, src: str, tgt: str, relationship: str) -> None:
    if G.has_edge(src, tgt):
        existing = G[src][tgt].setdefault("relationships", [])
        if relationship not in existing:
            existing.append(relationship)
    else:
        G.add_edge(src, tgt, relationships=[relationship])

    G[src][tgt]["relationship"] = ", ".join(G[src][tgt]["relationships"])


def analyze_scene_graph(G: nx.DiGraph) -> dict:
    n = G.number_of_nodes()
    avg_degree = sum(d for _, d in G.degree()) / n if n > 0 else 0.0

    analysis: dict = {
        "node_count": n,
        "edge_count": G.number_of_edges(),
        "connected_components": nx.number_weakly_connected_components(G),
        "avg_degree": avg_degree,
        "semantic_distribution": {},
        "element_type_distribution": {},
        "relationship_types": {},
    }

    for _, data in G.nodes(data=True):
        lbl = data.get("semantic_label", "unknown")
        etype = data.get("element_type", "unknown")
        analysis["semantic_distribution"][lbl] = analysis["semantic_distribution"].get(lbl, 0) + 1
        analysis["element_type_distribution"][etype] = analysis["element_type_distribution"].get(etype, 0) + 1

    for _, _, data in G.edges(data=True):
        relationships = data.get("relationships", [data.get("relationship", "unknown")])
        for rel in relationships:
            analysis["relationship_types"][rel] = analysis["relationship_types"].get(rel, 0) + 1

    return analysis
