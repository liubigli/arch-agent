import networkx as nx


ANNOTATION_NODE_FIELDS = ("material", "typology", "function", "description")


def build_scene_graph(
    objects: dict,
    relationships: list,
    features: dict,
    object_annotations: dict | None = None,
) -> nx.DiGraph:
    G = nx.DiGraph()
    object_annotations = object_annotations or {}

    for obj_name, obj_data in objects.items():
        node_attrs = features.get(obj_name, {}).copy()
        node_attrs.pop("semantic_label", None)
        node_attrs.pop("centroid", None)
        annotation_attrs = _annotation_node_attrs(object_annotations.get(obj_name, []))
        G.add_node(
            obj_name,
            semantic_label=obj_data["semantic_label"],
            centroid=obj_data["centroid"].tolist(),
            point_count=obj_data["point_count"],
            **node_attrs,
            **annotation_attrs,
        )

    for relationship in relationships:
        src, tgt, rel = relationship[:3]
        level = relationship[3] if len(relationship) > 3 else "geometric"
        _add_relation(G, src, tgt, rel, level)

    return G


def build_scene_graphs(
    objects: dict,
    relationship_layers: dict,
    features: dict,
    object_annotations: dict | None = None,
) -> dict[str, nx.DiGraph]:
    return {
        level: build_scene_graph(objects, relationships, features, object_annotations)
        for level, relationships in relationship_layers.items()
        if level != "all"
    }


def _annotation_node_attrs(annotations: list[dict]) -> dict:
    """Collapse matched CSV annotation rows into scene-graph node attributes."""
    attrs = {}
    for field in ANNOTATION_NODE_FIELDS:
        values = []
        for annotation in annotations:
            value = annotation.get(field)
            if value and value not in values:
                values.append(str(value))
        if values:
            attrs[field] = "; ".join(values)
    return attrs


def _add_relation(G: nx.DiGraph, src: str, tgt: str, relationship: str, level: str) -> None:
    item = {"type": relationship, "level": level}

    if G.has_edge(src, tgt):
        relations = G[src][tgt].setdefault("relations", [])
        if item not in relations:
            relations.append(item)
    else:
        G.add_edge(src, tgt, relations=[item])

    relations = G[src][tgt]["relations"]
    G[src][tgt]["relationships"] = [rel["type"] for rel in relations]
    G[src][tgt]["relationship_levels"] = [rel["level"] for rel in relations]
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
        "relationship_levels": {},
    }

    for _, data in G.nodes(data=True):
        lbl = data.get("semantic_label", "unknown")
        etype = data.get("element_type", "unknown")
        analysis["semantic_distribution"][lbl] = analysis["semantic_distribution"].get(lbl, 0) + 1
        analysis["element_type_distribution"][etype] = analysis["element_type_distribution"].get(etype, 0) + 1

    for _, _, data in G.edges(data=True):
        for rel in data.get("relations", []):
            rel_type = rel.get("type", "unknown")
            rel_level = rel.get("level", "unknown")
            analysis["relationship_types"][rel_type] = analysis["relationship_types"].get(rel_type, 0) + 1
            analysis["relationship_levels"][rel_level] = analysis["relationship_levels"].get(rel_level, 0) + 1

    return analysis
