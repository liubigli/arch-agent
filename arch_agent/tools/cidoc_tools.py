"""Tools that query the materialized CIDOC knowledge graph."""

from typing import Optional

from langchain_core.tools import tool

from ..pipeline.cidoc_context import build_cidoc_graph_from_context
from ..pipeline.pipeline import SceneContext
from ._shared import _canonical_semantic_label, _clean_optional


def create_cidoc_tools(ctx: SceneContext) -> list:
    @tool
    def query_cidoc_knowledge_graph(
        object_name: Optional[str] = None,
        semantic_label: Optional[str] = None,
        predicate: Optional[str] = None,
        direction: str = "both",
        limit: int = 40,
    ) -> str:
        """Query actual source-predicate-target triples in the CIDOC graph.

        Use this tool for explicit questions about the semantic knowledge
        graph, CIDOC triples, or semantic links between an architectural
        object and its material, typology, function, feature, or containing
        element. This tool builds and traverses the CIDOC graph; it does not
        treat CSV attributes or spatial relationships as if they were CIDOC
        triples.

        Args:
            object_name: Optional exact segmented object id, e.g. column_0.
            semantic_label: Optional scene class, e.g. column or moldings.
            predicate: Optional CIDOC predicate, e.g. crm:P45_consists_of.
                A short fragment such as P45 or consists_of is also accepted.
            direction: outgoing, incoming, or both relative to matched nodes.
            limit: Maximum number of triples returned. Use 0 to return all
                matches, capped at 200 rows for context safety.
        """
        object_name = _clean_optional(object_name)
        semantic_label = _canonical_semantic_label(semantic_label)
        predicate = _clean_optional(predicate)
        direction = (direction or "both").strip().lower()
        if direction not in {"outgoing", "incoming", "both"}:
            return "Invalid direction. Use outgoing, incoming, or both."
        if object_name and semantic_label:
            return "Provide only one of object_name or semantic_label."
        if object_name and object_name not in ctx.objects:
            return f"Object '{object_name}' is not present in the scene."
        if semantic_label:
            count = sum(
                obj.get("semantic_label") == semantic_label
                for obj in ctx.objects.values()
            )
            if not count:
                return (
                    f"Semantic class '{semantic_label}' is absent (0 objects). "
                    "No CIDOC triples can involve that scene class."
                )

        try:
            graph = build_cidoc_graph_from_context(ctx)
        except ValueError as exc:
            return f"CIDOC knowledge graph unavailable: {exc}"

        nodes = {node.node_id: node for node in graph.nodes}
        selected_ids = set(nodes)
        if object_name:
            selected_ids = {
                node.node_id
                for node in graph.nodes
                if node.properties.get("object_name") == object_name
            }
        elif semantic_label:
            selected_object_names = {
                name
                for name, obj in ctx.objects.items()
                if obj.get("semantic_label") == semantic_label
            }
            selected_ids = {
                node.node_id
                for node in graph.nodes
                if node.properties.get("object_name") in selected_object_names
            }

        predicate_key = predicate.casefold() if predicate else None
        matches = []
        for edge in graph.edges:
            if predicate_key and predicate_key not in edge.predicate.casefold():
                continue
            if object_name or semantic_label:
                outgoing = edge.source in selected_ids
                incoming = edge.target in selected_ids
                if direction == "outgoing" and not outgoing:
                    continue
                if direction == "incoming" and not incoming:
                    continue
                if direction == "both" and not (outgoing or incoming):
                    continue
            matches.append(edge)

        lines = [
            "CIDOC KNOWLEDGE GRAPH QUERY",
            f"Scene: {graph.scene_id}",
            f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} triples",
            f"Matched triples: {len(matches)}",
            "Source: materialized CIDOC graph built from matched CSV metadata; "
            "element-to-element semantic edges require spatial evidence.",
        ]
        if not selected_ids and (object_name or semantic_label):
            lines.append("No CIDOC node is linked to the requested scene object(s).")
            return "\n".join(lines)
        if not matches:
            lines.append("No matching CIDOC triples found.")
            return "\n".join(lines)

        effective_limit = 200 if limit <= 0 else min(limit, 200)
        lines.append("Triples:")
        for edge in matches[:effective_limit]:
            source = nodes[edge.source]
            target = nodes[edge.target]
            lines.append(
                f"- {_node_reference(source)} "
                f"--[{edge.predicate}]--> "
                f"{_node_reference(target)}"
            )
        hidden = len(matches) - effective_limit
        if hidden > 0:
            lines.append(f"... {hidden} additional triples not shown.")
        return "\n".join(lines)

    return [query_cidoc_knowledge_graph]


def _node_reference(node) -> str:
    object_name = node.properties.get("object_name", "")
    value = node.properties.get("value", "")
    primary = object_name or value or node.label
    return f"{primary} [node={node.node_id}; class={node.cidoc_class}]"
