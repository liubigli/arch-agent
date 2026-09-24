"""Build the CIDOC knowledge graph from an already computed scene context."""

from pathlib import Path

from .l3_cidoc_graph_builder import SceneGraph, build_scene_graph


def build_cidoc_graph_from_context(ctx) -> SceneGraph:
    """Build a CIDOC graph from matched CSV annotations and spatial evidence."""
    annotation_rows = []
    for object_name, annotations in getattr(ctx, "object_annotations", {}).items():
        obj = ctx.objects.get(object_name, {})
        for annotation in annotations:
            row = dict(annotation)
            row.setdefault("object_name", object_name)
            row.setdefault("semantic_label", obj.get("semantic_label", ""))
            annotation_rows.append(row)

    if not annotation_rows:
        raise ValueError(
            "No matched CSV annotations are available. The CIDOC knowledge "
            "graph requires an annotation CSV matched to scene objects."
        )

    spatial_rows = [
        {
            "source_object_name": source,
            "target_object_name": target,
            "relation_type": relationship,
        }
        for source, target, relationship, *_ in ctx.relationship_layers.get("L1", [])
    ]
    return build_scene_graph(
        Path(ctx.params.point_cloud_path).stem,
        annotation_rows,
        spatial_relations=spatial_rows,
    )
