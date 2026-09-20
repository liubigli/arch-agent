"""Assembly of the scene tool set.

The tools live in per-purpose modules; this module decides which of them the
agent actually receives. ``TOOL_ORDER`` pins the order they are presented to
the model in: the order is part of the prompt, so changing it changes model
behaviour and invalidates comparisons with earlier benchmark runs.
"""

from ..pipeline.pipeline import SceneContext
from .annotation_tools import create_annotation_tools
from .geometry_tools import create_geometry_tools
from .inventory_tools import create_inventory_tools
from .relationship_tools import create_relationship_tools
from .scene_state_tools import create_scene_state_tools

# Order in which tools are handed to the model. Do not reorder.
TOOL_ORDER = (
    "list_semantic_labels",
    "count_objects",
    "count_objects_by_class",
    "list_objects",
    "list_object_geometry",
    "get_object_info",
    "get_object_annotation",
    "list_csv_annotation_matches",
    "find_objects_by_material",
    "get_object_semantic_details",
    "find_relationships",
    "list_relationships",
    "find_relationship_anomalies",
    "get_scene_statistics",
    "find_sparse_objects",
    "get_point_cloud_info",
    "measure_scene_occupied_area",
    "measure_occupied_area",
    "estimate_room_volume",
    "measure_distance",
    "find_nearest_objects",
    "reload_scene",
)

BENCHMARK_TOOL_NAMES = {
    "get_scene_statistics",
    "list_semantic_labels",
    "count_objects",
    "count_objects_by_class",
    "list_objects",
    "get_object_info",
    "find_relationships",
    "list_relationships",
    "get_object_annotation",
    "get_object_semantic_details",
    "find_objects_by_material",
}

# Tools that expose CSV/user metadata (material, typology, function,
# description). Ablation condition "graph" removes them so the agent can only
# reach geometry and spatial relations.
CSV_TOOL_NAMES = {
    "get_object_annotation",
    "list_csv_annotation_matches",
    "find_objects_by_material",
    "get_object_semantic_details",
}

BENCHMARK_GRAPH_TOOL_NAMES = BENCHMARK_TOOL_NAMES - CSV_TOOL_NAMES


def create_scene_tools(ctx: SceneContext) -> list:
    """Return every scene tool, in TOOL_ORDER."""
    built = {}
    for factory in (
        create_inventory_tools,
        create_geometry_tools,
        create_annotation_tools,
        create_relationship_tools,
        create_scene_state_tools,
    ):
        for tool_item in factory(ctx):
            built[tool_item.name] = tool_item

    missing = set(TOOL_ORDER) - set(built)
    extra = set(built) - set(TOOL_ORDER)
    if missing or extra:
        raise RuntimeError(
            f"TOOL_ORDER is out of sync with the tool modules "
            f"(missing={sorted(missing)}, unexpected={sorted(extra)})"
        )
    return [built[name] for name in TOOL_ORDER]


def create_benchmark_scene_tools(
    ctx: SceneContext,
    allowed_names: set[str] | None = None,
) -> list:
    """Return the restricted tool set used by benchmark runs.

    The benchmark set keeps only tools that directly answer the official
    scene-understanding questions. Diagnostic, measurement, and pipeline
    mutation tools stay available in create_scene_tools for interactive use.

    ``allowed_names`` overrides the default set. Ablation runs pass
    BENCHMARK_GRAPH_TOOL_NAMES to withhold the CSV metadata tools.
    """
    allowed = BENCHMARK_TOOL_NAMES if allowed_names is None else allowed_names
    return [
        tool_item
        for tool_item in create_scene_tools(ctx)
        if tool_item.name in allowed
    ]
