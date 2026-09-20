"""Backwards-compatible shim.

The tools were split into per-purpose modules (inventory, geometry,
annotation, relationship, scene_state) assembled by ``registry``. This module
re-exports the previous public names so existing imports and local scripts
keep working. Import from ``arch_agent.tools`` in new code.
"""

from .registry import (
    BENCHMARK_GRAPH_TOOL_NAMES,
    BENCHMARK_TOOL_NAMES,
    CSV_TOOL_NAMES,
    TOOL_ORDER,
    create_benchmark_scene_tools,
    create_scene_tools,
)

__all__ = [
    "BENCHMARK_GRAPH_TOOL_NAMES",
    "BENCHMARK_TOOL_NAMES",
    "CSV_TOOL_NAMES",
    "TOOL_ORDER",
    "create_benchmark_scene_tools",
    "create_scene_tools",
]
