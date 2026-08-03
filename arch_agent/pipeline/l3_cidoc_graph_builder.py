from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CIDOC_E22 = "crm:E22_Human-Made_Object"
CIDOC_E26 = "crm:E26_Physical_Feature"
CIDOC_E55 = "crm:E55_Type"
CIDOC_E57 = "crm:E57_Material"
CIDOC_E16 = "crm:E16_Measurement"
CIDOC_E54 = "crm:E54_Dimension"

P2_HAS_TYPE = "crm:P2_has_type"
P45_CONSISTS_OF = "crm:P45_consists_of"
P103_WAS_INTENDED_FOR = "crm:P103_was_intended_for"
P46_IS_COMPOSED_OF = "crm:P46_is_composed_of"
P46I_FORMS_PART_OF = "crm:P46i_forms_part_of"
P56_BEARS_FEATURE = "crm:P56_bears_feature"
P56I_IS_FOUND_ON = "crm:P56i_is_found_on"
P39_MEASURED = "crm:P39_measured"
P43_HAS_DIMENSION = "crm:P43_has_dimension"
P90_HAS_VALUE = "crm:P90_has_value"

FEATURE_CLASSES = {"molding", "moulding", "cornice", "decoration", "decorative_feature"}

POINT_CLOUD_CLASS_REGISTRY = {
    "column": {
        "node_prefix": "Colonna",
        "display_label": "Colonna",
        "cidoc_class": CIDOC_E22,
        "local_class": "Column",
    },
    "colonnade": {
        "node_prefix": "Colonnato",
        "display_label": "Colonnato",
        "cidoc_class": CIDOC_E22,
        "local_class": "Colonnade",
    },
    "portico": {
        "node_prefix": "Portico",
        "display_label": "Portico/Porticato",
        "cidoc_class": CIDOC_E22,
        "local_class": "Portico",
    },
    "loggia": {
        "node_prefix": "Loggia",
        "display_label": "Loggia/Loggiato",
        "cidoc_class": CIDOC_E22,
        "local_class": "Loggia",
    },
    "wall": {
        "node_prefix": "Muro",
        "display_label": "Muro",
        "cidoc_class": CIDOC_E22,
        "local_class": "Wall",
    },
    "vault": {
        "node_prefix": "Volta",
        "display_label": "Volta",
        "cidoc_class": CIDOC_E22,
        "local_class": "Vault",
    },
    "roof": {
        "node_prefix": "Roof",
        "display_label": "Roof",
        "cidoc_class": CIDOC_E22,
        "local_class": "Roof",
    },
    "floor": {
        "node_prefix": "Pavimento",
        "display_label": "Pavimento",
        "cidoc_class": CIDOC_E22,
        "local_class": "Floor",
    },
    "stair": {
        "node_prefix": "Scala",
        "display_label": "Scala",
        "cidoc_class": CIDOC_E22,
        "local_class": "Stair",
    },
    "door_window": {
        "node_prefix": "PortaFinestra",
        "display_label": "Porta/Finestra",
        "cidoc_class": CIDOC_E22,
        "local_class": "DoorWindow",
    },
    "molding": {
        "node_prefix": "Modanatura",
        "display_label": "Modanatura",
        "cidoc_class": CIDOC_E26,
        "local_class": "Molding",
    },
    "arch": {
        "node_prefix": "Arco",
        "display_label": "Arco",
        "cidoc_class": CIDOC_E22,
        "local_class": "Arch",
    },
    "architrave": {
        "node_prefix": "Architrave",
        "display_label": "Architrave",
        "cidoc_class": CIDOC_E22,
        "local_class": "Architrave",
    },
    "pillar": {
        "node_prefix": "Pilastro",
        "display_label": "Pilastro",
        "cidoc_class": CIDOC_E22,
        "local_class": "Pillar",
    },
    "rib": {
        "node_prefix": "Costolone",
        "display_label": "Costolone",
        "cidoc_class": CIDOC_E22,
        "local_class": "Rib",
    },
    "ramp": {
        "node_prefix": "Rampa",
        "display_label": "Rampa",
        "cidoc_class": CIDOC_E22,
        "local_class": "Ramp",
    },
    "step": {
        "node_prefix": "Gradino",
        "display_label": "Gradino",
        "cidoc_class": CIDOC_E22,
        "local_class": "Step",
    },
}

CIDOC_SATELLITE_LABELLING = {
    "typology": {
        "node_prefix": "Tipo",
        "cidoc_class": CIDOC_E55,
        "predicate": P2_HAS_TYPE,
    },
    "material": {
        "node_prefix": "Materiale",
        "cidoc_class": CIDOC_E57,
        "predicate": P45_CONSISTS_OF,
    },
    "function": {
        "node_prefix": "Funzione",
        "cidoc_class": CIDOC_E55,
        "predicate": P103_WAS_INTENDED_FOR,
    },
}

ELEMENT_RELATION_PATTERNS = {
    "column": {
        "typology": P2_HAS_TYPE,
        "material": P45_CONSISTS_OF,
        "function": P103_WAS_INTENDED_FOR,
        "optional_context": (P46I_FORMS_PART_OF,),
    },
    "colonnade": {
        "components": (P46_IS_COMPOSED_OF,),
    },
    "portico": {
        "typology": P2_HAS_TYPE,
        "function": P103_WAS_INTENDED_FOR,
        "components": (P46_IS_COMPOSED_OF,),
    },
    "loggia": {
        "typology": P2_HAS_TYPE,
        "function": P103_WAS_INTENDED_FOR,
        "components": (P46_IS_COMPOSED_OF,),
    },
    "wall": {
        "typology": P2_HAS_TYPE,
        "material": P45_CONSISTS_OF,
        "features": (P56_BEARS_FEATURE,),
    },
    "vault": {
        "typology": P2_HAS_TYPE,
        "material": P45_CONSISTS_OF,
        "function": P103_WAS_INTENDED_FOR,
        "context": (P46I_FORMS_PART_OF,),
        "components": (P46_IS_COMPOSED_OF,),
    },
    "stair": {
        "material": P45_CONSISTS_OF,
        "function": P103_WAS_INTENDED_FOR,
        "components": (P46_IS_COMPOSED_OF,),
    },
    "door_window": {
        "typology": P2_HAS_TYPE,
        "material": P45_CONSISTS_OF,
        "function": P103_WAS_INTENDED_FOR,
        "context": (P46I_FORMS_PART_OF,),
    },
    "molding": {
        "typology": P2_HAS_TYPE,
        "material": P45_CONSISTS_OF,
        "context": (P56I_IS_FOUND_ON,),
    },
}


@dataclass(frozen=True)
class Node:
    node_id: str
    label: str
    cidoc_class: str
    local_class: str
    properties: dict[str, str]


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    predicate: str


@dataclass(frozen=True)
class SceneGraph:
    scene_id: str
    nodes: list[Node]
    edges: list[Edge]


def build_scene_graph(
    scene_id: str,
    annotations: Iterable[dict[str, str]],
    measurements: Iterable[dict[str, str]] | None = None,
    contextual_relations: Iterable[dict[str, str]] | None = None,
    spatial_relations: Iterable[dict[str, str]] | None = None,
    scene_evidence: Iterable[dict[str, str]] | None = None,
    *,
    add_door_window_wall_relation: bool = True,
    add_wall_molding_feature_relation: bool = True,
    infer_colonnades: bool = True,
    infer_porticoes: bool = True,
    infer_loggias: bool = True,
) -> SceneGraph:
    """Build a CIDOC L3 graph for a single point-cloud scene.

    The same annotation schema can be reused scene by scene. `scene_id`
    namespaces scene-local element identifiers, so `Colonna_1` in two
    different scenes does not collide.
    """

    scene_id = _slug(scene_id)
    raw_nodes, edges = build_l3_cidoc_graph(annotations)
    nodes = [_namespace_scene_node(scene_id, node) for node in raw_nodes]
    id_map = {raw.node_id: namespaced.node_id for raw, namespaced in zip(raw_nodes, nodes)}
    edges = [_namespace_scene_edge(id_map, edge) for edge in edges]

    spatial_index = build_spatial_relation_index(spatial_relations or ())
    if add_door_window_wall_relation:
        nodes, edges = add_door_window_wall_part_of_edges(nodes, edges, spatial_index=spatial_index)
    if add_wall_molding_feature_relation:
        nodes, edges = add_wall_molding_feature_edges(nodes, edges, spatial_index=spatial_index)
    nodes, edges = add_vault_roof_part_of_edges(nodes, edges, spatial_index=spatial_index)
    nodes, edges = add_vault_arch_composition_edges(nodes, edges, spatial_index=spatial_index)
    if infer_colonnades:
        nodes, edges = add_colonnade_nodes(scene_id, nodes, edges)
    if infer_porticoes:
        nodes, edges = add_portico_nodes(
            scene_id,
            nodes,
            edges,
            scene_evidence or (),
            spatial_index=spatial_index,
        )
    if infer_loggias:
        nodes, edges = add_loggia_nodes(
            scene_id,
            nodes,
            edges,
            scene_evidence or (),
            spatial_index=spatial_index,
        )
    if contextual_relations:
        nodes, edges = add_contextual_element_relations(
            nodes,
            edges,
            contextual_relations,
            spatial_index=spatial_index,
        )
    if measurements:
        nodes, edges = add_measurement_dimension_nodes(scene_id, nodes, edges, measurements)

    return SceneGraph(scene_id=scene_id, nodes=nodes, edges=edges)


def build_l3_cidoc_graph(annotation_rows: Iterable[dict[str, str]]) -> tuple[list[Node], list[Edge]]:
    """Build an L3 CIDOC graph from semantic annotation rows.

    Expected row fields are flexible, but the core columns are:
    - class_semantic_label or semantic_label
    - global_box_center_x/y/z or x/y/z
    - material
    - typology
    - function
    - description

    Material values are accepted only from the annotation rows attached to the
    scene. The builder does not infer materials from point-cloud geometry,
    visual features, semantic class, L1, or HBIM data.

    L3 is intentionally not mereological here. Each architectural element is
    connected to CIDOC satellite nodes:
    - typology: crm:E55_Type via crm:P2_has_type
    - material: crm:E57_Material via crm:P45_consists_of
    - function: crm:E55_Type via crm:P103_was_intended_for
    """

    nodes_by_id: dict[str, Node] = {}
    edges: list[Edge] = []
    element_counters: dict[str, int] = {}

    for row in annotation_rows:
        normalized = _normalize_row(row)
        local_class = normalized.get("class_semantic_label") or normalized.get("semantic_label")
        if not local_class:
            continue

        local_class = _normalize_class(local_class)
        element_counters[local_class] = element_counters.get(local_class, 0) + 1
        class_def = POINT_CLOUD_CLASS_REGISTRY.get(local_class, _fallback_class_def(local_class))
        element_id = f"{class_def['node_prefix']}_{element_counters[local_class]}"

        element_node = Node(
            node_id=element_id,
            label=element_id,
            cidoc_class=class_def["cidoc_class"],
            local_class=class_def["local_class"],
            properties={
                "display_label": class_def["display_label"],
                "global_box_center_x": _first_value(normalized, "global_box_center_x", "x"),
                "global_box_center_y": _first_value(normalized, "global_box_center_y", "y"),
                "global_box_center_z": _first_value(normalized, "global_box_center_z", "z"),
                "position": normalized.get("position", ""),
                "description": normalized.get("description", ""),
            },
        )
        nodes_by_id[element_id] = element_node

        typology = normalized.get("typology", "")
        if typology:
            satellite_def = CIDOC_SATELLITE_LABELLING["typology"]
            type_id = f"{satellite_def['node_prefix']}_{_slug(typology)}"
            nodes_by_id.setdefault(
                type_id,
                Node(type_id, typology, satellite_def["cidoc_class"], "typology", {"value": typology}),
            )
            edges.append(Edge(element_id, type_id, satellite_def["predicate"]))

        material = normalized.get("material", "")
        if material:
            satellite_def = CIDOC_SATELLITE_LABELLING["material"]
            material_id = f"{satellite_def['node_prefix']}_{_slug(material)}"
            nodes_by_id.setdefault(
                material_id,
                Node(material_id, material, satellite_def["cidoc_class"], "material", {"value": material}),
            )
            edges.append(Edge(element_id, material_id, satellite_def["predicate"]))

        functions = _split_multi_value(normalized.get("function", ""))
        for function in functions:
            satellite_def = CIDOC_SATELLITE_LABELLING["function"]
            function_id = f"{satellite_def['node_prefix']}_{_slug(function)}"
            nodes_by_id.setdefault(
                function_id,
                Node(function_id, function, satellite_def["cidoc_class"], "function", {"value": function}),
            )
            edges.append(Edge(element_id, function_id, satellite_def["predicate"]))

    return list(nodes_by_id.values()), edges


def add_physical_composition_edges(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    whole_local_class: str,
    part_local_classes: Iterable[str],
    *,
    inverse: bool = False,
    spatial_index: set[tuple[str, str]] | None = None,
) -> tuple[list[Node], list[Edge]]:
    """Add CIDOC P46/P46i physical composition links between existing elements.

    Use this for contextual architectural composition. For the current
    point-cloud labelling pattern, use the inverse direction:
    - PortaFinestra_1 crm:P46i_forms_part_of Muro_1

    This remains secondary to the L3 CIDOC labelling model, whose primary
    structure is element -> typology/material/function.
    """

    node_list = list(nodes)
    edge_list = list(edges)
    whole = _find_first_element(node_list, whole_local_class)
    if whole is None:
        return node_list, edge_list

    for part_local_class in part_local_classes:
        for part in _find_elements(node_list, part_local_class):
            if not _are_spatially_related(whole.node_id, part.node_id, spatial_index):
                continue
            if inverse:
                edge_list.append(Edge(part.node_id, whole.node_id, P46I_FORMS_PART_OF))
            else:
                edge_list.append(Edge(whole.node_id, part.node_id, P46_IS_COMPOSED_OF))

    return node_list, edge_list


def add_door_window_wall_part_of_edges(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    *,
    spatial_index: set[tuple[str, str]] | None = None,
) -> tuple[list[Node], list[Edge]]:
    """Apply the simple CIDOC pattern: door_window P46i forms part of wall.

    DoorWindow is an umbrella class for doors, windows and door-windows.
    Both DoorWindow and Wall are modeled as crm:E22_Human-Made_Object.
    The relation direction is always:
    - PortaFinestra_N crm:P46i_forms_part_of Muro_1
    """

    return add_physical_composition_edges(
        nodes,
        edges,
        whole_local_class="wall",
        part_local_classes=("door_window",),
        inverse=True,
        spatial_index=spatial_index,
    )


def add_wall_molding_feature_edges(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    *,
    spatial_index: set[tuple[str, str]] | None = None,
) -> tuple[list[Node], list[Edge]]:
    """Apply CIDOC pattern: Wall P56 bears feature Molding."""

    node_list = list(nodes)
    edge_list = list(edges)
    wall = _find_first_element(node_list, "wall")
    if wall is None:
        return node_list, edge_list

    for molding in _find_elements(node_list, "molding"):
        if not _are_spatially_related(wall.node_id, molding.node_id, spatial_index):
            continue
        edge_list.append(Edge(wall.node_id, molding.node_id, P56_BEARS_FEATURE))

    return node_list, edge_list


def add_molding_found_on_edges(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    target_local_classes: Iterable[str] = ("wall", "vault"),
    *,
    spatial_index: set[tuple[str, str]] | None = None,
) -> tuple[list[Node], list[Edge]]:
    """Apply CIDOC pattern: Molding P56i is found on Wall/Vault."""

    node_list = list(nodes)
    edge_list = list(edges)
    target_nodes: list[Node] = []
    for target_local_class in target_local_classes:
        target_nodes.extend(_find_elements(node_list, target_local_class))

    for molding in _find_elements(node_list, "molding"):
        for target in target_nodes:
            if not _are_spatially_related(molding.node_id, target.node_id, spatial_index):
                continue
            edge_list.append(Edge(molding.node_id, target.node_id, P56I_IS_FOUND_ON))

    return node_list, edge_list


def add_vault_roof_part_of_edges(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    *,
    spatial_index: set[tuple[str, str]] | None = None,
) -> tuple[list[Node], list[Edge]]:
    """Apply CIDOC pattern: Vault P46i forms part of Roof."""

    return add_physical_composition_edges(
        nodes,
        edges,
        whole_local_class="roof",
        part_local_classes=("vault",),
        inverse=True,
        spatial_index=spatial_index,
    )


def add_vault_arch_composition_edges(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    *,
    spatial_index: set[tuple[str, str]] | None = None,
) -> tuple[list[Node], list[Edge]]:
    """Apply CIDOC pattern: Vault P46 is composed of Arch."""

    return add_physical_composition_edges(
        nodes,
        edges,
        whole_local_class="vault",
        part_local_classes=("arch",),
        inverse=False,
        spatial_index=spatial_index,
    )


def add_colonnade_nodes(
    scene_id: str,
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    *,
    min_columns: int = 4,
    spacing_tolerance: float = 0.35,
) -> tuple[list[Node], list[Edge]]:
    """Create a Colonnade node only when explicit geometric/semantic evidence exists.

    Rule:
    - at least four columns;
    - columns have available Global box center coordinates;
    - columns form an approximately equispaced sequence;
    - columns have explicit support function through P103.

    The generated pattern is:
    - Colonnato_N a crm:E22_Human-Made_Object
    - Colonnato_N crm:P46_is_composed_of Colonna_N
    """

    node_list = list(nodes)
    edge_list = list(edges)
    columns = _find_elements(node_list, "column")
    candidate_columns = [column for column in columns if _node_xy(column) is not None]

    if len(candidate_columns) < min_columns:
        return node_list, edge_list
    if not _columns_have_structural_support_function(candidate_columns, node_list, edge_list):
        return node_list, edge_list

    sequence = _equispaced_column_sequence(candidate_columns, spacing_tolerance)
    if len(sequence) < min_columns:
        return node_list, edge_list

    colonnade_id = f"{scene_id}_Colonnato_1"
    if any(node.node_id == colonnade_id for node in node_list):
        return node_list, edge_list

    colonnade_node = Node(
        node_id=colonnade_id,
        label="Colonnato_1",
        cidoc_class=CIDOC_E22,
        local_class="Colonnade",
        properties={
            "scene_id": scene_id,
            "display_label": "Colonnato",
            "rule": "at_least_4_equispaced_structural_support_columns",
            "evidence": "global_box_center_coordinates_and_P103_support_function",
        },
    )
    node_list.append(colonnade_node)

    for column in sequence:
        edge_list.append(Edge(colonnade_id, column.node_id, P46_IS_COMPOSED_OF))

    return node_list, edge_list


def add_portico_nodes(
    scene_id: str,
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    scene_evidence: Iterable[dict[str, str]],
    *,
    spatial_index: set[tuple[str, str]] | None = None,
    min_arches: int = 2,
    min_columns: int = 2,
    spacing_tolerance: float = 0.35,
) -> tuple[list[Node], list[Edge]]:
    """Create a Portico/Porticato node only with explicit spatial and scene evidence.

    Rule:
    - aligned arches, or aligned columns plus architraves;
    - explicit evidence of a continuous cover above;
    - explicit evidence of covered walkable space behind the arches;
    - explicit evidence that the space is at ground floor;
    - explicit evidence that the arch side is open to an external space;
    - explicit evidence that the opposite side is attached to a building or closed by walls.

    If cover, covered space or ground-floor evidence is missing or false, no
    Portico node is created.
    """

    node_list = list(nodes)
    edge_list = list(edges)
    evidence = _find_scene_evidence(scene_evidence, {"portico", "porticato"})
    if evidence is None or not _portico_scene_evidence_allows(evidence):
        return node_list, edge_list

    arcade_sequence = _aligned_portico_arcade_sequence(
        node_list,
        min_arches=min_arches,
        min_columns=min_columns,
        spacing_tolerance=spacing_tolerance,
        spatial_index=spatial_index,
    )
    if not arcade_sequence:
        return node_list, edge_list

    portico_id = evidence.get("node_id", "") or f"{scene_id}_Portico_1"
    if any(node.node_id == portico_id for node in node_list):
        return node_list, edge_list

    portico_node = Node(
        node_id=portico_id,
        label=evidence.get("label", "") or "Portico_1",
        cidoc_class=CIDOC_E22,
        local_class="Portico",
        properties={
            "scene_id": scene_id,
            "display_label": "Portico/Porticato",
            "rule": "aligned_arcade_continuous_cover_ground_floor_open_external_side",
            "description": evidence.get(
                "description",
                "Insieme classificato come portico/porticato sulla base di evidenze L1/L2 esplicite.",
            ),
            "evidence_source": evidence.get("source", ""),
        },
    )
    node_list.append(portico_node)

    type_node_id = "Tipo_PorticoPorticato"
    if not any(node.node_id == type_node_id for node in node_list):
        node_list.append(
            Node(
                type_node_id,
                "Portico/Porticato",
                CIDOC_E55,
                "typology",
                {"value": "Portico/Porticato"},
            )
        )
    edge_list.append(Edge(portico_id, type_node_id, P2_HAS_TYPE))

    for component in arcade_sequence:
        edge_list.append(Edge(portico_id, component.node_id, P46_IS_COMPOSED_OF))

    for cover in _find_cover_elements(node_list):
        if _any_spatial_relation(cover.node_id, [component.node_id for component in arcade_sequence], spatial_index):
            edge_list.append(Edge(portico_id, cover.node_id, P46_IS_COMPOSED_OF))

    return node_list, edge_list


def add_loggia_nodes(
    scene_id: str,
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    scene_evidence: Iterable[dict[str, str]],
    *,
    spatial_index: set[tuple[str, str]] | None = None,
    min_arches: int = 2,
    min_supports: int = 2,
    spacing_tolerance: float = 0.35,
) -> tuple[list[Node], list[Edge]]:
    """Create a Loggia/Loggiato node only with explicit spatial and functional evidence.

    Rule:
    - covered room/gallery;
    - at least one side fully open to the outside;
    - open side made of arches on columns or pillars;
    - space inserted in, or part of, the building volume;
    - intermediate/representative function between inside and outside;
    - can be at ground floor or upper floors.

    If the scene is ground floor and the function is direct building access,
    the preferred classification is Portico, so no Loggia node is created.
    """

    node_list = list(nodes)
    edge_list = list(edges)
    evidence = _find_scene_evidence(scene_evidence, {"loggia", "loggiato"})
    if evidence is None or not _loggia_scene_evidence_allows(evidence):
        return node_list, edge_list

    arcade_sequence = _aligned_open_side_sequence(
        node_list,
        min_arches=min_arches,
        min_supports=min_supports,
        spacing_tolerance=spacing_tolerance,
        spatial_index=spatial_index,
    )
    if not arcade_sequence:
        return node_list, edge_list

    loggia_id = evidence.get("node_id", "") or f"{scene_id}_Loggia_1"
    if any(node.node_id == loggia_id for node in node_list):
        return node_list, edge_list

    loggia_node = Node(
        node_id=loggia_id,
        label=evidence.get("label", "") or "Loggia_1",
        cidoc_class=CIDOC_E22,
        local_class="Loggia",
        properties={
            "scene_id": scene_id,
            "display_label": "Loggia/Loggiato",
            "rule": "covered_room_open_side_arcades_integrated_building_intermediate_function",
            "description": evidence.get(
                "description",
                "Insieme classificato come loggia/loggiato sulla base di evidenze L1/L2 esplicite.",
            ),
            "evidence_source": evidence.get("source", ""),
        },
    )
    node_list.append(loggia_node)

    type_node_id = "Tipo_LoggiaLoggiato"
    if not any(node.node_id == type_node_id for node in node_list):
        node_list.append(
            Node(
                type_node_id,
                "Loggia/Loggiato",
                CIDOC_E55,
                "typology",
                {"value": "Loggia/Loggiato"},
            )
        )
    edge_list.append(Edge(loggia_id, type_node_id, P2_HAS_TYPE))

    function_node_id = "Funzione_SpazioIntermedioRappresentativo"
    if not any(node.node_id == function_node_id for node in node_list):
        node_list.append(
            Node(
                function_node_id,
                "Spazio intermedio/rappresentativo tra interno ed esterno",
                CIDOC_E55,
                "function",
                {"value": "Spazio intermedio/rappresentativo tra interno ed esterno"},
            )
        )
    edge_list.append(Edge(loggia_id, function_node_id, P103_WAS_INTENDED_FOR))

    for component in arcade_sequence:
        edge_list.append(Edge(loggia_id, component.node_id, P46_IS_COMPOSED_OF))

    for cover in _find_cover_elements(node_list):
        if _any_spatial_relation(cover.node_id, [component.node_id for component in arcade_sequence], spatial_index):
            edge_list.append(Edge(loggia_id, cover.node_id, P46_IS_COMPOSED_OF))

    return node_list, edge_list


def add_contextual_element_relations(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    relations: Iterable[dict[str, str]],
    *,
    spatial_index: set[tuple[str, str]] | None = None,
) -> tuple[list[Node], list[Edge]]:
    """Attach explicit element-to-element CIDOC relations from input data.

    Expected relation row fields:
    - source_node_id, or source_local_class + source_index
    - target_node_id, or target_local_class + target_index
    - predicate, one of the supported CIDOC predicates such as:
      crm:P46_is_composed_of, crm:P46i_forms_part_of,
      crm:P56_bears_feature, crm:P56i_is_found_on

    No relation is created if source, target, or predicate is missing/invalid.
    """

    node_list = list(nodes)
    edge_list = list(edges)

    for relation in relations:
        normalized = _normalize_row(relation)
        predicate = _normalize_predicate(normalized.get("predicate", ""))
        if predicate not in _allowed_element_relation_predicates():
            continue

        source = _resolve_relation_endpoint(node_list, normalized, "source")
        target = _resolve_relation_endpoint(node_list, normalized, "target")
        if source is None or target is None:
            continue
        if not _relation_has_spatial_support(normalized, source.node_id, target.node_id, spatial_index):
            continue

        edge_list.append(Edge(source.node_id, target.node_id, predicate))

    return node_list, edge_list


def build_spatial_relation_index(spatial_relations: Iterable[dict[str, str]]) -> set[tuple[str, str]]:
    """Build an undirected spatial support index from L1 scenegraph relations.

    Expected fields:
    - source_node_id
    - target_node_id
    - relation_type, e.g. near, adjacent, touches, intersects, contains

    CIDOC element-to-element relations should be created only when there is
    spatial support from L1 or when an input relation explicitly marks itself
    as spatially supported.
    """

    supported_relation_types = {"near", "adjacent", "touches", "intersects", "contains", "within", "overlaps"}
    index: set[tuple[str, str]] = set()
    for row in spatial_relations:
        normalized = _normalize_row(row)
        relation_type = normalized.get("relation_type", "").lower()
        if relation_type and relation_type not in supported_relation_types:
            continue
        source = normalized.get("source_node_id", "")
        target = normalized.get("target_node_id", "")
        if source and target:
            index.add((source, target))
            index.add((target, source))
    return index


def add_measurement_dimension_nodes(
    scene_id: str,
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    measurements: Iterable[dict[str, str]],
) -> tuple[list[Node], list[Edge]]:
    """Attach L1/spatial-tool measurements to scene elements using CIDOC.

    Expected measurement row fields:
    - target_node_id, or target_local_class + target_index
    - dimension_type, e.g. height, thickness, arch_radius
    - value
    - unit, e.g. m, cm

    Pattern:
    - Measurement_* a crm:E16_Measurement
    - Measurement_* crm:P39_measured Element_N
    - Measurement_* crm:P43_has_dimension Dimension_*
    - Dimension_* a crm:E54_Dimension with numeric value and unit
    """

    node_list = list(nodes)
    edge_list = list(edges)
    nodes_by_id = {node.node_id: node for node in node_list}

    for index, row in enumerate(measurements, start=1):
        normalized = _normalize_row(row)
        target = _resolve_measurement_target(node_list, normalized)
        if target is None:
            continue

        dimension_type = normalized.get("dimension_type", "dimension")
        value = normalized.get("value", "")
        unit = normalized.get("unit", "")
        measurement_id = normalized.get(
            "measurement_id",
            f"{scene_id}_Measurement_{target.node_id}_{_slug(dimension_type)}_{index}",
        )
        dimension_id = normalized.get(
            "dimension_id",
            f"{scene_id}_Dimension_{target.node_id}_{_slug(dimension_type)}_{index}",
        )

        if measurement_id not in nodes_by_id:
            measurement_node = Node(
                node_id=measurement_id,
                label=measurement_id,
                cidoc_class=CIDOC_E16,
                local_class="Measurement",
                properties={
                    "scene_id": scene_id,
                    "dimension_type": dimension_type,
                    "source": normalized.get("source", ""),
                },
            )
            node_list.append(measurement_node)
            nodes_by_id[measurement_id] = measurement_node

        if dimension_id not in nodes_by_id:
            dimension_node = Node(
                node_id=dimension_id,
                label=dimension_id,
                cidoc_class=CIDOC_E54,
                local_class="Dimension",
                properties={
                    "scene_id": scene_id,
                    "dimension_type": dimension_type,
                    P90_HAS_VALUE: value,
                    "unit": unit,
                },
            )
            node_list.append(dimension_node)
            nodes_by_id[dimension_id] = dimension_node

        edge_list.append(Edge(measurement_id, target.node_id, P39_MEASURED))
        edge_list.append(Edge(measurement_id, dimension_id, P43_HAS_DIMENSION))

    return node_list, edge_list


def read_annotation_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {str(key).strip().strip('"'): str(value or "").strip().strip('"') for key, value in row.items()}


def _normalize_class(value: str) -> str:
    value = value.strip().lower().replace("/", "_").replace(" ", "_")
    aliases = {
        "door": "door_window",
        "porta": "door_window",
        "finestra": "door_window",
        "window": "door_window",
        "window_door": "door_window",
        "doorwindow": "door_window",
        "porta_finestra": "door_window",
        "portafinestra": "door_window",
        "moulding": "molding",
        "moldings": "molding",
        "modanatura": "molding",
        "modanature": "molding",
        "arches": "arch",
        "arco": "arch",
        "archi": "arch",
        "arc": "arch",
        "arcade_arch": "arch",
        "architravi": "architrave",
        "architraves": "architrave",
        "ribs": "rib",
        "costolone": "rib",
        "costoloni": "rib",
        "ramps": "ramp",
        "rampa": "ramp",
        "rampe": "ramp",
        "steps": "step",
        "gradino": "step",
        "gradini": "step",
        "tetto": "roof",
        "roofing": "roof",
        "colonnato": "colonnade",
        "colonnade": "colonnade",
        "portico": "portico",
        "porticato": "portico",
        "arcade": "portico",
        "loggia": "loggia",
        "loggiato": "loggia",
        "pillar": "pillar",
        "pillars": "pillar",
        "pilastro": "pillar",
        "pilastri": "pillar",
    }
    return aliases.get(value, value)


def _cidoc_element_class(local_class: str) -> str:
    return CIDOC_E26 if local_class in FEATURE_CLASSES else CIDOC_E22


def _fallback_class_def(local_class: str) -> dict[str, str]:
    return {
        "node_prefix": _pascal_case(local_class),
        "display_label": _pascal_case(local_class),
        "cidoc_class": _cidoc_element_class(local_class),
        "local_class": _pascal_case(local_class),
    }


def _find_first_element(nodes: Iterable[Node], local_class: str) -> Node | None:
    local_class = _registry_local_class(local_class)
    for node in nodes:
        if node.local_class == local_class:
            return node
    return None


def _find_elements(nodes: Iterable[Node], local_class: str) -> list[Node]:
    local_class = _registry_local_class(local_class)
    return [node for node in nodes if node.local_class == local_class]


def _registry_local_class(local_class: str) -> str:
    normalized = _normalize_class(local_class)
    class_def = POINT_CLOUD_CLASS_REGISTRY.get(normalized, _fallback_class_def(normalized))
    return class_def["local_class"]


def _node_xy(node: Node) -> tuple[float, float] | None:
    try:
        x = float(node.properties.get("global_box_center_x", ""))
        y = float(node.properties.get("global_box_center_y", ""))
    except ValueError:
        return None
    return x, y


def _equispaced_column_sequence(columns: list[Node], spacing_tolerance: float) -> list[Node]:
    if len(columns) < 2:
        return columns

    xs = [float(_node_xy(column)[0]) for column in columns if _node_xy(column) is not None]
    ys = [float(_node_xy(column)[1]) for column in columns if _node_xy(column) is not None]
    sort_axis = 0 if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else 1
    sorted_columns = sorted(columns, key=lambda column: _node_xy(column)[sort_axis])
    distances = [
        _xy_distance(_node_xy(left), _node_xy(right))
        for left, right in zip(sorted_columns, sorted_columns[1:])
    ]
    distances = [distance for distance in distances if distance > 0]
    if not distances:
        return []

    median_spacing = sorted(distances)[len(distances) // 2]
    if median_spacing == 0:
        return []

    max_deviation = max(abs(distance - median_spacing) / median_spacing for distance in distances)
    return sorted_columns if max_deviation <= spacing_tolerance else []


def _aligned_portico_arcade_sequence(
    nodes: Iterable[Node],
    *,
    min_arches: int,
    min_columns: int,
    spacing_tolerance: float,
    spatial_index: set[tuple[str, str]] | None,
) -> list[Node]:
    arches = [arch for arch in _find_elements(nodes, "arch") if _node_xy(arch) is not None]
    if len(arches) >= min_arches:
        arch_sequence = _equispaced_column_sequence(arches, spacing_tolerance)
        if len(arch_sequence) >= min_arches:
            return arch_sequence

    columns = [column for column in _find_elements(nodes, "column") if _node_xy(column) is not None]
    architraves = _find_elements(nodes, "architrave")
    if len(columns) < min_columns or not architraves:
        return []

    column_sequence = _equispaced_column_sequence(columns, spacing_tolerance)
    if len(column_sequence) < min_columns:
        return []

    if spatial_index is not None and not any(
        _any_spatial_relation(architrave.node_id, [column.node_id for column in column_sequence], spatial_index)
        for architrave in architraves
    ):
        return []

    return [*column_sequence, *architraves]


def _aligned_open_side_sequence(
    nodes: Iterable[Node],
    *,
    min_arches: int,
    min_supports: int,
    spacing_tolerance: float,
    spatial_index: set[tuple[str, str]] | None,
) -> list[Node]:
    arches = [arch for arch in _find_elements(nodes, "arch") if _node_xy(arch) is not None]
    supports = [
        support
        for support in [*_find_elements(nodes, "column"), *_find_elements(nodes, "pillar")]
        if _node_xy(support) is not None
    ]

    arch_sequence = _equispaced_column_sequence(arches, spacing_tolerance) if len(arches) >= min_arches else []
    support_sequence = (
        _equispaced_column_sequence(supports, spacing_tolerance) if len(supports) >= min_supports else []
    )

    if len(arch_sequence) < min_arches or len(support_sequence) < min_supports:
        return []

    if spatial_index is not None and not any(
        _any_spatial_relation(arch.node_id, [support.node_id for support in support_sequence], spatial_index)
        for arch in arch_sequence
    ):
        return []

    return [*arch_sequence, *support_sequence]


def _find_scene_evidence(scene_evidence: Iterable[dict[str, str]], accepted_rules: set[str]) -> dict[str, str] | None:
    for row in scene_evidence:
        normalized = _normalize_row(row)
        rule = _normalize_class(
            normalized.get("rule", "")
            or normalized.get("feature", "")
            or normalized.get("class_semantic_label", "")
            or normalized.get("semantic_label", "")
        )
        if rule in accepted_rules:
            return normalized
    return None


def _portico_scene_evidence_allows(evidence: dict[str, str]) -> bool:
    required_positive_fields = (
        "continuous_cover_above",
        "covered_walkable_space_behind",
        "ground_floor",
        "open_to_external_space",
        "opposite_side_against_building_or_walls",
    )
    return all(_is_truthy(evidence.get(field, "")) for field in required_positive_fields)


def _loggia_scene_evidence_allows(evidence: dict[str, str]) -> bool:
    required_positive_fields = (
        "covered_room_or_gallery",
        "fully_open_side_to_external_space",
        "arches_on_columns_or_pillars",
        "inside_building_volume",
        "intermediate_representative_function",
    )
    if not all(_is_truthy(evidence.get(field, "")) for field in required_positive_fields):
        return False

    ground_floor_direct_access = (
        _is_truthy(evidence.get("ground_floor", ""))
        and _is_truthy(evidence.get("direct_building_access_function", ""))
    )
    simple_entrance_filter = _is_truthy(evidence.get("simple_entrance_filter", ""))
    return not (ground_floor_direct_access or simple_entrance_filter)


def _is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "s"}


def _find_cover_elements(nodes: Iterable[Node]) -> list[Node]:
    cover_classes = {"Roof", "Vault", "Floor"}
    cover_terms = ("cover", "copertura", "solaio", "terrazza", "tetto", "orizzontamento")
    covers = []
    for node in nodes:
        searchable = " ".join(
            [
                node.local_class,
                node.label,
                node.properties.get("display_label", ""),
                node.properties.get("description", ""),
                node.properties.get("position", ""),
            ]
        ).lower()
        if node.local_class in cover_classes or any(term in searchable for term in cover_terms):
            covers.append(node)
    return covers


def _any_spatial_relation(
    source_node_id: str,
    target_node_ids: Iterable[str],
    spatial_index: set[tuple[str, str]] | None,
) -> bool:
    return any(_are_spatially_related(source_node_id, target_node_id, spatial_index) for target_node_id in target_node_ids)


def _xy_distance(left: tuple[float, float] | None, right: tuple[float, float] | None) -> float:
    if left is None or right is None:
        return 0.0
    return math.dist(left, right)


def _columns_have_structural_support_function(
    columns: Iterable[Node],
    nodes: Iterable[Node],
    edges: Iterable[Edge],
) -> bool:
    node_by_id = {node.node_id: node for node in nodes}
    column_ids = {column.node_id for column in columns}
    supported_columns = set()

    for edge in edges:
        if edge.source not in column_ids or edge.predicate != P103_WAS_INTENDED_FOR:
            continue
        target = node_by_id.get(edge.target)
        if target is None:
            continue
        searchable = " ".join(
            [
                target.label,
                target.properties.get("value", ""),
                target.properties.get("description", ""),
            ]
        ).lower()
        if any(token in searchable for token in ("support", "sostegno", "struttur")):
            supported_columns.add(edge.source)

    return len(supported_columns) == len(column_ids)


def _resolve_measurement_target(nodes: Iterable[Node], row: dict[str, str]) -> Node | None:
    target_node_id = row.get("target_node_id", "")
    if target_node_id:
        for node in nodes:
            if node.node_id == target_node_id:
                return node

    local_class = row.get("target_local_class", "") or row.get("class_semantic_label", "")
    target_index = row.get("target_index", "")
    if not local_class or not target_index:
        return None

    local_class = _registry_local_class(local_class)
    matches = [node for node in nodes if node.local_class == local_class]
    try:
        return matches[int(target_index) - 1]
    except (IndexError, ValueError):
        return None


def _resolve_relation_endpoint(nodes: Iterable[Node], row: dict[str, str], prefix: str) -> Node | None:
    node_id = row.get(f"{prefix}_node_id", "")
    if node_id:
        for node in nodes:
            if node.node_id == node_id:
                return node

    local_class = row.get(f"{prefix}_local_class", "")
    index = row.get(f"{prefix}_index", "")
    if not local_class or not index:
        return None

    local_class = _registry_local_class(local_class)
    matches = [node for node in nodes if node.local_class == local_class]
    try:
        return matches[int(index) - 1]
    except (IndexError, ValueError):
        return None


def _normalize_predicate(predicate: str) -> str:
    predicate = predicate.strip()
    aliases = {
        "P46": P46_IS_COMPOSED_OF,
        "P46_is_composed_of": P46_IS_COMPOSED_OF,
        "crm:P46": P46_IS_COMPOSED_OF,
        "P46i": P46I_FORMS_PART_OF,
        "P46i_forms_part_of": P46I_FORMS_PART_OF,
        "crm:P46i": P46I_FORMS_PART_OF,
        "P56": P56_BEARS_FEATURE,
        "P56_bears_feature": P56_BEARS_FEATURE,
        "crm:P56": P56_BEARS_FEATURE,
        "P56i": P56I_IS_FOUND_ON,
        "P56i_is_found_on": P56I_IS_FOUND_ON,
        "crm:P56i": P56I_IS_FOUND_ON,
    }
    return aliases.get(predicate, predicate)


def _allowed_element_relation_predicates() -> set[str]:
    return {
        P46_IS_COMPOSED_OF,
        P46I_FORMS_PART_OF,
        P56_BEARS_FEATURE,
        P56I_IS_FOUND_ON,
    }


def _are_spatially_related(
    source_node_id: str,
    target_node_id: str,
    spatial_index: set[tuple[str, str]] | None,
) -> bool:
    if spatial_index is None:
        return True
    return (source_node_id, target_node_id) in spatial_index


def _relation_has_spatial_support(
    relation: dict[str, str],
    source_node_id: str,
    target_node_id: str,
    spatial_index: set[tuple[str, str]] | None,
) -> bool:
    explicit_support = relation.get("spatially_supported", "").strip().lower()
    if explicit_support in {"1", "true", "yes", "y"}:
        return True
    return _are_spatially_related(source_node_id, target_node_id, spatial_index)


def _namespace_scene_node(scene_id: str, node: Node) -> Node:
    if node.local_class in {"typology", "material", "function"}:
        return node

    return Node(
        node_id=f"{scene_id}_{node.node_id}",
        label=node.label,
        cidoc_class=node.cidoc_class,
        local_class=node.local_class,
        properties={**node.properties, "scene_id": scene_id},
    )


def _namespace_scene_edge(id_map: dict[str, str], edge: Edge) -> Edge:
    return Edge(
        source=id_map.get(edge.source, edge.source),
        target=id_map.get(edge.target, edge.target),
        predicate=edge.predicate,
    )


def _first_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if value:
            return value
    return ""


def _split_multi_value(value: str) -> list[str]:
    if not value:
        return []
    parts = re.split(r"\s*(?:;|\|)\s*", value)
    return [part.strip() for part in parts if part.strip()]


def _pascal_case(value: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[_\W]+", value) if part)


def _slug(value: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", _ascii_fold(value))
    return "".join(part.capitalize() for part in parts) or "Unspecified"


def _ascii_fold(value: str) -> str:
    replacements = {
        "à": "a",
        "è": "e",
        "é": "e",
        "ì": "i",
        "ò": "o",
        "ù": "u",
        "À": "A",
        "È": "E",
        "É": "E",
        "Ì": "I",
        "Ò": "O",
        "Ù": "U",
        "'": "",
        "’": "",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value
