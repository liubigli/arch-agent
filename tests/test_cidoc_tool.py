from types import SimpleNamespace

from arch_agent.tools.cidoc_tools import create_cidoc_tools


def _context(*, annotations=True):
    objects = {
        "column_0": {"semantic_label": "column"},
        "door_window_0": {"semantic_label": "door_window"},
        "wall_0": {"semantic_label": "wall"},
    }
    object_annotations = {}
    if annotations:
        object_annotations = {
            "column_0": [{
                "semantic_label": "column",
                "material": "Breccia policroma",
                "typology": "Colonna dorica",
                "function": "Sostegno",
            }],
            "door_window_0": [{
                "semantic_label": "door_window",
                "typology": "Porta",
            }],
            "wall_0": [{
                "semantic_label": "wall",
                "material": "Muratura intonacata",
            }],
        }
    return SimpleNamespace(
        params=SimpleNamespace(point_cloud_path="scena_test.laz"),
        objects=objects,
        object_annotations=object_annotations,
        relationship_layers={
            "L1": [
                (
                    "door_window_0",
                    "wall_0",
                    "is_opening_in",
                    "architectural_rule",
                )
            ]
        },
    )


def _tool(ctx):
    return create_cidoc_tools(ctx)[0]


def test_queries_material_triple_from_materialized_cidoc_graph():
    result = _tool(_context()).invoke({
        "object_name": "column_0",
        "predicate": "P45",
        "direction": "outgoing",
    })

    assert "crm:P45_consists_of" in result
    assert "Breccia policroma" in result
    assert "Matched triples: 1" in result


def test_queries_element_relation_supported_by_spatial_graph():
    result = _tool(_context()).invoke({
        "object_name": "door_window_0",
        "predicate": "P46i",
        "direction": "outgoing",
    })

    assert "crm:P46i_forms_part_of" in result
    assert "wall_0" in result


def test_reports_missing_csv_instead_of_using_spatial_graph_as_cidoc():
    result = _tool(_context(annotations=False)).invoke({
        "object_name": "column_0",
    })

    assert "CIDOC knowledge graph unavailable" in result
    assert "requires an annotation CSV" in result


def test_reports_absent_class_without_building_fake_nodes():
    result = _tool(_context()).invoke({"semantic_label": "vault"})

    assert "absent (0 objects)" in result
    assert "No CIDOC triples" in result


def test_zero_limit_returns_all_matching_triples():
    result = _tool(_context()).invoke({
        "object_name": "column_0",
        "direction": "outgoing",
        "limit": 0,
    })

    assert "Matched triples: 3" in result
    assert "crm:P2_has_type" in result
    assert "crm:P45_consists_of" in result
    assert "crm:P103_was_intended_for" in result
    assert "additional triples not shown" not in result
    assert "column_0 [node=" in result


if __name__ == "__main__":
    test_queries_material_triple_from_materialized_cidoc_graph()
    test_queries_element_relation_supported_by_spatial_graph()
    test_reports_missing_csv_instead_of_using_spatial_graph_as_cidoc()
    test_reports_absent_class_without_building_fake_nodes()
    test_zero_limit_returns_all_matching_triples()
    print("CIDOC tool tests: 5 passed")
