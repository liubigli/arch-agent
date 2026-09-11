"""Export and validate spatial-graph triples for an architectural scene.

This script is intentionally separate from the LLM benchmark. It tests the
computed triples directly:

    source --relationship--> target

It can also compare the generated triples with an optional manual/gold CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Iterable

import numpy as np

from arch_agent.pipeline.pipeline import PipelineParams, run_pipeline
from arch_agent.pipeline.l3_cidoc_graph_builder import (
    POINT_CLOUD_CLASS_REGISTRY,
    build_scene_graph as build_cidoc_scene_graph,
    _fallback_class_def,
    _normalize_class,
    _slug,
)
from arch_agent.pipeline.relationships import (
    _determine_geometric_relationships,
    _has_relationship_contact,
    _rests_on,
    mereological_relation_type,
    supports_label_pair,
)


ARCHITECTURAL_RULE_LEVEL = "architectural_rule"
CSV_METADATA_LEVEL = "csv_metadata"
STRUCTURAL_EVIDENCE_LEVEL = "csv_structural_evidence"

VALID_RELATIONSHIPS = {
    "near",
    "adjacent_to",
    "above",
    "below",
    "supports",
    "rests_on",
    "has_part",
    "part_of",
    "is_opening_in",
    "is_ornament_of",
    "is_attached_to",
    "is_connected_to",
    "is_placed_on",
    "is_rib_of",
}

GEOMETRIC_RELATIONSHIPS = {"near", "adjacent_to", "above", "below"}
SUPPORT_RELATIONSHIPS = {"supports", "rests_on"}
COMPOSITION_RELATIONSHIPS = {
    "has_part",
    "part_of",
    "is_opening_in",
    "is_ornament_of",
    "is_attached_to",
    "is_connected_to",
    "is_placed_on",
    "is_rib_of",
}

INVERSE_RELATIONSHIP = {
    "near": "near",
    "adjacent_to": "adjacent_to",
    "above": "below",
    "below": "above",
    "supports": "rests_on",
    "rests_on": "supports",
    "has_part": None,
    "part_of": "has_part",
    "is_opening_in": "has_part",
    "is_ornament_of": "has_part",
    "is_attached_to": "has_part",
    "is_connected_to": "has_part",
    "is_placed_on": "has_part",
    "is_rib_of": "has_part",
}

RAW_FIELDS = [
    "source",
    "relationship",
    "target",
    "derivation_level",
    "source_label",
    "target_label",
    "source_centroid_x",
    "source_centroid_y",
    "source_centroid_z",
    "target_centroid_x",
    "target_centroid_y",
    "target_centroid_z",
    "source_box_center_x",
    "source_box_center_y",
    "source_box_center_z",
    "target_box_center_x",
    "target_box_center_y",
    "target_box_center_z",
    "centroid_distance",
]

VALIDATION_FIELDS = RAW_FIELDS + [
    "status",
    "inverse_expected",
    "inverse_exists",
    "semantic_rule_ok",
    "geometry_ok",
    "invalid_issues",
    "review_issues",
]

COMPARISON_FIELDS = [
    "source",
    "relationship",
    "target",
    "status",
    "in_generated",
    "in_gold",
    "source_label",
    "target_label",
    "derivation_level",
]

CIDOC_RAW_FIELDS = [
    "source",
    "relationship",
    "target",
    "source_label",
    "target_label",
    "source_local_class",
    "target_local_class",
    "source_cidoc_class",
    "target_cidoc_class",
    "source_value",
    "target_value",
    "source_scene_id",
    "target_scene_id",
]

CIDOC_VALIDATION_FIELDS = CIDOC_RAW_FIELDS + [
    "status",
    "invalid_issues",
    "review_issues",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate spatial-graph triples without calling the LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("point_cloud_path", help="Input LAZ scene.")
    parser.add_argument("--annotation-csv", default=None, help="Optional scene annotation CSV.")
    parser.add_argument("--gold-csv", default=None, help="Optional manual/gold triples CSV.")
    parser.add_argument("--output-dir", default="triple_validation_outputs")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument(
        "--graph",
        choices=("spatial", "cidoc", "both"),
        default="spatial",
        help="Which triples to export and validate.",
    )

    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--distance-threshold", type=float, default=2.0)
    parser.add_argument("--sample-n", type=int, default=150_000)
    parser.add_argument("--use-normals", action="store_true")
    parser.add_argument("--annotation-match-threshold", type=float, default=2.0)
    parser.add_argument("--surface-contact-threshold", type=float, default=0.10)
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Do not write Markdown table copies of the CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scene_path = resolve_local_path(args.point_cloud_path)
    annotation_csv = resolve_optional_path(args.annotation_csv)
    gold_csv = resolve_optional_path(args.gold_csv)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = PipelineParams(
        point_cloud_path=str(scene_path),
        sample_n=args.sample_n,
        eps=args.eps,
        min_samples=args.min_samples,
        distance_threshold=args.distance_threshold,
        use_normals=args.use_normals,
        annotation_csv_path=str(annotation_csv) if annotation_csv else None,
        annotation_match_threshold=args.annotation_match_threshold,
    )
    ctx = run_pipeline(params)

    scene_name = sanitize_token(args.scene_name or scene_path.stem)
    threshold_token = str(args.distance_threshold).replace(".", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{scene_name}_threshold_{threshold_token}_{timestamp}"

    written_paths = []

    if args.graph in {"spatial", "both"}:
        spatial_triples = ctx.relationship_layers.get("L1", [])
        spatial_raw = build_raw_records(ctx.objects, spatial_triples)
        spatial_validation = validate_records(
            ctx.objects,
            spatial_triples,
            distance_threshold=args.distance_threshold,
            surface_contact_threshold=args.surface_contact_threshold,
        )
        spatial_summary = build_summary(ctx, spatial_raw, spatial_validation, args)
        written_paths.extend(
            write_spatial_outputs(
                output_dir,
                stem,
                scene_name,
                spatial_raw,
                spatial_validation,
                spatial_summary,
                write_markdown=not args.no_markdown,
            )
        )
        if gold_csv:
            written_paths.extend(
                write_gold_comparison_outputs(
                    output_dir,
                    f"spatial_{stem}",
                    spatial_raw,
                    gold_csv,
                )
            )

    if args.graph in {"cidoc", "both"}:
        if not annotation_csv:
            raise ValueError("CIDOC/KG triples require --annotation-csv.")
        cidoc_scene_graph = build_cidoc_graph_from_context(ctx)
        cidoc_raw = build_cidoc_raw_records(cidoc_scene_graph)
        cidoc_validation = validate_cidoc_records(cidoc_scene_graph)
        cidoc_summary = build_cidoc_summary(ctx, cidoc_scene_graph, cidoc_raw, cidoc_validation, args)
        written_paths.extend(
            write_cidoc_outputs(
                output_dir,
                stem,
                scene_name,
                cidoc_raw,
                cidoc_validation,
                cidoc_summary,
                write_markdown=not args.no_markdown,
            )
        )
        if gold_csv:
            written_paths.extend(
                write_gold_comparison_outputs(
                    output_dir,
                    f"cidoc_{stem}",
                    cidoc_raw,
                    gold_csv,
                )
            )

    print()
    print("Triple validation completed.")
    for path in written_paths:
        print(path)


def resolve_local_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.exists():
        return path

    windows_path = PureWindowsPath(path_value)
    if windows_path.drive:
        drive = windows_path.drive.rstrip(":").lower()
        wsl_path = Path("/mnt") / drive / Path(*windows_path.parts[1:])
        if wsl_path.exists() or Path("/mnt").exists():
            return wsl_path

    return path


def resolve_optional_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    return resolve_local_path(path_value)


def resolve_output_dir(path_value: str) -> Path:
    return resolve_local_path(path_value)


def sanitize_token(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum() or char in {"_", "-"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "scene"


def write_spatial_outputs(
    output_dir: Path,
    stem: str,
    scene_name: str,
    raw_records: list[dict],
    validation_records: list[dict],
    summary: dict,
    *,
    write_markdown: bool,
) -> list[str]:
    raw_json = output_dir / f"spatial_triples_raw_{stem}.json"
    raw_csv = output_dir / f"spatial_triples_raw_{stem}.csv"
    validation_json = output_dir / f"spatial_triples_validation_{stem}.json"
    validation_csv = output_dir / f"spatial_triples_validation_{stem}.csv"
    summary_csv = output_dir / f"spatial_triples_summary_by_relation_{stem}.csv"
    report_txt = output_dir / f"spatial_triples_report_{stem}.txt"

    write_json(raw_json, {"summary": summary, "records": raw_records})
    write_csv(raw_csv, raw_records, RAW_FIELDS)
    write_json(validation_json, {"summary": summary, "records": validation_records})
    write_csv(validation_csv, validation_records, VALIDATION_FIELDS)
    write_csv(summary_csv, relation_summary_rows(raw_records), ["relationship", "count"])
    write_report(report_txt, summary)

    paths = [
        f"Spatial raw JSON       : {raw_json}",
        f"Spatial raw CSV        : {raw_csv}",
        f"Spatial validation JSON: {validation_json}",
        f"Spatial validation CSV : {validation_csv}",
        f"Spatial summary CSV    : {summary_csv}",
        f"Spatial report TXT     : {report_txt}",
    ]
    if write_markdown:
        table_md = output_dir / f"spatial_triples_validation_table_{stem}.md"
        write_markdown_table(
            table_md,
            validation_records,
            [
                "source",
                "relationship",
                "target",
                "derivation_level",
                "source_label",
                "target_label",
                "status",
                "invalid_issues",
                "review_issues",
            ],
            title=f"Spatial Triple Validation - {scene_name}",
        )
        paths.append(f"Spatial table MD       : {table_md}")
    return paths


def write_cidoc_outputs(
    output_dir: Path,
    stem: str,
    scene_name: str,
    raw_records: list[dict],
    validation_records: list[dict],
    summary: dict,
    *,
    write_markdown: bool,
) -> list[str]:
    raw_json = output_dir / f"cidoc_triples_raw_{stem}.json"
    raw_csv = output_dir / f"cidoc_triples_raw_{stem}.csv"
    validation_json = output_dir / f"cidoc_triples_validation_{stem}.json"
    validation_csv = output_dir / f"cidoc_triples_validation_{stem}.csv"
    summary_csv = output_dir / f"cidoc_triples_summary_by_relation_{stem}.csv"
    report_txt = output_dir / f"cidoc_triples_report_{stem}.txt"

    write_json(raw_json, {"summary": summary, "records": raw_records})
    write_csv(raw_csv, raw_records, CIDOC_RAW_FIELDS)
    write_json(validation_json, {"summary": summary, "records": validation_records})
    write_csv(validation_csv, validation_records, CIDOC_VALIDATION_FIELDS)
    write_csv(summary_csv, relation_summary_rows(raw_records), ["relationship", "count"])
    write_report(report_txt, summary)

    paths = [
        f"CIDOC raw JSON         : {raw_json}",
        f"CIDOC raw CSV          : {raw_csv}",
        f"CIDOC validation JSON  : {validation_json}",
        f"CIDOC validation CSV   : {validation_csv}",
        f"CIDOC summary CSV      : {summary_csv}",
        f"CIDOC report TXT       : {report_txt}",
    ]
    if write_markdown:
        table_md = output_dir / f"cidoc_triples_validation_table_{stem}.md"
        write_markdown_table(
            table_md,
            validation_records,
            [
                "source",
                "relationship",
                "target",
                "source_local_class",
                "target_local_class",
                "status",
                "invalid_issues",
                "review_issues",
            ],
            title=f"CIDOC/KG Triple Validation - {scene_name}",
        )
        paths.append(f"CIDOC table MD         : {table_md}")
    return paths


def write_gold_comparison_outputs(
    output_dir: Path,
    stem: str,
    raw_records: list[dict],
    gold_csv: Path,
) -> list[str]:
    comparison_records, comparison_summary = compare_with_gold(raw_records, gold_csv)
    comparison_json = output_dir / f"triples_gold_comparison_{stem}.json"
    comparison_csv = output_dir / f"triples_gold_comparison_{stem}.csv"
    comparison_report = output_dir / f"triples_gold_report_{stem}.txt"
    write_json(comparison_json, {"summary": comparison_summary, "records": comparison_records})
    write_csv(comparison_csv, comparison_records, COMPARISON_FIELDS)
    write_report(comparison_report, comparison_summary)
    return [
        f"Gold comparison JSON   : {comparison_json}",
        f"Gold comparison CSV    : {comparison_csv}",
        f"Gold comparison TXT    : {comparison_report}",
    ]


def build_raw_records(objects: dict, triples: Iterable[tuple]) -> list[dict]:
    records = []
    for src, tgt, rel, *rest in triples:
        level = rest[0] if rest else ""
        records.append(build_raw_record(objects, src, rel, tgt, level))
    return records


def build_raw_record(objects: dict, src: str, rel: str, tgt: str, level: str = "") -> dict:
    source = objects.get(src, {})
    target = objects.get(tgt, {})
    source_centroid = vector3(source.get("centroid"))
    target_centroid = vector3(target.get("centroid"))
    source_box = box_center(source)
    target_box = box_center(target)
    distance = ""
    if source_centroid and target_centroid:
        distance = round(float(np.linalg.norm(np.asarray(source_centroid) - np.asarray(target_centroid))), 6)

    return {
        "source": src,
        "relationship": rel,
        "target": tgt,
        "derivation_level": level,
        "source_label": source.get("semantic_label", ""),
        "target_label": target.get("semantic_label", ""),
        "source_centroid_x": coord(source_centroid, 0),
        "source_centroid_y": coord(source_centroid, 1),
        "source_centroid_z": coord(source_centroid, 2),
        "target_centroid_x": coord(target_centroid, 0),
        "target_centroid_y": coord(target_centroid, 1),
        "target_centroid_z": coord(target_centroid, 2),
        "source_box_center_x": coord(source_box, 0),
        "source_box_center_y": coord(source_box, 1),
        "source_box_center_z": coord(source_box, 2),
        "target_box_center_x": coord(target_box, 0),
        "target_box_center_y": coord(target_box, 1),
        "target_box_center_z": coord(target_box, 2),
        "centroid_distance": distance,
    }


def validate_records(
    objects: dict,
    triples: list[tuple],
    distance_threshold: float,
    surface_contact_threshold: float,
) -> list[dict]:
    triple_set = {(src, rel, tgt) for src, tgt, rel, *_ in triples}
    rows = []
    for src, tgt, rel, *rest in triples:
        level = rest[0] if rest else ""
        invalid_issues: list[str] = []
        review_issues: list[str] = []

        inverse_rel = INVERSE_RELATIONSHIP.get(rel)
        inverse_exists = ""
        semantic_rule_ok = ""
        geometry_ok = ""

        if src not in objects:
            invalid_issues.append("source_missing")
        if tgt not in objects:
            invalid_issues.append("target_missing")
        if src == tgt:
            invalid_issues.append("self_relation")
        if rel not in VALID_RELATIONSHIPS:
            invalid_issues.append("unknown_relationship_type")

        if src in objects and tgt in objects:
            if inverse_rel:
                inverse_exists = (tgt, inverse_rel, src) in triple_set
                if not inverse_exists:
                    invalid_issues.append("missing_inverse")

            semantic_rule_ok, semantic_issue = semantic_rule_check(objects, src, rel, tgt, level)
            if semantic_issue:
                if level in {CSV_METADATA_LEVEL, STRUCTURAL_EVIDENCE_LEVEL}:
                    review_issues.append(semantic_issue)
                else:
                    invalid_issues.append(semantic_issue)

            geometry_ok, geometry_issue = geometry_check(
                objects,
                src,
                rel,
                tgt,
                level,
                distance_threshold=distance_threshold,
                surface_contact_threshold=surface_contact_threshold,
            )
            if geometry_issue:
                if level in {CSV_METADATA_LEVEL, STRUCTURAL_EVIDENCE_LEVEL}:
                    review_issues.append(geometry_issue)
                else:
                    invalid_issues.append(geometry_issue)

        status = "valid"
        if invalid_issues:
            status = "invalid"
        elif review_issues:
            status = "review"

        row = build_raw_record(objects, src, rel, tgt, level)
        row.update(
            {
                "status": status,
                "inverse_expected": inverse_rel or "",
                "inverse_exists": bool_to_text(inverse_exists),
                "semantic_rule_ok": bool_to_text(semantic_rule_ok),
                "geometry_ok": bool_to_text(geometry_ok),
                "invalid_issues": "; ".join(invalid_issues),
                "review_issues": "; ".join(review_issues),
            }
        )
        rows.append(row)
    return rows


def semantic_rule_check(objects: dict, src: str, rel: str, tgt: str, level: str) -> tuple[bool | str, str | None]:
    src_label = objects[src].get("semantic_label")
    tgt_label = objects[tgt].get("semantic_label")

    if rel in GEOMETRIC_RELATIONSHIPS:
        return "", None
    if rel == "supports":
        ok = supports_label_pair(src_label, tgt_label)
        return ok, None if ok else "semantic_schema_not_allowed"
    if rel == "rests_on":
        ok = supports_label_pair(tgt_label, src_label)
        return ok, None if ok else "semantic_schema_not_allowed"
    if rel == "has_part":
        ok = mereological_relation_type(tgt_label, src_label) is not None
        return ok, None if ok else "semantic_schema_not_allowed"
    if rel in COMPOSITION_RELATIONSHIPS:
        ok = mereological_relation_type(src_label, tgt_label) == rel
        return ok, None if ok else "semantic_schema_not_allowed"

    return False, "unknown_relationship_type"


def geometry_check(
    objects: dict,
    src: str,
    rel: str,
    tgt: str,
    level: str,
    distance_threshold: float,
    surface_contact_threshold: float,
) -> tuple[bool | str, str | None]:
    source = objects[src]
    target = objects[tgt]

    if rel in GEOMETRIC_RELATIONSHIPS:
        pair_geometry = _determine_geometric_relationships(
            src,
            source,
            tgt,
            target,
            distance_threshold,
        )
        ok = any((s, t, r) == (src, tgt, rel) for s, t, r, *_ in pair_geometry)
        return ok, None if ok else "geometry_check_failed"

    if rel == "supports":
        ok = _rests_on(target, source)
        return ok, None if ok else "resting_geometry_not_confirmed"
    if rel == "rests_on":
        ok = _rests_on(source, target)
        return ok, None if ok else "resting_geometry_not_confirmed"

    if rel in COMPOSITION_RELATIONSHIPS:
        max_gap = max(surface_contact_threshold, min(distance_threshold * 0.25, 0.75))
        if rel == "has_part":
            ok = _has_relationship_contact(target, source, max_gap)
        else:
            ok = _has_relationship_contact(source, target, max_gap)
        return ok, None if ok else "contact_geometry_not_confirmed"

    return False, "unknown_relationship_type"


def build_cidoc_graph_from_context(ctx):
    annotation_rows, object_to_node_ids = cidoc_annotation_rows_and_object_map(ctx)
    if not annotation_rows:
        raise ValueError(
            "No matched CSV annotations found. CIDOC/KG triples require a CSV "
            "whose rows match scene objects by semantic_label and global_box_center."
        )

    scene_id = Path(ctx.params.point_cloud_path).stem
    spatial_rows = cidoc_spatial_relation_rows(ctx, object_to_node_ids)
    return build_cidoc_scene_graph(
        scene_id,
        annotation_rows,
        spatial_relations=spatial_rows,
    )


def cidoc_annotation_rows_and_object_map(ctx) -> tuple[list[dict], dict[str, list[str]]]:
    scene_id = _slug(Path(ctx.params.point_cloud_path).stem)
    rows: list[dict] = []
    object_to_node_ids: dict[str, list[str]] = {}
    class_counters: dict[str, int] = {}

    for object_name, annotations in getattr(ctx, "object_annotations", {}).items():
        obj = ctx.objects.get(object_name, {})
        for annotation in annotations:
            row = dict(annotation)
            row.setdefault("object_name", object_name)
            row.setdefault("semantic_label", obj.get("semantic_label", ""))

            local_class = row.get("class_semantic_label") or row.get("semantic_label", "")
            normalized_class = _normalize_class(local_class)
            if not normalized_class:
                continue
            class_counters[normalized_class] = class_counters.get(normalized_class, 0) + 1
            class_def = POINT_CLOUD_CLASS_REGISTRY.get(
                normalized_class,
                _fallback_class_def(normalized_class),
            )
            raw_node_id = f"{class_def['node_prefix']}_{class_counters[normalized_class]}"
            object_to_node_ids.setdefault(object_name, []).append(f"{scene_id}_{raw_node_id}")
            rows.append(row)

    return rows, object_to_node_ids


def cidoc_spatial_relation_rows(ctx, object_to_node_ids: dict[str, list[str]]) -> list[dict]:
    rows = []
    for src, tgt, rel, *_ in ctx.relationship_layers.get("L1", []):
        source_nodes = object_to_node_ids.get(src, [])
        target_nodes = object_to_node_ids.get(tgt, [])
        relation_type = cidoc_spatial_relation_type(rel)
        if not relation_type:
            continue
        for source_node in source_nodes:
            for target_node in target_nodes:
                rows.append(
                    {
                        "source_node_id": source_node,
                        "target_node_id": target_node,
                        "relation_type": relation_type,
                    }
                )
    return rows


def cidoc_spatial_relation_type(relationship: str) -> str:
    if relationship == "near":
        return "near"
    if relationship == "adjacent_to":
        return "adjacent"
    if relationship in {
        "above",
        "below",
        "supports",
        "rests_on",
        "has_part",
        "part_of",
        "is_opening_in",
        "is_ornament_of",
        "is_attached_to",
        "is_connected_to",
        "is_placed_on",
        "is_rib_of",
    }:
        return "touches"
    return ""


def build_cidoc_raw_records(scene_graph) -> list[dict]:
    node_by_id = {node.node_id: node for node in scene_graph.nodes}
    records = []
    for edge in scene_graph.edges:
        source = node_by_id.get(edge.source)
        target = node_by_id.get(edge.target)
        records.append(
            {
                "source": edge.source,
                "relationship": edge.predicate,
                "target": edge.target,
                "source_label": source.label if source else "",
                "target_label": target.label if target else "",
                "source_local_class": source.local_class if source else "",
                "target_local_class": target.local_class if target else "",
                "source_cidoc_class": source.cidoc_class if source else "",
                "target_cidoc_class": target.cidoc_class if target else "",
                "source_value": cidoc_node_value(source),
                "target_value": cidoc_node_value(target),
                "source_scene_id": cidoc_node_scene_id(source),
                "target_scene_id": cidoc_node_scene_id(target),
            }
        )
    return records


def validate_cidoc_records(scene_graph) -> list[dict]:
    node_by_id = {node.node_id: node for node in scene_graph.nodes}
    seen_edges = Counter((edge.source, edge.predicate, edge.target) for edge in scene_graph.edges)
    records = []
    for edge in scene_graph.edges:
        source = node_by_id.get(edge.source)
        target = node_by_id.get(edge.target)
        invalid_issues: list[str] = []
        review_issues: list[str] = []

        if source is None:
            invalid_issues.append("source_node_missing")
        if target is None:
            invalid_issues.append("target_node_missing")
        if edge.source == edge.target:
            invalid_issues.append("self_relation")
        if not edge.predicate.startswith("crm:P"):
            invalid_issues.append("non_cidoc_predicate")
        if seen_edges[(edge.source, edge.predicate, edge.target)] > 1:
            review_issues.append("duplicate_edge")

        if source is not None and target is not None:
            cidoc_issue = cidoc_predicate_issue(source, edge.predicate, target)
            if cidoc_issue:
                invalid_issues.append(cidoc_issue)

        status = "valid"
        if invalid_issues:
            status = "invalid"
        elif review_issues:
            status = "review"

        row = build_cidoc_raw_records(
            type(
                "_SingleEdgeGraph",
                (),
                {"nodes": [node for node in (source, target) if node is not None], "edges": [edge]},
            )()
        )[0]
        row.update(
            {
                "status": status,
                "invalid_issues": "; ".join(invalid_issues),
                "review_issues": "; ".join(review_issues),
            }
        )
        records.append(row)
    return records


def cidoc_predicate_issue(source, predicate: str, target) -> str | None:
    if predicate == "crm:P2_has_type":
        return None if target.local_class == "typology" else "P2_target_not_typology"
    if predicate == "crm:P45_consists_of":
        return None if target.local_class == "material" else "P45_target_not_material"
    if predicate == "crm:P103_was_intended_for":
        return None if target.local_class == "function" else "P103_target_not_function"
    if predicate == "crm:P46_is_composed_of":
        return None if cidoc_is_element(source) and cidoc_is_element(target) else "P46_requires_element_nodes"
    if predicate == "crm:P46i_forms_part_of":
        return None if cidoc_is_element(source) and cidoc_is_element(target) else "P46i_requires_element_nodes"
    if predicate == "crm:P56_bears_feature":
        return None if cidoc_is_element(source) and target.cidoc_class == "crm:E26_Physical_Feature" else "P56_target_not_feature"
    if predicate == "crm:P56i_is_found_on":
        return None if source.cidoc_class == "crm:E26_Physical_Feature" and cidoc_is_element(target) else "P56i_source_not_feature"
    if predicate == "crm:P39_measured":
        return None if source.local_class == "Measurement" and cidoc_is_element(target) else "P39_invalid_measurement_pattern"
    if predicate == "crm:P43_has_dimension":
        return None if source.local_class == "Measurement" and target.local_class == "Dimension" else "P43_invalid_dimension_pattern"
    return "unsupported_cidoc_predicate"


def cidoc_is_element(node) -> bool:
    return node.local_class not in {"typology", "material", "function", "Measurement", "Dimension"}


def cidoc_node_value(node) -> str:
    if node is None:
        return ""
    return (
        node.properties.get("value")
        or node.properties.get("description")
        or node.properties.get("display_label")
        or ""
    )


def cidoc_node_scene_id(node) -> str:
    if node is None:
        return ""
    return node.properties.get("scene_id", "")


def build_cidoc_summary(ctx, scene_graph, raw_records: list[dict], validation_records: list[dict], args) -> dict:
    relationship_counts = Counter(row["relationship"] for row in raw_records)
    status_counts = Counter(row["status"] for row in validation_records)
    invalid_issue_counts = Counter(
        issue
        for row in validation_records
        for issue in split_issues(row.get("invalid_issues", ""))
    )
    review_issue_counts = Counter(
        issue
        for row in validation_records
        for issue in split_issues(row.get("review_issues", ""))
    )
    node_classes = Counter(node.local_class for node in scene_graph.nodes)
    return {
        "scene": Path(str(ctx.params.point_cloud_path)).stem,
        "point_cloud_path": str(ctx.params.point_cloud_path),
        "annotation_csv_path": str(ctx.params.annotation_csv_path or ""),
        "source": "CIDOC/KG from matched CSV annotations plus spatial graph support",
        "distance_threshold": args.distance_threshold,
        "objects": len(ctx.objects),
        "cidoc_nodes": len(scene_graph.nodes),
        "triples": len(raw_records),
        "node_class_counts": dict(sorted(node_classes.items())),
        "relationship_counts": dict(sorted(relationship_counts.items())),
        "validation_status_counts": dict(sorted(status_counts.items())),
        "invalid_issue_counts": dict(sorted(invalid_issue_counts.items())),
        "review_issue_counts": dict(sorted(review_issue_counts.items())),
    }


def compare_with_gold(raw_records: list[dict], gold_csv: Path) -> tuple[list[dict], dict]:
    generated = {(row["source"], row["relationship"], row["target"]): row for row in raw_records}
    gold = read_gold_triples(gold_csv)
    gold_keys = set(gold)
    generated_keys = set(generated)

    rows = []
    for key in sorted(gold_keys | generated_keys):
        in_generated = key in generated_keys
        in_gold = key in gold_keys
        if in_generated and in_gold:
            status = "true_positive"
        elif in_generated:
            status = "false_positive"
        else:
            status = "false_negative"

        generated_row = generated.get(key, {})
        source, relationship, target = key
        rows.append(
            {
                "source": source,
                "relationship": relationship,
                "target": target,
                "status": status,
                "in_generated": in_generated,
                "in_gold": in_gold,
                "source_label": generated_row.get("source_label", ""),
                "target_label": generated_row.get("target_label", ""),
                "derivation_level": generated_row.get("derivation_level", ""),
            }
        )

    tp = sum(1 for row in rows if row["status"] == "true_positive")
    fp = sum(1 for row in rows if row["status"] == "false_positive")
    fn = sum(1 for row in rows if row["status"] == "false_negative")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "gold_csv": str(gold_csv),
        "generated_triples": len(generated_keys),
        "gold_triples": len(gold_keys),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    return rows, summary


def read_gold_triples(path: Path) -> set[tuple[str, str, str]]:
    text = read_text_fallback(path)
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(text.splitlines(), dialect=dialect))
    if not rows:
        return set()

    field_map = {field.lower().strip(): field for field in rows[0].keys() if field}
    source_col = first_existing(field_map, ("source", "src", "subject", "from"))
    rel_col = first_existing(field_map, ("relationship", "relation", "predicate", "rel"))
    target_col = first_existing(field_map, ("target", "tgt", "object", "to"))
    if not source_col or not rel_col or not target_col:
        raise ValueError(
            "Gold CSV must contain source/relationship/target columns "
            "(or aliases: src, relation, target)."
        )

    triples = set()
    for row in rows:
        source = str(row.get(source_col, "")).strip()
        relationship = str(row.get(rel_col, "")).strip()
        target = str(row.get(target_col, "")).strip()
        if source and relationship and target:
            triples.add((source, relationship, target))
    return triples


def first_existing(field_map: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in field_map:
            return field_map[candidate]
    return None


def read_text_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text()


def build_summary(ctx, raw_records: list[dict], validation_records: list[dict], args: argparse.Namespace) -> dict:
    relationship_counts = Counter(row["relationship"] for row in raw_records)
    level_counts = Counter(row["derivation_level"] for row in raw_records)
    status_counts = Counter(row["status"] for row in validation_records)
    invalid_issue_counts = Counter(
        issue
        for row in validation_records
        for issue in split_issues(row.get("invalid_issues", ""))
    )
    review_issue_counts = Counter(
        issue
        for row in validation_records
        for issue in split_issues(row.get("review_issues", ""))
    )
    return {
        "scene": Path(str(ctx.params.point_cloud_path)).stem,
        "point_cloud_path": str(ctx.params.point_cloud_path),
        "annotation_csv_path": str(ctx.params.annotation_csv_path or ""),
        "distance_threshold": args.distance_threshold,
        "eps": args.eps,
        "min_samples": args.min_samples,
        "sample_n": args.sample_n,
        "objects": len(ctx.objects),
        "triples": len(raw_records),
        "relationship_counts": dict(sorted(relationship_counts.items())),
        "derivation_level_counts": dict(sorted(level_counts.items())),
        "validation_status_counts": dict(sorted(status_counts.items())),
        "invalid_issue_counts": dict(sorted(invalid_issue_counts.items())),
        "review_issue_counts": dict(sorted(review_issue_counts.items())),
    }


def relation_summary_rows(raw_records: list[dict]) -> list[dict]:
    counts = Counter(row["relationship"] for row in raw_records)
    return [
        {"relationship": relationship, "count": count}
        for relationship, count in sorted(counts.items())
    ]


def split_issues(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(";") if item.strip()]


def vector3(value) -> list[float]:
    if value is None:
        return []
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) < 3:
        return []
    return [round(float(array[0]), 6), round(float(array[1]), 6), round(float(array[2]), 6)]


def box_center(obj: dict) -> list[float]:
    bounds = obj.get("bounds")
    if not bounds:
        return []
    return vector3((np.asarray(bounds["min"], dtype=float) + np.asarray(bounds["max"], dtype=float)) / 2.0)


def coord(values: list[float], index: int):
    if not values or len(values) <= index:
        return ""
    return values[index]


def bool_to_text(value) -> str:
    if value == "":
        return ""
    return "true" if bool(value) else "false"


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_table(path: Path, rows: list[dict], fields: list[str], title: str) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        values = [markdown_cell(row.get(field, "")) for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def write_report(path: Path, summary: dict) -> None:
    lines = ["Triple validation report", ""]
    for key, value in summary.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            if value:
                for sub_key, sub_value in value.items():
                    lines.append(f"  - {sub_key}: {sub_value}")
            else:
                lines.append("  - none")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
