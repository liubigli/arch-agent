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
import posixpath
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Iterable
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from arch_agent.pipeline.pipeline import PipelineParams, run_pipeline
from arch_agent.pipeline.l3_cidoc_graph_builder import (
    POINT_CLOUD_CLASS_REGISTRY,
    build_scene_graph as build_cidoc_scene_graph,
    _fallback_class_def,
    _normalize_class,
    _slug,
)
try:
    from arch_agent.semantic_schema import ARCHITECTURAL_CLASS_RULES
except ImportError:  # compatibility with older WSL copies
    from arch_agent.pipeline.relationships import ARCHITECTURAL_CLASS_RULES


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
UNSUPPORTED_ABOVE_PAIRS = {
    ("column", "arch"),
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

MANUAL_REVIEW_FIELDS = [
    "manual_label",
    "manual_issue",
    "manual_note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate spatial-graph triples without calling the LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("point_cloud_path", help="Input LAZ scene.")
    parser.add_argument("--annotation-csv", default=None, help="Optional scene annotation CSV.")
    parser.add_argument(
        "--gold-csv",
        "--gold-file",
        dest="gold_csv",
        default=None,
        help="Optional manual/gold triples file (.csv, .tsv, or .xlsx).",
    )
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
    manual_xlsx = output_dir / f"spatial_triples_manual_review_{stem}.xlsx"
    summary_csv = output_dir / f"spatial_triples_summary_by_relation_{stem}.csv"
    report_txt = output_dir / f"spatial_triples_report_{stem}.txt"

    write_json(raw_json, {"summary": summary, "records": raw_records})
    write_csv(raw_csv, raw_records, RAW_FIELDS)
    write_json(validation_json, {"summary": summary, "records": validation_records})
    write_csv(validation_csv, validation_records, VALIDATION_FIELDS)
    write_xlsx(
        manual_xlsx,
        add_blank_manual_fields(validation_records),
        VALIDATION_FIELDS + MANUAL_REVIEW_FIELDS,
        sheet_name="spatial_triples",
    )
    write_csv(summary_csv, relation_summary_rows(raw_records), ["relationship", "count"])
    write_report(report_txt, summary)

    paths = [
        f"Spatial raw JSON       : {raw_json}",
        f"Spatial raw CSV        : {raw_csv}",
        f"Spatial validation JSON: {validation_json}",
        f"Spatial validation CSV : {validation_csv}",
        f"Spatial manual XLSX    : {manual_xlsx}",
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
    manual_xlsx = output_dir / f"cidoc_triples_manual_review_{stem}.xlsx"
    summary_csv = output_dir / f"cidoc_triples_summary_by_relation_{stem}.csv"
    report_txt = output_dir / f"cidoc_triples_report_{stem}.txt"

    write_json(raw_json, {"summary": summary, "records": raw_records})
    write_csv(raw_csv, raw_records, CIDOC_RAW_FIELDS)
    write_json(validation_json, {"summary": summary, "records": validation_records})
    write_csv(validation_csv, validation_records, CIDOC_VALIDATION_FIELDS)
    write_xlsx(
        manual_xlsx,
        add_blank_manual_fields(validation_records),
        CIDOC_VALIDATION_FIELDS + MANUAL_REVIEW_FIELDS,
        sheet_name="cidoc_triples",
    )
    write_csv(summary_csv, relation_summary_rows(raw_records), ["relationship", "count"])
    write_report(report_txt, summary)

    paths = [
        f"CIDOC raw JSON         : {raw_json}",
        f"CIDOC raw CSV          : {raw_csv}",
        f"CIDOC validation JSON  : {validation_json}",
        f"CIDOC validation CSV   : {validation_csv}",
        f"CIDOC manual XLSX      : {manual_xlsx}",
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
    comparison_xlsx = output_dir / f"triples_gold_comparison_{stem}.xlsx"
    comparison_report = output_dir / f"triples_gold_report_{stem}.txt"
    write_json(comparison_json, {"summary": comparison_summary, "records": comparison_records})
    write_csv(comparison_csv, comparison_records, COMPARISON_FIELDS)
    write_xlsx(comparison_xlsx, comparison_records, COMPARISON_FIELDS, sheet_name="gold_comparison")
    write_report(comparison_report, comparison_summary)
    return [
        f"Gold comparison JSON   : {comparison_json}",
        f"Gold comparison CSV    : {comparison_csv}",
        f"Gold comparison XLSX   : {comparison_xlsx}",
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
        if not ok and rel in {"above", "below"}:
            ok = vertical_relation_plausible(
                source,
                rel,
                target,
                distance_threshold=distance_threshold,
            )
        if not ok and rel in {"near", "adjacent_to"}:
            ok = local_spatial_relation_plausible(
                source,
                rel,
                target,
                distance_threshold=distance_threshold,
            )
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
            ok = relationship_contact_confirmed(target, source, max_gap)
        else:
            ok = relationship_contact_confirmed(source, target, max_gap)
        return ok, None if ok else "contact_geometry_not_confirmed"

    return False, "unknown_relationship_type"


def relationship_contact_confirmed(child: dict, parent: dict, max_gap: float) -> bool:
    child_bounds = child["bounds"]
    parent_bounds = parent["bounds"]
    if bounds_gap(child_bounds, parent_bounds) <= max_gap:
        return True
    if local_is_above(child, parent) or local_is_above(parent, child):
        return True
    return (
        overlap_xy_ratio(child_bounds, parent_bounds) >= 0.05
        and axis_overlap_ratio(child_bounds, parent_bounds, axis=2) >= 0.05
    )


def axis_gap(min1: float, max1: float, min2: float, max2: float) -> float:
    if max1 < min2:
        return float(min2 - max1)
    if max2 < min1:
        return float(min1 - max2)
    return 0.0


def bounds_gap(left: dict, right: dict) -> float:
    gaps = [
        axis_gap(
            float(left["min"][axis]),
            float(left["max"][axis]),
            float(right["min"][axis]),
            float(right["max"][axis]),
        )
        for axis in range(3)
    ]
    return float(np.linalg.norm(gaps))


def xy_area(bounds: dict) -> float:
    dims = np.asarray(bounds["max"][:2], dtype=float) - np.asarray(bounds["min"][:2], dtype=float)
    return float(max(dims[0], 0.0) * max(dims[1], 0.0))


def overlap_xy_ratio(left: dict, right: dict) -> float:
    x_overlap = max(
        0.0,
        min(float(left["max"][0]), float(right["max"][0]))
        - max(float(left["min"][0]), float(right["min"][0])),
    )
    y_overlap = max(
        0.0,
        min(float(left["max"][1]), float(right["max"][1]))
        - max(float(left["min"][1]), float(right["min"][1])),
    )
    reference_area = min(xy_area(left), xy_area(right))
    if reference_area <= 0:
        return 0.0
    return float((x_overlap * y_overlap) / reference_area)


def axis_overlap_ratio(left: dict, right: dict, axis: int) -> float:
    overlap = max(
        0.0,
        min(float(left["max"][axis]), float(right["max"][axis]))
        - max(float(left["min"][axis]), float(right["min"][axis])),
    )
    size_left = max(float(left["max"][axis]) - float(left["min"][axis]), 0.0)
    size_right = max(float(right["max"][axis]) - float(right["min"][axis]), 0.0)
    reference = min(size_left, size_right)
    if reference <= 0:
        return 0.0
    return float(overlap / reference)


def vertical_gap(upper_bounds: dict, lower_bounds: dict) -> float:
    return float(upper_bounds["min"][2] - lower_bounds["max"][2])


def local_is_above(upper: dict, lower: dict, max_gap: float = 0.75) -> bool:
    upper_label = upper.get("semantic_label")
    lower_label = lower.get("semantic_label")
    if (upper_label, lower_label) in UNSUPPORTED_ABOVE_PAIRS:
        return False

    upper_bounds = upper["bounds"]
    lower_bounds = lower["bounds"]
    z_gap = vertical_gap(upper_bounds, lower_bounds)
    return (
        float(upper["centroid"][2]) > float(lower["centroid"][2])
        and -0.15 <= z_gap <= max_gap
        and overlap_xy_ratio(upper_bounds, lower_bounds) >= 0.05
    )


def supports_label_pair(lower_label: str | None, upper_label: str | None) -> bool:
    lower_rules = ARCHITECTURAL_CLASS_RULES.get(lower_label or "", {})
    upper_rules = ARCHITECTURAL_CLASS_RULES.get(upper_label or "", {})
    return (
        upper_label in lower_rules.get("can_support", set())
        and lower_label in upper_rules.get("can_rest_on", set())
    )


def mereological_relation_type(child_label: str | None, parent_label: str | None) -> str | None:
    rules = ARCHITECTURAL_CLASS_RULES.get(child_label or "", {})
    return rules.get("part_of", {}).get(parent_label)


def _rests_on(upper: dict, lower: dict) -> bool:
    upper_label = upper.get("semantic_label")
    lower_label = lower.get("semantic_label")
    if not supports_label_pair(lower_label, upper_label):
        return False

    upper_bounds = upper["bounds"]
    lower_bounds = lower["bounds"]
    z_gap = vertical_gap(upper_bounds, lower_bounds)
    return (
        float(upper["centroid"][2]) > float(lower["centroid"][2])
        and -0.15 <= z_gap <= 0.35
        and overlap_xy_ratio(upper_bounds, lower_bounds) >= 0.10
    )


def _determine_geometric_relationships(
    name1: str,
    obj1: dict,
    name2: str,
    obj2: dict,
    distance_threshold: float,
) -> list[tuple[str, str, str, str]]:
    relationships = []
    c1 = np.asarray(obj1["centroid"], dtype=float)
    c2 = np.asarray(obj2["centroid"], dtype=float)
    centroid_distance = float(np.linalg.norm(c1 - c2))

    obj1_above_obj2 = local_is_above(obj1, obj2)
    obj2_above_obj1 = local_is_above(obj2, obj1)

    if obj1_above_obj2:
        relationships.append((name1, name2, "above", "geometric"))
        relationships.append((name2, name1, "below", "geometric"))
    elif obj2_above_obj1:
        relationships.append((name2, name1, "above", "geometric"))
        relationships.append((name1, name2, "below", "geometric"))

    adjacent_gap = min(distance_threshold * 0.15, 0.35)
    is_adjacent = local_is_adjacent_laterally(obj1, obj2, adjacent_gap)
    if is_adjacent:
        relationships.append((name1, name2, "adjacent_to", "geometric"))
        relationships.append((name2, name1, "adjacent_to", "geometric"))

    if (
        centroid_distance <= distance_threshold
        and not obj1_above_obj2
        and not obj2_above_obj1
        and not is_adjacent
    ):
        relationships.append((name1, name2, "near", "geometric"))
        relationships.append((name2, name1, "near", "geometric"))

    return relationships


def local_is_adjacent_laterally(obj1: dict, obj2: dict, max_gap: float) -> bool:
    label1 = obj1.get("semantic_label")
    label2 = obj2.get("semantic_label")
    if "floor" in {label1, label2} and label1 != label2:
        return False
    if local_is_above(obj1, obj2) or local_is_above(obj2, obj1):
        return False

    b1 = obj1["bounds"]
    b2 = obj2["bounds"]
    if horizontal_gap(b1, b2) > max_gap:
        return False
    if axis_overlap_ratio(b1, b2, axis=2) < 0.30:
        return False

    x_overlap = axis_overlap_ratio(b1, b2, axis=0)
    y_overlap = axis_overlap_ratio(b1, b2, axis=1)
    x_gap = axis_gap(float(b1["min"][0]), float(b1["max"][0]), float(b2["min"][0]), float(b2["max"][0]))
    y_gap = axis_gap(float(b1["min"][1]), float(b1["max"][1]), float(b2["min"][1]), float(b2["max"][1]))
    return (
        (x_gap <= max_gap and y_overlap >= 0.15)
        or (y_gap <= max_gap and x_overlap >= 0.15)
    )


def horizontal_gap(left: dict, right: dict) -> float:
    gaps = [
        axis_gap(
            float(left["min"][axis]),
            float(left["max"][axis]),
            float(right["min"][axis]),
            float(right["max"][axis]),
        )
        for axis in range(2)
    ]
    return float(np.linalg.norm(gaps))


def vertical_relation_plausible(
    source: dict,
    relationship: str,
    target: dict,
    *,
    distance_threshold: float,
) -> bool:
    if relationship == "above":
        upper = source
        lower = target
    elif relationship == "below":
        upper = target
        lower = source
    else:
        return False

    if float(upper["centroid"][2]) <= float(lower["centroid"][2]):
        return False

    upper_label = upper.get("semantic_label")
    lower_label = lower.get("semantic_label")
    if not supports_label_pair(lower_label, upper_label):
        return False

    upper_bounds = upper["bounds"]
    lower_bounds = lower["bounds"]
    return (
        horizontal_gap(upper_bounds, lower_bounds) <= max(distance_threshold, 0.75)
        or overlap_xy_ratio(upper_bounds, lower_bounds) >= 0.01
    )


def local_spatial_relation_plausible(
    source: dict,
    relationship: str,
    target: dict,
    *,
    distance_threshold: float,
) -> bool:
    if not architecturally_related_pair(source, target):
        return False

    source_bounds = source["bounds"]
    target_bounds = target["bounds"]
    contact_gap = max(distance_threshold * 0.25, 0.75)

    if relationship == "adjacent_to":
        return (
            relationship_contact_confirmed(source, target, contact_gap)
            or (
                horizontal_gap(source_bounds, target_bounds) <= max(distance_threshold, 0.75)
                and axis_overlap_ratio(source_bounds, target_bounds, axis=2) >= 0.05
            )
        )

    if relationship == "near":
        source_centroid = np.asarray(source["centroid"], dtype=float)
        target_centroid = np.asarray(target["centroid"], dtype=float)
        centroid_distance = float(np.linalg.norm(source_centroid - target_centroid))
        return (
            centroid_distance <= distance_threshold * 1.25
            or bounds_gap(source_bounds, target_bounds) <= max(distance_threshold, 0.75)
            or relationship_contact_confirmed(source, target, contact_gap)
        )

    return False


def architecturally_related_pair(source: dict, target: dict) -> bool:
    source_label = source.get("semantic_label")
    target_label = target.get("semantic_label")
    return (
        supports_label_pair(source_label, target_label)
        or supports_label_pair(target_label, source_label)
        or mereological_relation_type(source_label, target_label) is not None
        or mereological_relation_type(target_label, source_label) is not None
    )


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

    y_true = [1 if row["in_gold"] else 0 for row in rows]
    y_pred = [1 if row["in_generated"] else 0 for row in rows]

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]

    summary = {
        "gold_file": str(gold_csv),
        "gold_csv": str(gold_csv),
        "generated_triples": len(generated_keys),
        "gold_triples": len(gold_keys),
        "evaluated_candidate_triples": len(rows),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision_score": round(precision, 4),
        "recall_score": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_format": "rows=y_true [0,1], columns=y_pred [0,1]",
        "metric_note": (
            "Metrics are computed on the union of generated and gold triples. "
            "True negatives outside this candidate universe are not enumerated."
        ),
    }
    return rows, summary


def read_gold_triples(path: Path) -> set[tuple[str, str, str]]:
    if path.suffix.lower() == ".xlsx":
        return read_gold_triples_xlsx(path)
    return read_gold_triples_csv(path)


def read_gold_triples_csv(path: Path) -> set[tuple[str, str, str]]:
    text = read_text_fallback(path)
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(text.splitlines(), dialect=dialect))
    return triples_from_gold_rows(rows, source_name="Gold CSV")


def read_gold_triples_xlsx(path: Path) -> set[tuple[str, str, str]]:
    rows = read_xlsx_rows(path)
    return triples_from_gold_rows(rows, source_name="Gold XLSX")


def triples_from_gold_rows(rows: list[dict], *, source_name: str) -> set[tuple[str, str, str]]:
    if not rows:
        return set()
    field_map = {field.lower().strip(): field for field in rows[0].keys() if field}
    source_col = first_existing(field_map, ("source", "src", "subject", "from"))
    rel_col = first_existing(field_map, ("relationship", "relation", "predicate", "rel"))
    target_col = first_existing(field_map, ("target", "tgt", "object", "to"))
    manual_label_col = first_existing(
        field_map,
        ("manual_label", "gold", "label", "correct", "is_correct", "manual", "valid"),
    )
    if not source_col or not rel_col or not target_col:
        raise ValueError(
            f"{source_name} must contain source/relationship/target columns "
            "(or aliases: src, relation, target)."
        )

    triples = set()
    for row in rows:
        if manual_label_col and not is_gold_positive_label(row.get(manual_label_col, "")):
            continue
        source = str(row.get(source_col, "")).strip()
        relationship = str(row.get(rel_col, "")).strip()
        target = str(row.get(target_col, "")).strip()
        if source and relationship and target:
            triples.add((source, relationship, target))
    return triples


def is_gold_positive_label(value: object) -> bool:
    normalized = normalize_manual_label(value)
    return normalized in {
        "1",
        "true",
        "yes",
        "y",
        "si",
        "s",
        "ok",
        "correct",
        "corretto",
        "corretta",
        "valid",
        "valida",
        "valido",
    }


def normalize_manual_label(value: object) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def read_xlsx_rows(path: Path) -> list[dict]:
    main_ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        worksheet_path = first_xlsx_worksheet_path(archive)
        shared_strings = read_xlsx_shared_strings(archive, main_ns)
        root = ET.fromstring(archive.read(worksheet_path))

    table_rows: list[list[str]] = []
    for row in root.findall(".//m:sheetData/m:row", main_ns):
        values: list[str] = []
        for cell in row.findall("m:c", main_ns):
            cell_ref = cell.attrib.get("r", "")
            cell_index = xlsx_cell_column_index(cell_ref)
            if cell_index is None:
                cell_index = len(values)
            while len(values) <= cell_index:
                values.append("")
            values[cell_index] = xlsx_cell_value(cell, shared_strings, main_ns)
        table_rows.append(values)

    header_index = next(
        (index for index, values in enumerate(table_rows) if any(str(value).strip() for value in values)),
        None,
    )
    if header_index is None:
        return []

    headers = [str(value).strip() for value in table_rows[header_index]]
    rows = []
    for values in table_rows[header_index + 1 :]:
        if not any(str(value).strip() for value in values):
            continue
        row = {
            header: values[index] if index < len(values) else ""
            for index, header in enumerate(headers)
            if header
        }
        rows.append(row)
    return rows


def first_xlsx_worksheet_path(archive: zipfile.ZipFile) -> str:
    names = set(archive.namelist())
    fallback = "xl/worksheets/sheet1.xml"
    if "xl/workbook.xml" not in names or "xl/_rels/workbook.xml.rels" not in names:
        return fallback

    main_ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    office_rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    package_rel_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"

    workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
    first_sheet = workbook_root.find(".//m:sheet", main_ns)
    if first_sheet is None:
        return fallback
    relationship_id = first_sheet.attrib.get(office_rel_ns)
    if not relationship_id:
        return fallback

    rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for relationship in rels_root.findall(f"{package_rel_ns}Relationship"):
        if relationship.attrib.get("Id") != relationship_id:
            continue
        target = relationship.attrib.get("Target", "")
        if not target:
            return fallback
        if target.startswith("/"):
            return target.lstrip("/")
        return posixpath.normpath(posixpath.join("xl", target))
    return fallback


def read_xlsx_shared_strings(archive: zipfile.ZipFile, main_ns: dict[str, str]) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall("m:si", main_ns):
        strings.append("".join(text_node.text or "" for text_node in item.findall(".//m:t", main_ns)))
    return strings


def xlsx_cell_value(cell: ET.Element, shared_strings: list[str], main_ns: dict[str, str]) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(text_node.text or "" for text_node in cell.findall(".//m:t", main_ns))

    value_node = cell.find("m:v", main_ns)
    value = value_node.text if value_node is not None and value_node.text is not None else ""
    if cell_type == "s":
        try:
            return shared_strings[int(value)]
        except (ValueError, IndexError):
            return ""
    return value


def xlsx_cell_column_index(cell_ref: str) -> int | None:
    letters = "".join(char for char in cell_ref if char.isalpha()).upper()
    if not letters:
        return None
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


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


def add_blank_manual_fields(rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        item = dict(row)
        for field in MANUAL_REVIEW_FIELDS:
            item.setdefault(field, "")
        enriched.append(item)
    return enriched


def write_xlsx(path: Path, rows: list[dict], fieldnames: list[str], sheet_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_sheet_name = sanitize_sheet_name(sheet_name)
    row_count = len(rows) + 1
    col_count = len(fieldnames)
    last_cell = f"{excel_col(col_count)}{max(row_count, 1)}"

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", xlsx_content_types())
        archive.writestr("_rels/.rels", xlsx_root_rels())
        archive.writestr("docProps/app.xml", xlsx_app_props(safe_sheet_name))
        archive.writestr("docProps/core.xml", xlsx_core_props())
        archive.writestr("xl/workbook.xml", xlsx_workbook(safe_sheet_name))
        archive.writestr("xl/_rels/workbook.xml.rels", xlsx_workbook_rels())
        archive.writestr("xl/styles.xml", xlsx_styles())
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            xlsx_sheet_xml(rows, fieldnames, ref=f"A1:{last_cell}"),
        )


def sanitize_sheet_name(value: str) -> str:
    invalid_chars = set("[]:*?/\\")
    cleaned = "".join("_" if char in invalid_chars else char for char in value)
    return (cleaned[:31] or "Sheet1").strip("'") or "Sheet1"


def excel_col(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters or "A"


def xlsx_cell(row_index: int, col_index: int, value: object, style: int = 0) -> str:
    cell_ref = f"{excel_col(col_index)}{row_index}"
    text = "" if value is None else str(value)
    escaped = escape(text, {'"': "&quot;"})
    style_attr = f' s="{style}"' if style else ""
    return f'<c r="{cell_ref}" t="inlineStr"{style_attr}><is><t>{escaped}</t></is></c>'


def xlsx_sheet_xml(rows: list[dict], fieldnames: list[str], ref: str) -> str:
    column_xml = "".join(
        f'<col min="{index}" max="{index}" width="{xlsx_column_width(field)}" customWidth="1"/>'
        for index, field in enumerate(fieldnames, start=1)
    )
    row_xml = []
    header_cells = "".join(
        xlsx_cell(1, col_index, field, style=1)
        for col_index, field in enumerate(fieldnames, start=1)
    )
    row_xml.append(f'<row r="1">{header_cells}</row>')

    for row_index, row in enumerate(rows, start=2):
        cells = "".join(
            xlsx_cell(row_index, col_index, row.get(field, ""))
            for col_index, field in enumerate(fieldnames, start=1)
        )
        row_xml.append(f'<row r="{row_index}">{cells}</row>')

    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheetViews>
    <sheetView tabSelected="1" workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
      <selection pane="bottomLeft" activeCell="A2" sqref="A2"/>
    </sheetView>
  </sheetViews>
  <cols>{column_xml}</cols>
  <sheetData>{''.join(row_xml)}</sheetData>
  <autoFilter ref="{ref}"/>
</worksheet>'''


def xlsx_column_width(field: str) -> int:
    if field in {"source", "target", "relationship", "status", "manual_label", "manual_issue"}:
        return 24
    if field.endswith("_issues") or field.endswith("_note") or field in {"manual_note"}:
        return 42
    if field in {"source_label", "target_label", "derivation_level"}:
        return 22
    return min(max(len(field) + 2, 12), 28)


def xlsx_content_types() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>'''


def xlsx_root_rels() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''


def xlsx_workbook(sheet_name: str) -> str:
    escaped_sheet = escape(sheet_name, {'"': "&quot;"})
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="{escaped_sheet}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>'''


def xlsx_workbook_rels() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''


def xlsx_styles() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="2">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
  </cellXfs>
</styleSheet>'''


def xlsx_app_props(sheet_name: str) -> str:
    escaped_sheet = escape(sheet_name)
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>arch-agent</Application>
  <TitlesOfParts><vt:vector size="1" baseType="lpstr"><vt:lpstr>{escaped_sheet}</vt:lpstr></vt:vector></TitlesOfParts>
</Properties>'''


def xlsx_core_props() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>arch-agent</dc:creator>
  <cp:lastModifiedBy>arch-agent</cp:lastModifiedBy>
</cp:coreProperties>'''


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
