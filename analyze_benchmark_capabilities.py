"""Compare benchmark capability and tool routing directly from raw outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from arch_agent.benchmark.scoring import score_answer
from arch_agent.benchmark.structured_reference import (
    load_structured_reference,
    reference_for_question,
)


CSV_TOOLS = {
    "get_object_annotation",
    "list_csv_annotation_matches",
    "find_objects_by_material",
    "get_object_semantic_details",
}

INTENT_CAPABILITY = {
    "scene_summary": "scene_understanding",
    "interior_exterior_classification": "scene_understanding",
    "main_elements": "scene_understanding",
    "high_confidence_elements": "scene_understanding",
    "observation_inference_separation": "reasoning_quality",
    "confidence_assessment": "reasoning_quality",
    "class_material": "semantic_metadata",
    "class_material_per_object": "semantic_metadata",
    "class_function": "semantic_metadata",
    "per_object_semantic_details": "semantic_metadata",
    "material_filtered_presence": "semantic_metadata",
    "material_filtered_positions": "semantic_metadata",
    "class_role": "semantic_roles",
    "class_count_whith_function": "semantic_roles",
    "class_relationship_summary": "relationships",
    "supported_targets": "relationships",
    "relationship_exists_with_absent_class": "absence_reasoning",
    "absent_class_relationships": "absence_reasoning",
    "absent_class_function": "absence_reasoning",
    "absent_class_count": "absence_reasoning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dir", required=True)
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_structured_reference(args.reference)

    rows = []
    for condition, folder in (("full_csv", args.full_dir), ("graph_no_csv", args.graph_dir)):
        raw_files = sorted(Path(folder).glob("benchmark_raw_*.json"))
        if not raw_files:
            raise FileNotFoundError(f"No raw benchmark JSON files in {folder}")
        for raw_path in raw_files:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            if payload.get("questions_loaded") != 60 or len(payload.get("records", [])) != 60:
                raise ValueError(f"Expected 60 questions in {raw_path}")
            evaluation = _load_matching_evaluation(raw_path)
            evaluation_by_id = {
                int(item["question_id"]): item for item in evaluation.get("records", [])
            }
            for record in payload["records"]:
                qid = int(record["question_id"])
                spec = reference_for_question(reference, qid)
                rows.append(
                    _analyze_record(
                        condition,
                        payload,
                        record,
                        spec,
                        evaluation_by_id.get(qid, {}),
                        raw_path,
                    )
                )

    summary = _summarize(rows)
    capability = _capability_summary(rows)
    tool_usage = _tool_usage(rows)
    deltas = _condition_deltas(summary)

    _write_csv(output_dir / "question_level_audit.csv", rows)
    _write_csv(output_dir / "model_metrics.csv", summary)
    _write_csv(output_dir / "capability_breakdown.csv", capability)
    _write_csv(output_dir / "tool_usage.csv", tool_usage)
    _write_csv(output_dir / "condition_deltas.csv", deltas)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_metrics": summary,
                "capability_breakdown": capability,
                "tool_usage": tool_usage,
                "condition_deltas": deltas,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _report(summary, capability, tool_usage, deltas),
        encoding="utf-8",
    )
    print(f"Analyzed records: {len(rows)}")
    print(f"Output: {output_dir}")


def _load_matching_evaluation(raw_path: Path) -> dict:
    evaluation_name = raw_path.name.replace("benchmark_raw_", "benchmark_evaluation_", 1)
    evaluation_path = raw_path.with_name(evaluation_name)
    if not evaluation_path.exists():
        return {"records": []}
    return json.loads(evaluation_path.read_text(encoding="utf-8"))


def _analyze_record(condition, payload, record, spec, legacy, raw_path):
    calls = record.get("tool_calls") or []
    called = [call.get("name", "") for call in calls]
    outputs = "\n".join(str(call.get("output") or "") for call in calls)
    preferred = list((spec or {}).get("preferred_tools") or [])
    acceptable = list((spec or {}).get("acceptable_tools") or [])
    route_status, route_scored = _route_status(condition, called, preferred, acceptable)
    args_status, args_detail = _argument_status(calls, spec)
    score = score_answer(record.get("final_answer"), spec, tool_output=outputs)
    expected_without_csv = (spec or {}).get("expected_without_csv", "unspecified")
    abstention_ok = _abstention_status(
        condition,
        expected_without_csv,
        record.get("final_answer") or "",
    )
    unsupported_ids = sorted(
        _object_ids(record.get("final_answer") or "")
        - _object_ids(outputs)
        - _object_ids(record.get("question") or "")
    )
    invalid_tool_calls = sum(
        "not a valid tool" in str(call.get("output") or "").lower()
        for call in calls
    )
    answer_present = bool((record.get("final_answer") or "").strip())
    operational_success = (
        answer_present
        and not record.get("error")
        and (not route_scored or route_status in {"optimal", "acceptable"})
        and args_status in {"pass", "not_applicable"}
        and not unsupported_ids
        and abstention_ok is not False
    )
    intent = (spec or {}).get("intent", "unknown")
    return {
        "condition": condition,
        "source_date": payload.get("date", ""),
        "model": payload.get("model", record.get("model", "")),
        "think": payload.get("think", ""),
        "question_id": record.get("question_id"),
        "question": record.get("question", ""),
        "intent": intent,
        "capability": _capability(intent),
        "validation_mode": (spec or {}).get("validation_mode", ""),
        "expected_without_csv": expected_without_csv,
        "preferred_tools": " | ".join(preferred),
        "acceptable_tools": " | ".join(acceptable),
        "tools_called": " | ".join(called),
        "tool_args": json.dumps([call.get("args") or {} for call in calls], ensure_ascii=False),
        "tool_call_count": len(calls),
        "invalid_tool_calls": invalid_tool_calls,
        "route_status": route_status,
        "route_scored": route_scored,
        "arguments_status": args_status,
        "arguments_detail": args_detail,
        "answer_score_outcome": score.outcome,
        "answer_score": score.score,
        "answer_score_family": score.family,
        "answer_score_detail": score.detail,
        "abstention_ok": abstention_ok,
        "unsupported_object_ids": " | ".join(unsupported_ids),
        "operational_success": operational_success,
        "legacy_reliability": legacy.get("reliability", ""),
        "language_ok": legacy.get("language_ok", ""),
        "latency_s": record.get("latency_s", 0),
        "error": record.get("error") or "",
        "final_answer": record.get("final_answer") or "",
        "tool_outputs": outputs,
        "raw_file": raw_path.name,
    }


def _route_status(condition, called, preferred, acceptable):
    effective_preferred = list(preferred)
    effective_acceptable = list(acceptable)
    if condition == "graph_no_csv":
        effective_preferred = [tool for tool in preferred if tool not in CSV_TOOLS]
        effective_acceptable = [tool for tool in acceptable if tool not in CSV_TOOLS]
        if preferred and not effective_preferred:
            return "not_applicable_withheld", False
    if not effective_preferred:
        return "not_applicable", False
    if not called:
        return "no_tool", True
    if any(tool in called for tool in effective_preferred):
        return "optimal", True
    if any(tool in called for tool in effective_acceptable):
        return "acceptable", True
    return "suboptimal", True


def _argument_status(calls, spec):
    required = (spec or {}).get("required_tool_args") or {}
    if not required:
        return "not_applicable", ""
    if not calls:
        return "fail", "no tool call"
    preferred = set((spec or {}).get("preferred_tools") or [])
    acceptable = set((spec or {}).get("acceptable_tools") or [])
    candidates = [call for call in calls if call.get("name") in preferred | acceptable] or calls
    failures = []
    for key, expected in required.items():
        if key == "semantic_labels":
            if expected is None:
                if not any(
                    not (call.get("args") or {}).get("semantic_labels")
                    and not (call.get("args") or {}).get("semantic_label")
                    for call in candidates
                ):
                    failures.append("expected no semantic class filter")
                continue
            expected_set = {_canonical_label(item) for item in expected}
            if not any(_labels_from_args(call.get("args") or {}) == expected_set for call in candidates):
                failures.append(f"semantic_labels expected={sorted(expected_set)}")
        elif key == "semantic_label":
            if not any(
                _canonical_label((call.get("args") or {}).get("semantic_label", ""))
                == _canonical_label(expected)
                for call in candidates
            ):
                failures.append(f"semantic_label expected={expected}")
        elif key == "material_aliases":
            aliases = {_normalize_text(item) for item in expected}
            if not any(
                any(alias in _normalize_text(json.dumps(call.get("args") or {}, ensure_ascii=False)) for alias in aliases)
                for call in candidates
            ):
                failures.append("material alias missing")
        elif not any((call.get("args") or {}).get(key) == expected for call in candidates):
            failures.append(f"{key} expected={expected!r}")
    return ("fail", "; ".join(failures)) if failures else ("pass", "")


def _labels_from_args(args):
    values = args.get("semantic_labels")
    if isinstance(values, str):
        values = [values]
    if not values and args.get("semantic_label"):
        values = [args["semantic_label"]]
    return {_canonical_label(item) for item in (values or []) if item}


def _canonical_label(value):
    normalized = _normalize_text(str(value)).replace(" ", "_")
    aliases = {
        "colonna": "column", "colonne": "column", "muro": "wall", "muri": "wall",
        "parete": "wall", "pareti": "wall", "volta": "vault", "volte": "vault",
        "tetto": "roof", "tetti": "roof", "arco": "arch", "archi": "arch",
        "scala": "stairs", "scale": "stairs", "pavimento": "floor", "pavimenti": "floor",
        "porta": "door_window", "porte": "door_window", "finestra": "door_window",
        "finestre": "door_window", "apertura": "door_window", "aperture": "door_window",
        "molding": "moldings", "modanatura": "moldings", "modanature": "moldings",
    }
    return aliases.get(normalized, normalized)


def _abstention_status(condition, policy, answer):
    if condition != "graph_no_csv" or policy == "unspecified":
        return None
    if policy == "manual_review":
        return None
    text = _normalize_text(answer)
    markers = (
        "non disponibile", "non sono disponibili", "non e disponibile", "manca il csv",
        "senza csv", "nessuna annotazione", "no annotation", "not available",
        "cannot determine", "cannot be determined", "non posso determinare",
        "non e possibile determinare", "non posso stabilire",
    )
    abstains = any(marker in text for marker in markers)
    if policy in {"abstention", "abstention_or_role_only"}:
        return abstains
    return None


def _object_ids(text):
    return set(re.findall(r"\b(?:arch|column|door_window|floor|moldings|other|roof|stairs|vault|wall)_\d+\b", text.lower()))


def _capability(intent):
    if intent in INTENT_CAPABILITY:
        return INTENT_CAPABILITY[intent]
    if "count" in intent or intent in {"present_classes", "absent_classes", "full_inventory_with_counts", "class_distribution", "quantity_of_elements"}:
        return "inventory_counts"
    if "object_list" in intent:
        return "object_identification"
    return "other"


def _summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["model"])].append(row)
    result = []
    for (condition, model), items in sorted(grouped.items()):
        route = [row for row in items if row["route_scored"]]
        args = [row for row in items if row["arguments_status"] != "not_applicable"]
        scored = [row for row in items if row["answer_score_outcome"] != "not_scored"]
        abstention = [row for row in items if row["abstention_ok"] is not None]
        result.append({
            "condition": condition,
            "model": model,
            "questions": len(items),
            "routing_scored": len(route),
            "optimal_routing_pct": _pct(route, lambda r: r["route_status"] == "optimal"),
            "acceptable_routing_pct": _pct(route, lambda r: r["route_status"] in {"optimal", "acceptable"}),
            "clean_acceptable_routing_pct": _pct(
                route,
                lambda r: r["route_status"] in {"optimal", "acceptable"}
                and int(r["invalid_tool_calls"]) == 0,
            ),
            "argument_accuracy_pct": _pct(args, lambda r: r["arguments_status"] == "pass"),
            "answer_accuracy_pct": _pct(scored, lambda r: r["answer_score_outcome"] == "correct"),
            "answer_partial_credit_pct": round(100 * statistics.mean([r["answer_score"] for r in scored]), 1) if scored else None,
            "scoring_coverage_pct": round(100 * len(scored) / len(items), 1),
            "operational_success_pct": _pct(items, lambda r: r["operational_success"]),
            "abstention_accuracy_pct": _pct(abstention, lambda r: r["abstention_ok"] is True),
            "unsupported_id_answers": sum(bool(r["unsupported_object_ids"]) for r in items),
            "legacy_grounded_pct": _pct(items, lambda r: r["legacy_reliability"] == "grounded"),
            "language_accuracy_pct": _pct(items, lambda r: r["language_ok"] is True),
            "avg_tool_calls": round(statistics.mean(r["tool_call_count"] for r in items), 2),
            "invalid_tool_attempts": sum(int(r["invalid_tool_calls"]) for r in items),
            "questions_with_invalid_tool": sum(int(r["invalid_tool_calls"]) > 0 for r in items),
            "avg_latency_s": round(statistics.mean(float(r["latency_s"] or 0) for r in items), 3),
            "errors": sum(bool(r["error"]) for r in items),
        })
    return result


def _capability_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["model"], row["capability"])].append(row)
    result = []
    for (condition, model, capability), items in sorted(grouped.items()):
        scored = [row for row in items if row["answer_score_outcome"] != "not_scored"]
        route = [row for row in items if row["route_scored"]]
        result.append({
            "condition": condition,
            "model": model,
            "capability": capability,
            "questions": len(items),
            "answer_scored": len(scored),
            "answer_accuracy_pct": _pct(scored, lambda r: r["answer_score_outcome"] == "correct"),
            "partial_credit_pct": round(100 * statistics.mean([r["answer_score"] for r in scored]), 1) if scored else None,
            "acceptable_routing_pct": _pct(route, lambda r: r["route_status"] in {"optimal", "acceptable"}),
            "operational_success_pct": _pct(items, lambda r: r["operational_success"]),
        })
    return result


def _tool_usage(rows):
    grouped = Counter()
    for row in rows:
        for tool in filter(None, row["tools_called"].split(" | ")):
            grouped[(row["condition"], row["model"], tool)] += 1
    totals = Counter((row["condition"], row["model"]) for row in rows)
    return [
        {
            "condition": condition,
            "model": model,
            "tool": tool,
            "calls": count,
            "calls_per_question": round(count / totals[(condition, model)], 3),
        }
        for (condition, model, tool), count in sorted(grouped.items())
    ]


def _condition_deltas(summary):
    by_model = defaultdict(dict)
    for row in summary:
        by_model[row["model"]][row["condition"]] = row
    metrics = (
        "optimal_routing_pct", "acceptable_routing_pct", "clean_acceptable_routing_pct", "argument_accuracy_pct",
        "answer_accuracy_pct", "answer_partial_credit_pct", "operational_success_pct",
        "legacy_grounded_pct", "language_accuracy_pct", "avg_tool_calls", "avg_latency_s",
    )
    result = []
    for model, conditions in sorted(by_model.items()):
        if "full_csv" not in conditions or "graph_no_csv" not in conditions:
            continue
        row = {"model": model}
        for metric in metrics:
            full = conditions["full_csv"].get(metric)
            graph = conditions["graph_no_csv"].get(metric)
            row[f"full_{metric}"] = full
            row[f"graph_{metric}"] = graph
            row[f"delta_graph_minus_full_{metric}"] = round(graph - full, 3) if full is not None and graph is not None else None
        result.append(row)
    return result


def _report(summary, capability, tool_usage, deltas):
    lines = [
        "# Benchmark capability and tool-routing analysis",
        "",
        "Primary source: raw JSON records. The saved evaluation is used only for legacy groundedness and language checks.",
        "The full-CSV and graph-only runs were produced on different dates and may use different code revisions; deltas are descriptive, not causal.",
        "The no-CSV abstention policy is marked as proposed in the approved reference and is reported separately.",
        "",
        "## Model summary",
        "",
        "| Condition | Model | Preferred routing | Acceptable routing | Clean routing | Args | Answer accuracy | Partial credit | Operational success | Invalid tool attempts | Avg latency |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['condition']} | {row['model']} | {_fmt(row['optimal_routing_pct'])} | "
            f"{_fmt(row['acceptable_routing_pct'])} | {_fmt(row['clean_acceptable_routing_pct'])} | {_fmt(row['argument_accuracy_pct'])} | "
            f"{_fmt(row['answer_accuracy_pct'])} | {_fmt(row['answer_partial_credit_pct'])} | "
            f"{_fmt(row['operational_success_pct'])} | {row['invalid_tool_attempts']} | "
            f"{row['avg_latency_s']:.3f}s |"
        )
    lines.extend(["", "## Interpretation", ""])
    for condition in ("full_csv", "graph_no_csv"):
        subset = [row for row in summary if row["condition"] == condition]
        if not subset:
            continue
        best_route = max(subset, key=lambda row: row["acceptable_routing_pct"] or -1)
        best_answer = max(subset, key=lambda row: row["answer_partial_credit_pct"] or -1)
        best_operational = max(subset, key=lambda row: row["operational_success_pct"] or -1)
        lines.append(
            f"- **{condition}:** best acceptable routing: {best_route['model']} "
            f"({_fmt(best_route['acceptable_routing_pct'])}); best answer partial credit: "
            f"{best_answer['model']} ({_fmt(best_answer['answer_partial_credit_pct'])}); "
            f"best operational success: {best_operational['model']} "
            f"({_fmt(best_operational['operational_success_pct'])})."
        )
    lines.extend([
        "",
        "## Metric definitions",
        "",
        "- Preferred routing: at least one preferred reference tool was called.",
        "- Acceptable routing: a preferred or explicitly acceptable tool was called.",
        "- Clean routing: acceptable routing without first attempting a tool unavailable in that condition.",
        "- Argument accuracy: required semantic classes/material filters match the structured reference.",
        "- Answer accuracy: exact correctness on automatically scorable reference questions.",
        "- Partial credit: mean structured-reference score across scorable questions.",
        "- Operational success: answered, no runtime error, acceptable routing, valid required arguments, no unsupported object IDs, and correct no-CSV abstention when applicable.",
        "- Legacy groundedness: secondary metric copied from the saved evaluation, not the primary judgment source.",
        "",
        "## Files",
        "",
        "See question_level_audit.csv for every question, tool call, argument verdict, score, and answer.",
    ])
    return "\n".join(lines) + "\n"


def _pct(items, predicate):
    return round(100 * sum(bool(predicate(item)) for item in items) / len(items), 1) if items else None


def _fmt(value):
    return "n/a" if value is None else f"{value:.1f}%"


def _normalize_text(value):
    text = unicodedata.normalize("NFKD", str(value).lower())
    return " ".join("".join(char for char in text if not unicodedata.combining(char)).split())


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
