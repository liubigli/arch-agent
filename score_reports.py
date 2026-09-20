"""Score benchmark reports that already exist, without re-running any model.

Scoring is decoupled from execution, so improving the validator does not cost
a GPU hour: point this at the JSON reports a run produced and it recomputes
the metrics against the approved structured reference.

Reports written before 2026-09-17 carry no question_id - the field was added
after the 16 September run - so the question is resolved by matching its text
against the question file, falling back to file order when the texts match
exactly and in sequence.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from arch_agent.benchmark.scoring import (
    CORRECT,
    INCORRECT,
    NOT_SCORED,
    PARTIAL,
    aggregate,
    score_answer,
)
from arch_agent.benchmark.structured_reference import (
    load_structured_reference,
    reference_for_question,
)

DEFAULT_QUESTIONS = "benchmark/domande_per_scene.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-score existing benchmark reports against the structured reference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("reports", nargs="+", help="Benchmark report JSON files (raw or evaluation).")
    parser.add_argument(
        "--reference-file",
        default=None,
        help="Structured reference JSON. Default: benchmark/references/<scene>_reference_draft.json",
    )
    parser.add_argument("--questions-file", default=DEFAULT_QUESTIONS)
    parser.add_argument("--scene", default="scena4_VAL", help="Scene name used to find the reference.")
    parser.add_argument("--output", default=None, help="Optional path to write the scores as JSON.")
    parser.add_argument("--show", type=int, default=0, help="Print this many scored questions per report (0 = none).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reference_path = Path(
        args.reference_file
        or f"benchmark/references/{args.scene}_reference_draft.json"
    )
    payload = load_structured_reference(reference_path)
    questions = load_questions(Path(args.questions_file))
    print(f"Reference : {reference_path} ({payload.get('status')})")
    print(f"Questions : {len(questions)} from {args.questions_file}\n")

    report_summaries = {}
    for report_path in args.reports:
        path = Path(report_path)
        records = read_records(path)
        if not records:
            print(f"{path}: no records, skipped")
            continue

        resolution = resolve_question_ids(records, questions)
        scores = []
        for record, question_id in zip(records, resolution["ids"]):
            spec = reference_for_question(payload, question_id)
            scores.append(
                score_answer(
                    record.get("final_answer"),
                    spec,
                    tool_output=tool_output_text(record),
                )
            )

        summary = aggregate(scores)
        summary["resolution"] = {k: v for k, v in resolution.items() if k != "ids"}
        summary["records"] = len(records)
        report_summaries[str(path)] = summary
        print_report(path, records, scores, summary, show=args.show)

    if args.output:
        Path(args.output).write_text(
            json.dumps(report_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nWritten: {args.output}")


def load_questions(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return [line.strip() for line in lines if line.strip().endswith("?")]


def read_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("results", "records", "questions"):
        if isinstance(payload.get(key), list):
            return payload[key]
    return []


def resolve_question_ids(records: list[dict], questions: list[str]) -> dict:
    """Give every record a question id, by field, by text, or by position."""
    by_text = {normalize(q): index + 1 for index, q in enumerate(questions)}
    ids: list[int | None] = []
    counts = {"from_field": 0, "from_text": 0, "from_position": 0, "unresolved": 0}

    positional_ok = len(records) == len(questions) and all(
        normalize(record.get("question", "")) == normalize(question)
        for record, question in zip(records, questions)
    )

    for index, record in enumerate(records):
        question_id = record.get("question_id")
        if isinstance(question_id, int):
            ids.append(question_id)
            counts["from_field"] += 1
            continue
        matched = by_text.get(normalize(record.get("question", "")))
        if matched is not None:
            ids.append(matched)
            counts["from_text"] += 1
            continue
        if positional_ok:
            ids.append(index + 1)
            counts["from_position"] += 1
            continue
        ids.append(None)
        counts["unresolved"] += 1

    return {"ids": ids, **counts}


def tool_output_text(record: dict) -> str:
    calls = record.get("tool_calls") or []
    parts = []
    for call in calls:
        if isinstance(call, dict):
            parts.append(str(call.get("output") or ""))
        else:
            parts.append(str(call))
    return "\n".join(parts)


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", (text or "").lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", folded).strip()


def print_report(path: Path, records, scores, summary: dict, show: int) -> None:
    resolution = summary["resolution"]
    print(f"== {path}")
    print(
        f"   records {summary['records']} | ids: field={resolution['from_field']} "
        f"text={resolution['from_text']} position={resolution['from_position']} "
        f"unresolved={resolution['unresolved']}"
    )
    if resolution["unresolved"]:
        print(
            "   WARNING: unresolved questions are not scored. This report was "
            "probably produced with a different question set."
        )
    print(
        f"   scored {summary['scored']}/{summary['questions']} "
        f"(coverage {summary['coverage']:.0%}) | accuracy {summary['accuracy']:.1%} "
        f"| partial credit {summary['partial_credit']:.1%}"
    )
    for family, stats in sorted(summary["by_family"].items()):
        print(f"     {family:14} n={stats['n']:3}  accuracy={stats['accuracy']:.1%}")

    if show:
        shown = [s for s in scores if s.outcome in (INCORRECT, PARTIAL)][:show]
        for score in shown:
            print(f"     Q{score.question_id:<3} {score.outcome:9} {score.detail[:88]}")
    print()


if __name__ == "__main__":
    main()
