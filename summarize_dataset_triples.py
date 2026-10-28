"""Aggregate per-scene spatial-triple validation reports for a dataset run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_directory)
    report_files = sorted(run_dir.glob("*/spatial_triples_raw_*.json"))
    if not report_files:
        raise SystemExit(f"No spatial raw reports found under: {run_dir}")

    summaries = []
    relation_names: set[str] = set()
    status_names: set[str] = set()
    totals_relations: Counter[str] = Counter()
    totals_statuses: Counter[str] = Counter()

    for path in report_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        relationships = summary.get("relationship_counts", {})
        statuses = summary.get("validation_status_counts", {})
        relation_names.update(relationships)
        status_names.update(statuses)
        totals_relations.update(relationships)
        totals_statuses.update(statuses)
        summaries.append(
            {
                "scene": summary.get("scene", path.parent.name),
                "objects": int(summary.get("objects", 0)),
                "triples": int(summary.get("triples", 0)),
                "relationship_counts": relationships,
                "status_counts": statuses,
                "output_directory": str(path.parent),
            }
        )

    ordered_relations = sorted(relation_names)
    ordered_statuses = sorted(status_names)
    rows = []
    for item in summaries:
        row = {
            "scene": item["scene"],
            "objects": item["objects"],
            "triples": item["triples"],
        }
        row.update(
            {f"relation_{name}": item["relationship_counts"].get(name, 0) for name in ordered_relations}
        )
        row.update(
            {f"status_{name}": item["status_counts"].get(name, 0) for name in ordered_statuses}
        )
        row["output_directory"] = item["output_directory"]
        rows.append(row)

    total_row = {
        "scene": "TOTAL",
        "objects": sum(item["objects"] for item in summaries),
        "triples": sum(item["triples"] for item in summaries),
    }
    total_row.update({f"relation_{name}": totals_relations[name] for name in ordered_relations})
    total_row.update({f"status_{name}": totals_statuses[name] for name in ordered_statuses})
    total_row["output_directory"] = str(run_dir)
    rows.append(total_row)

    fields = list(rows[0])
    csv_path = run_dir / "dataset_spatial_triples_summary.csv"
    json_path = run_dir / "dataset_spatial_triples_summary.json"
    txt_path = run_dir / "dataset_spatial_triples_summary.txt"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps({"scenes": summaries, "totals": total_row}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "Dataset spatial-triple validation summary",
        "",
        f"Scenes completed: {len(summaries)}",
        f"Objects: {total_row['objects']}",
        f"Triples: {total_row['triples']}",
        "",
        "Relationship totals:",
    ]
    lines.extend(f"- {name}: {totals_relations[name]}" for name in ordered_relations)
    lines.extend(["", "Validation status totals:"])
    lines.extend(f"- {name}: {totals_statuses[name]}" for name in ordered_statuses)
    lines.extend(["", "Per scene:"])
    lines.extend(
        f"- {item['scene']}: objects={item['objects']}, triples={item['triples']}"
        for item in summaries
    )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(txt_path.read_text(encoding="utf-8"))
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
