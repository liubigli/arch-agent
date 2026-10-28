"""Compare automatic triple validation with a manually reviewed gold file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CORRECT_LABELS = {"correct", "valid", "true", "1", "yes", "si", "sì"}
INCORRECT_LABELS = {"incorrect", "invalid", "false", "0", "no"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manual_review_file", help="Reviewed .xlsx or .csv file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--blank-means-confirm",
        action="store_true",
        help="Treat blank manual labels as confirmation of the validator status.",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def normalized_manual_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    label = str(value).strip().lower()
    if not label:
        return None
    if label in CORRECT_LABELS:
        return "correct"
    if label in INCORRECT_LABELS:
        return "incorrect"
    raise ValueError(f"Unknown manual_label value: {value!r}")


def normalized_validator_status(value: object) -> str:
    status = str(value).strip().lower()
    if status == "valid":
        return "valid"
    if status in {"invalid", "review"}:
        return "invalid"
    raise ValueError(f"Unknown automatic status value: {value!r}")


def main() -> None:
    args = parse_args()
    source = Path(args.manual_review_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = read_table(source)
    required = {"source", "relationship", "target", "status", "manual_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    records = []
    unreviewed = 0
    for row in frame.to_dict(orient="records"):
        validator_status = normalized_validator_status(row["status"])
        manual_label = normalized_manual_label(row.get("manual_label"))
        if manual_label is None:
            unreviewed += 1
            if not args.blank_means_confirm:
                continue
            manual_label = "correct" if validator_status == "valid" else "incorrect"

        manual_correct = manual_label == "correct"
        validator_correct = validator_status == "valid"
        if manual_correct and validator_correct:
            confusion_class = "TP"
        elif not manual_correct and validator_correct:
            confusion_class = "FP"
        elif manual_correct and not validator_correct:
            confusion_class = "FN"
        else:
            confusion_class = "TN"

        record = dict(row)
        record.update(
            {
                "validator_status": validator_status,
                "manual_label": manual_label,
                "validator_predicted_correct": validator_correct,
                "manual_correct": manual_correct,
                "confusion_class": confusion_class,
            }
        )
        records.append(record)

    if not records:
        raise ValueError("No manually reviewed triples were available for evaluation.")

    y_true = [int(row["manual_correct"]) for row in records]
    y_pred = [int(row["validator_predicted_correct"]) for row in records]
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    summary = {
        "source_file": str(source),
        "total_rows_in_file": int(len(frame)),
        "evaluated_triples": len(records),
        "unreviewed_rows": unreviewed,
        "blank_means_confirm": bool(args.blank_means_confirm),
        "manual_correct": int(sum(y_true)),
        "manual_incorrect": int(len(y_true) - sum(y_true)),
        "validator_valid": int(sum(y_pred)),
        "validator_invalid": int(len(y_pred) - sum(y_pred)),
        "confusion_matrix_format": "[[TN, FP], [FN, TP]]",
        "confusion_matrix": matrix.tolist(),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "precision_score": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_score": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy_score": round(float(accuracy_score(y_true, y_pred)), 4),
        "manual_acceptance_rate": round(float(sum(y_true) / len(y_true)), 4),
    }

    stem = source.stem.replace("spatial_triples_manual_review_", "")
    json_path = output_dir / f"manual_evaluation_{stem}.json"
    csv_path = output_dir / f"manual_evaluation_{stem}.csv"
    xlsx_path = output_dir / f"manual_evaluation_{stem}.xlsx"
    report_path = output_dir / f"manual_evaluation_{stem}.txt"
    matrix_path = output_dir / f"manual_evaluation_{stem}_confusion_matrix.png"

    json_path.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result_frame = pd.DataFrame(records)
    result_frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    result_frame.to_excel(xlsx_path, index=False)
    report_path.write_text(
        "Manual vs automatic triple validation\n\n"
        + "\n".join(f"{key}: {value}" for key, value in summary.items())
        + "\n",
        encoding="utf-8",
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["incorrect triple", "correct triple"],
    )
    figure, axis = plt.subplots(figsize=(8, 6))
    display.plot(ax=axis, cmap="viridis", colorbar=True, values_format="d")
    axis.set_title("Automatic validator vs manual review")
    axis.set_xlabel("Automatic validator")
    axis.set_ylabel("Manual review")
    figure.tight_layout()
    figure.savefig(matrix_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nReport: {report_path}")
    print(f"Matrix: {matrix_path}")
    print(f"Records: {xlsx_path}")


if __name__ == "__main__":
    main()
