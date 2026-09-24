"""What each model does when the data it is asked for has been withheld.

The graph condition removes the CSV layer, so eleven questions - the material,
typology and function ones - have no reachable answer. The correct behaviour is
to say so. This draws what each model does instead.

Three outcomes, and keeping them apart is the point: abstaining is right,
inventing a value is the failure the ablation exists to measure, and returning
nothing is a third thing altogether - the agent loop breaking, not the model
hallucinating. Collapsing the last two would make a model that produces no
answer look like one that makes things up.

    python plot_ablation.py <no-csv raw report json>... -o docs/figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from arch_agent.benchmark.scoring import CORRECT, score_answer  # noqa: E402
from arch_agent.benchmark.structured_reference import (  # noqa: E402
    load_structured_reference,
    reference_for_question,
)

# The documented diverging poles for the two evaluative outcomes, and a neutral
# for the third: an empty answer is the absence of an outcome, not a worse one.
ABSTAINS = "#2a78d6"
INVENTS = "#e34948"
EMPTY = "#c3c2b7"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

WITHHELD = ("abstention", "abstention_or_role_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--reference-file", default="benchmark/references/scena4_VAL_reference_draft.json")
    parser.add_argument("-o", "--output-dir", default="docs/figures")
    parser.add_argument("--name", default="benchmark_withheld_csv_outcomes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_structured_reference(args.reference_file)
    question_ids = [
        entry["question_id"] for entry in payload["references"]
        if entry.get("expected_without_csv") in WITHHELD
    ]

    rows = [tally(Path(report), payload, question_ids) for report in sorted(args.reports)]
    rows.sort(key=lambda r: (r["abstains"], -r["invents"]), reverse=True)
    draw(rows, len(question_ids), args)
    for row in rows:
        print(f"  {row['model']:14} si astiene {row['abstains']:2} | "
              f"inventa {row['invents']:2} | vuota {row['empty']:2}")


def tally(path: Path, payload: dict, question_ids: list[int]) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = {r["question_id"]: r for r in (data["records"] if isinstance(data, dict) else data)}
    out = {"model": model_name(path), "abstains": 0, "invents": 0, "empty": 0}
    for question_id in question_ids:
        record = records.get(question_id)
        answer = (record or {}).get("final_answer") or ""
        if not answer.strip():
            out["empty"] += 1
            continue
        score = score_answer(answer, reference_for_question(payload, question_id), condition="graph")
        out["abstains" if score.outcome == CORRECT else "invents"] += 1
    return out


def model_name(path: Path) -> str:
    stem = re.sub(r"^benchmark_raw_scena4_VAL_|_test_\d+$", "", path.stem)
    stem = re.sub(r"_think_(true|false)|_cond_(graph|full|none)|_\d{8}$", "", stem)
    return (stem.replace("gpt_oss_20b", "gpt-oss:20b").replace("gemma4_31b", "gemma4:31b")
                .replace("qwen3_5", "qwen3.5").replace("llama3_1", "llama3.1")
                .replace("command_r", "command-r"))


def draw(rows, total: int, args) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 0.66 * len(rows) + 2.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for index, row in enumerate(rows):
        left = 0.0
        for key, color in (("abstains", ABSTAINS), ("invents", INVENTS), ("empty", EMPTY)):
            value = row[key]
            if not value:
                left += value
                continue
            # 2px surface gap between adjacent segments
            ax.barh(index, value, left=left, height=0.62, color=color,
                    edgecolor=SURFACE, linewidth=1.6)
            ax.text(left + value / 2, index, str(value), ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="#ffffff" if key != "empty" else INK)
            left += value

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["model"] for r in rows], fontsize=11.5, color=INK)
    ax.set_xlim(0, total)
    ax.set_xticks(range(0, total + 1, 2))
    ax.tick_params(axis="x", labelsize=9, colors=INK_SECONDARY, length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"domande su {total} in cui il dato richiesto non è raggiungibile",
                  fontsize=9.5, color=INK_SECONDARY, labelpad=10)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.text(0.012, 0.965, "Quando il dato non c'è, il modello lo dice?",
             fontsize=14.5, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.012, 0.912,
             "Condizione senza CSV: materiale, tipologia e funzione non sono caricati. "
             "La risposta corretta è dichiararlo.",
             fontsize=10, color=INK_SECONDARY, ha="left", va="top")
    ax.legend(
        handles=[Patch(facecolor=ABSTAINS, label="dichiara il dato non disponibile"),
                 Patch(facecolor=INVENTS, label="fornisce comunque un valore"),
                 Patch(facecolor=EMPTY, label="nessuna risposta prodotta")],
        loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=3, frameon=False,
        fontsize=9.5, labelcolor=INK_SECONDARY, handlelength=1.4, handleheight=1.0,
    )
    fig.text(
        0.012, 0.035,
        "Nessuna risposta prodotta non è un'allucinazione: è il ciclo dell'agente che si "
        "interrompe dopo la chiamata allo strumento.\nLe due cose sono tenute separate perché "
        "hanno cause diverse e richiedono interventi diversi.",
        fontsize=8.5, color=INK_MUTED, ha="left", va="bottom",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.2, 1, 0.84))
    for suffix in ("png", "svg"):
        path = output_dir / f"{args.name}.{suffix}"
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        print(f"written: {path}")


if __name__ == "__main__":
    main()
