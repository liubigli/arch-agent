"""Render the benchmark scores as a model x question-family heatmap.

The grid compares magnitude, so the encoding is sequential: one hue, light to
dark, from the validated blue ramp. Column headers carry the number of
questions behind each cell, because a family of one question would otherwise
look as authoritative as a family of eighteen.

    python plot_scores.py <raw report json>... -o docs/figures/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from arch_agent.benchmark.scoring import aggregate, score_answer  # noqa: E402
from arch_agent.benchmark.structured_reference import (  # noqa: E402
    load_structured_reference,
    reference_for_question,
)

# Sequential blue ramp, steps 100 -> 700. Light means near zero and is allowed
# to recede toward the surface.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

FAMILY_LABEL = {
    "exact_value": "Valore\nesatto",
    "absence": "Assenza",
    "set": "Insiemi",
    "tool_grounded": "Coerenza\ncon i tool",
    "counts": "Conteggi",
    "per_object": "Per\noggetto",
    "relationship": "Relazioni",
    "coordinates": "Coordinate",
    "withheld_csv": "Astensione\n(senza CSV)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--reference-file", default="benchmark/references/scena4_VAL_reference_draft.json")
    parser.add_argument("-o", "--output-dir", default="docs/figures")
    parser.add_argument("--name", default="benchmark_accuracy_heatmap",
                        help="Basename for the written files.")
    parser.add_argument("--title", default="Accuratezza per famiglia di domande")
    parser.add_argument("--subtitle", default="scena4_VAL, 60 domande, run del 17 settembre 2026")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_structured_reference(args.reference_file)

    rows = []
    for path in args.reports:
        name, summary = score_report(Path(path), payload)
        rows.append((name, summary))
    rows.sort(key=lambda item: item[1]["accuracy"], reverse=True)

    families = order_families(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    draw(rows, families, args, output_dir)


def score_report(path: Path, payload: dict) -> tuple[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"] if isinstance(data, dict) else data
    condition = data.get("condition", "full") if isinstance(data, dict) else "full"
    scores = []
    for record in records:
        spec = reference_for_question(payload, record.get("question_id"))
        tool_output = "\n".join(
            (call.get("output") or "") for call in (record.get("tool_calls") or [])
        )
        scores.append(score_answer(record.get("final_answer"), spec,
                                   tool_output=tool_output, condition=condition))
    return model_name(path), aggregate(scores)


def model_name(path: Path) -> str:
    stem = re.sub(r"^benchmark_raw_scena4_VAL_|_test_\d+$", "", path.stem)
    stem = re.sub(r"_think_(true|false)|_cond_(graph|full|none)", "", stem)
    stem = re.sub(r"_\d{8}$", "", stem)
    return stem.replace("gpt_oss_20b", "gpt-oss:20b").replace("gemma4_31b", "gemma4:31b") \
               .replace("qwen3_5", "qwen3.5").replace("llama3_1", "llama3.1") \
               .replace("command_r", "command-r").replace("deepseek_r1_14b", "deepseek-r1:14b")


def order_families(rows) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for _, summary in rows:
        for family, stats in summary["by_family"].items():
            counts[family] = max(counts.get(family, 0), stats["n"])
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)


def draw(rows, families, args, output_dir: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("seq_blue", RAMP)
    columns = [("__overall__", 50)] + families
    grid = [
        [
            summary["accuracy"] if key == "__overall__"
            else summary["by_family"].get(key, {}).get("accuracy")
            for key, _ in columns
        ]
        for _, summary in rows
    ]

    # Two gaps carry meaning: the total is an aggregate, not a family, and the
    # families past the third rest on four questions or fewer.
    SMALL_SAMPLE_AT = next(
        (i for i, (_, n) in enumerate(columns) if i and n < 4), len(columns)
    )
    x_of = []
    offset = 0.0
    for index in range(len(columns)):
        if index == 1:
            offset += 0.34
        if index == SMALL_SAMPLE_AT:
            offset += 0.34
        x_of.append(index + offset)
    width = x_of[-1] + 1

    fig_w = 1.05 * width + 2.6
    fig_h = 0.62 * len(rows) + 2.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value is None:
                continue
            x = x_of[c]
            # 2px surface gap between fills
            ax.add_patch(plt.Rectangle((x + 0.03, r + 0.03), 0.94, 0.94,
                                       facecolor=cmap(value), edgecolor=SURFACE, linewidth=1.5))
            ax.text(x + 0.5, r + 0.5, f"{value * 100:.0f}",
                    ha="center", va="center", fontsize=11,
                    color="#ffffff" if value > 0.55 else INK,
                    fontweight="bold" if c == 0 else "normal")

    if SMALL_SAMPLE_AT < len(columns):
        edge = x_of[SMALL_SAMPLE_AT] - 0.17
        ax.plot([edge, edge], [0, len(rows)], color="#e1e0d9", linewidth=1.2, zorder=0)
        ax.text(edge + 0.1, -0.62, "campione ridotto", fontsize=8.5,
                color=INK_MUTED, ha="left", va="center")

    ax.set_xlim(0, width)
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([x + 0.5 for x in x_of])
    ax.set_xticklabels(
        ["Totale\n(n=50)"] + [f"{FAMILY_LABEL.get(k, k)}\n(n={n})" for k, n in families],
        fontsize=9, color=INK_SECONDARY,
    )
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([r + 0.5 for r in range(len(rows))])
    ax.set_yticklabels([name for name, _ in rows], fontsize=11, color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.text(0.012, 0.965, args.title, fontsize=14, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.012, 0.915, args.subtitle, fontsize=10, color=INK_SECONDARY, ha="left", va="top")
    fig.text(
        0.012, 0.045,
        "Percentuale di risposte pienamente corrette. Le 10 domande interpretative "
        "restano escluse: richiedono giudizio umano.\nLe famiglie con n basso sono "
        "indicative, non conclusive.",
        fontsize=8.5, color=INK_MUTED, ha="left", va="bottom",
    )

    bar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.025, pad=0.02)
    bar.set_ticks([0, 0.5, 1])
    bar.set_ticklabels(["0%", "50%", "100%"])
    bar.ax.tick_params(labelsize=8, length=0, colors=INK_SECONDARY)
    bar.outline.set_visible(False)

    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    for suffix in ("png", "svg"):
        path = output_dir / f"{args.name}.{suffix}"
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        print(f"written: {path}")

    write_table(rows, columns, output_dir, args)


def write_table(rows, columns, output_dir: Path, args) -> None:
    """The table view the accessibility pass requires."""
    header = ["modello", "totale"] + [k for k, _ in columns[1:]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, summary in rows:
        cells = [name, f"{summary['accuracy'] * 100:.0f}%"]
        for key, _ in columns[1:]:
            stats = summary["by_family"].get(key)
            cells.append("-" if stats is None else f"{stats['accuracy'] * 100:.0f}%")
        lines.append("| " + " | ".join(cells) + " |")
    path = output_dir / f"{args.name}_table.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
