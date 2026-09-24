"""What the CSV layer is worth: the full condition against the graph-only one.

Same models, same questions, two conditions - so the comparison is paired twice
over and McNemar applies per model.

The subtlety is which questions may be compared at all. Eleven ask for
material, typology or function, and their correct answer is not the same in the
two conditions: with the CSV it is the value, without it is saying the value
cannot be reached. Scoring both and subtracting would compare two different
tasks and report the difference as a change in ability. Those questions are
therefore excluded from the paired test and reported separately, on their own
terms - what the model does when the data is withheld.

    python compare_conditions.py --full <dir> --graph <dir> [--figure docs/figures]
"""

from __future__ import annotations

import argparse
import json
import re
from math import comb
from pathlib import Path

from arch_agent.benchmark.scoring import CORRECT, score_answer
from arch_agent.benchmark.structured_reference import (
    load_structured_reference,
    reference_for_question,
)

WITHHELD = ("abstention", "abstention_or_role_only")

ACCENT_FULL = "#1c5cab"   # sequential blue, step 550 - the condition with more information
ACCENT_GRAPH = "#86b6ef"  # step 250 - the same hue, lighter: one measure, two states
RULE = "#c3c2b7"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, help="Directory of the full-condition raw reports.")
    parser.add_argument("--graph", required=True, help="Directory of the graph-condition raw reports.")
    parser.add_argument("--reference-file", default="benchmark/references/scena4_VAL_reference_draft.json")
    parser.add_argument("--figure", default=None, help="Directory to write the figure into.")
    parser.add_argument("--name", default="benchmark_conditions_dumbbell")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_structured_reference(args.reference_file)
    withheld_ids = {
        entry["question_id"] for entry in payload["references"]
        if entry.get("expected_without_csv") in WITHHELD
    }

    full = {name: data for name, data in load_dir(Path(args.full), payload).items()}
    graph = {name: data for name, data in load_dir(Path(args.graph), payload).items()}
    models = sorted(set(full) & set(graph))
    if not models:
        raise SystemExit("Nessun modello in comune fra le due condizioni.")

    rows = []
    for model in models:
        b, g = full[model], graph[model]
        shared = sorted((set(b["scored"]) & set(g["scored"])) - withheld_ids)
        only_full = sum(1 for q in shared if b["scored"][q] and not g["scored"][q])
        only_graph = sum(1 for q in shared if not b["scored"][q] and g["scored"][q])
        rows.append({
            "model": model,
            "n": len(shared),
            "acc_full": mean([b["scored"][q] for q in shared]),
            "acc_graph": mean([g["scored"][q] for q in shared]),
            "only_full": only_full,
            "only_graph": only_graph,
            "p": mcnemar_exact(only_full, only_graph),
            "empty_full": b["empty"],
            "empty_graph": g["empty"],
        })
    rows.sort(key=lambda r: -r["acc_full"])

    n_shared = rows[0]["n"]
    print(f"Domande confrontabili: {n_shared}")
    print(f"Escluse dal test appaiato: {len(withheld_ids)} che dipendono dal CSV "
          f"(risposta attesa diversa nelle due condizioni)\n")
    print(f"  {'modello':14} {'con CSV':>8} {'senza':>8} {'delta':>7} "
          f"{'b':>3} {'c':>3} {'p':>8}  esito")
    for r in rows:
        delta = r["acc_graph"] - r["acc_full"]
        verdict = "peggiora" if r["p"] < args.alpha and delta < 0 else (
            "migliora" if r["p"] < args.alpha else "non distinguibile")
        print(f"  {r['model']:14} {r['acc_full']:7.1%} {r['acc_graph']:7.1%} "
              f"{delta:+6.1%} {r['only_full']:3} {r['only_graph']:3} {r['p']:8.4f}  {verdict}")

    print(f"\nRisposte vuote, tutte le 60 domande:")
    for r in rows:
        print(f"  {r['model']:14} con CSV {r['empty_full']:2}  ->  senza CSV {r['empty_graph']:2}")

    if args.figure:
        draw(rows, n_shared, len(withheld_ids), args)


def load_dir(directory: Path, payload: dict) -> dict:
    out: dict[str, dict] = {}
    for path in sorted(directory.glob("benchmark_raw_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data["records"] if isinstance(data, dict) else data
        condition = data.get("condition", "full") if isinstance(data, dict) else "full"
        scored: dict[int, bool] = {}
        empty = 0
        for record in records:
            answer = record.get("final_answer") or ""
            if not answer.strip():
                empty += 1
            spec = reference_for_question(payload, record.get("question_id"))
            score = score_answer(answer, spec,
                                 tool_output="\n".join((c.get("output") or "")
                                                       for c in (record.get("tool_calls") or [])),
                                 condition=condition)
            if score.is_scored:
                scored[record["question_id"]] = score.outcome == CORRECT
        out[model_name(path)] = {"scored": scored, "empty": empty}
    return out


def model_name(path: Path) -> str:
    stem = re.sub(r"^benchmark_raw_scena4_VAL_|_test_\d+$", "", path.stem)
    stem = re.sub(r"_think_(true|false)|_cond_(graph|full|none)|_\d{8}$", "", stem)
    return (stem.replace("gpt_oss_20b", "gpt-oss:20b").replace("gemma4_31b", "gemma4:31b")
                .replace("qwen3_5", "qwen3.5").replace("llama3_1", "llama3.1")
                .replace("command_r", "command-r"))


def mean(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def draw(rows, n_shared: int, n_withheld: int, args) -> None:
    """Before/after per model: a dumbbell, one hue in two states."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10.2, 0.6 * len(rows) + 2.3))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for index, row in enumerate(rows):
        a, g = row["acc_full"], row["acc_graph"]
        ax.plot([g, a], [index, index], color=RULE, linewidth=2, zorder=1, solid_capstyle="round")
        ax.scatter([g], [index], s=150, color=ACCENT_GRAPH, zorder=2,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.scatter([a], [index], s=150, color=ACCENT_FULL, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        delta = g - a
        mark = "*" if row["p"] < args.alpha else ""
        ax.text(max(a, g) + 0.028, index, f"{delta:+.0%}{mark}", va="center", fontsize=10,
                color=INK if row["p"] < args.alpha else INK_MUTED,
                fontweight="bold" if row["p"] < args.alpha else "normal")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["model"] for r in rows], fontsize=11.5, color=INK)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="x", labelsize=9, colors=INK_SECONDARY, length=0)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"accuratezza sulle {n_shared} domande la cui risposta attesa non cambia",
                  fontsize=9.5, color=INK_SECONDARY, labelpad=10)

    fig.text(0.012, 0.965, "Quanto vale il layer CSV", fontsize=14.5,
             color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.012, 0.905,
             "Stessi modelli, stesse domande, con e senza le annotazioni dell'architetto",
             fontsize=10, color=INK_SECONDARY, ha="left", va="top")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", markersize=10,
                   markerfacecolor=ACCENT_FULL, markeredgecolor=SURFACE, label="con CSV"),
            Line2D([], [], marker="o", linestyle="", markersize=10,
                   markerfacecolor=ACCENT_GRAPH, markeredgecolor=SURFACE, label="senza CSV"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False,
        fontsize=9.5, labelcolor=INK_SECONDARY,
    )
    fig.text(
        0.012, 0.035,
        f"Le {n_withheld} domande che dipendono dal CSV sono escluse: senza CSV la risposta "
        "corretta non è il valore ma dichiararne l'assenza,\ne confrontarle misurerebbe due "
        "compiti diversi. L'asterisco marca le differenze che superano il test di McNemar esatto.",
        fontsize=8.5, color=INK_MUTED, ha="left", va="bottom",
    )

    output_dir = Path(args.figure)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.23, 1, 0.83))
    for suffix in ("png", "svg"):
        path = output_dir / f"{args.name}.{suffix}"
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        print(f"written: {path}")


if __name__ == "__main__":
    main()
