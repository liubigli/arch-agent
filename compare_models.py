"""Compare two models on the same questions with McNemar's exact test.

Every model answers the same 60 questions, so the data are paired. Comparing
two accuracies as if they came from independent samples throws that pairing
away and needs a much larger sample to show anything: with 50 items, a
confidence interval on a single proportion near 80% is roughly +-11 points, so
almost no difference between two models would clear it.

McNemar looks only at the questions where the two models disagree. Agreement
carries no information about which is better - if both get a question right,
it separates nothing - so the test conditions on the discordant pairs and asks
whether they fall one way more often than a coin would explain.

    python compare_models.py <raw report json>...
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from math import comb
from pathlib import Path

from arch_agent.benchmark.scoring import CORRECT, score_answer
from arch_agent.benchmark.structured_reference import (
    load_structured_reference,
    reference_for_question,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--reference-file", default="benchmark/references/scena4_VAL_reference_draft.json")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--figure", default=None,
                        help="Directory to write the pairwise matrix figure into.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_structured_reference(args.reference_file)

    results = {}
    for report in sorted(args.reports):
        name, per_question = score_report(Path(report), payload)
        results[name] = per_question

    names = sorted(results, key=lambda n: -accuracy(results[n]))
    print("Accuratezza sulle domande valutabili\n")
    for name in names:
        outcomes = results[name]
        print(f"  {name:16} {accuracy(outcomes):6.1%}   ({sum(outcomes.values())}/{len(outcomes)})")

    pairs = list(combinations(names, 2))
    print(f"\nTest di McNemar esatto, {len(pairs)} confronti a coppie")
    print("b = solo il primo corretto, c = solo il secondo corretto\n")
    print(f"  {'coppia':38} {'b':>3} {'c':>3} {'p':>9}  esito")

    significant = []
    for first, second in pairs:
        shared = sorted(set(results[first]) & set(results[second]))
        b = sum(1 for q in shared if results[first][q] and not results[second][q])
        c = sum(1 for q in shared if not results[first][q] and results[second][q])
        p = mcnemar_exact(b, c)
        verdict = "differenza reale" if p < args.alpha else "non distinguibili"
        if p < args.alpha:
            significant.append((first, second, p))
        print(f"  {first + ' vs ' + second:38} {b:>3} {c:>3} {p:>9.4f}  {verdict}")

    print(f"\nSoglia alpha = {args.alpha}. Con {len(pairs)} confronti la correzione di")
    print(f"Bonferroni porta la soglia a {args.alpha / len(pairs):.4f}:")
    for first, second, p in significant:
        holds = "regge" if p < args.alpha / len(pairs) else "NON regge alla correzione"
        print(f"  {first} vs {second}: p={p:.4f} {holds}")
    if not significant:
        print("  nessuna differenza significativa da correggere")

    if args.figure:
        draw_matrix(names, results, args)


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

ACCENT = "#2a78d6"      # the difference is real
NEUTRAL = "#e1e0d9"     # not distinguishable
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def draw_matrix(names, results, args) -> None:
    """Lower triangle, models ordered by accuracy.

    Two categories, not a magnitude: a p-value ramp would invite reading a
    small p as a large difference, which it is not. Emphasis encoding instead -
    the accent marks the comparisons that resolve, everything else recedes -
    and the p-value is printed in every cell so identity never rests on colour.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(names)
    pair_count = int(n * (n - 1) / 2)
    corrected = args.alpha / pair_count
    # Counted, not asserted: the caption used to claim every marked difference
    # survived the correction, which stopped being true when the scores moved.
    resolved = survived = 0
    scored_n = 0
    for row in range(1, n):
        for col in range(row):
            first, second = names[col], names[row]
            shared = sorted(set(results[first]) & set(results[second]))
            scored_n = max(scored_n, len(shared))
            b = sum(1 for q in shared if results[first][q] and not results[second][q])
            c = sum(1 for q in shared if not results[first][q] and results[second][q])
            p_value = mcnemar_exact(b, c)
            resolved += p_value < args.alpha
            survived += p_value < corrected
    fig, ax = plt.subplots(figsize=(1.45 * n + 1.6, 0.82 * n + 2.3))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for row in range(1, n):
        for col in range(row):
            first, second = names[col], names[row]
            shared = sorted(set(results[first]) & set(results[second]))
            b = sum(1 for q in shared if results[first][q] and not results[second][q])
            c = sum(1 for q in shared if not results[first][q] and results[second][q])
            p = mcnemar_exact(b, c)
            real = p < args.alpha
            ax.add_patch(plt.Rectangle((col + 0.03, row + 0.03), 0.94, 0.94,
                                       facecolor=ACCENT if real else NEUTRAL,
                                       edgecolor=SURFACE, linewidth=1.5))
            label = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            ax.text(col + 0.5, row + 0.60, label, ha="center", va="center",
                    fontsize=10.5, color="#ffffff" if real else INK,
                    fontweight="bold" if real else "normal")
            mark = "differenza reale" if real else "non distinguibili"
            ax.text(col + 0.5, row + 0.34, mark, ha="center", va="center",
                    fontsize=8, color="#dbe8fa" if real else INK_SECONDARY)

    for index, name in enumerate(names):
        accuracy_pct = accuracy(results[name])
        ax.text(index + 0.5, index + 0.5, f"{name}\n{accuracy_pct:.0%}",
                ha="center", va="center", fontsize=10.5, color=INK, fontweight="bold",
                linespacing=1.45)

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(0.012, 0.965, "Quali differenze tra modelli sono reali",
             fontsize=14, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.012, 0.915,
             f"Test di McNemar esatto sulle {scored_n} domande valutabili, "
             "stesse domande per ogni modello",
             fontsize=10, color=INK_SECONDARY, ha="left", va="top")
    # The empty half of the triangle is the natural home for the legend.
    ax.legend(
        handles=[Patch(facecolor=ACCENT, label="differenza reale"),
                 Patch(facecolor=NEUTRAL, label="non distinguibili")],
        loc="upper right", bbox_to_anchor=(0.995, 0.86), frameon=False,
        fontsize=9.5, labelcolor=INK_SECONDARY, handlelength=1.4, handleheight=1.0,
    )
    holds = ("tutte reggono" if survived == resolved
             else f"{survived} su {resolved} reggono")
    fig.text(
        0.012, 0.035,
        f"I modelli sulla diagonale sono ordinati per accuratezza. Soglia alpha = {args.alpha}; "
        f"con {pair_count} confronti la correzione di Bonferroni la porta a {corrected:.4f}, "
        f"e delle differenze marcate\ncome reali {holds} anche a quella soglia. "
        "Dentro ciascun gruppo grigio la classifica non e' sostenuta dai dati.",
        fontsize=8.5, color=INK_MUTED, ha="left", va="bottom",
    )

    output_dir = Path(args.figure)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    for suffix in ("png", "svg"):
        path = output_dir / f"benchmark_mcnemar_matrix.{suffix}"
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        print(f"\nwritten: {path}")


def score_report(path: Path, payload: dict) -> tuple[str, dict[int, bool]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"] if isinstance(data, dict) else data
    condition = data.get("condition", "full") if isinstance(data, dict) else "full"
    out: dict[int, bool] = {}
    for record in records:
        question_id = record.get("question_id")
        spec = reference_for_question(payload, question_id)
        tool_output = "\n".join((c.get("output") or "") for c in (record.get("tool_calls") or []))
        score = score_answer(record.get("final_answer"), spec,
                             tool_output=tool_output, condition=condition)
        if score.is_scored:
            out[question_id] = score.outcome == CORRECT
    return model_name(path), out


def accuracy(outcomes: dict[int, bool]) -> float:
    return sum(outcomes.values()) / len(outcomes) if outcomes else 0.0


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact test: P(as lopsided as observed) under a fair coin.

    Exact rather than the chi-square approximation because the discordant
    counts here are small, and the approximation is unreliable below about 25.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def model_name(path: Path) -> str:
    stem = re.sub(r"^benchmark_raw_scena4_VAL_|_test_\d+$", "", path.stem)
    stem = re.sub(r"_think_(true|false)|_cond_(graph|full|none)", "", stem)
    stem = re.sub(r"_\d{8}$", "", stem)
    return (stem.replace("gpt_oss_20b", "gpt-oss:20b").replace("gemma4_31b", "gemma4:31b")
                .replace("qwen3_5", "qwen3.5").replace("llama3_1", "llama3.1")
                .replace("command_r", "command-r"))


if __name__ == "__main__":
    main()
