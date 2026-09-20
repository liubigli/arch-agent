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


def score_report(path: Path, payload: dict) -> tuple[str, dict[int, bool]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"] if isinstance(data, dict) else data
    out: dict[int, bool] = {}
    for record in records:
        question_id = record.get("question_id")
        spec = reference_for_question(payload, question_id)
        tool_output = "\n".join((c.get("output") or "") for c in (record.get("tool_calls") or []))
        score = score_answer(record.get("final_answer"), spec, tool_output=tool_output)
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
    stem = re.sub(r"_think_(true|false)", "", stem)
    stem = re.sub(r"_\d{8}$", "", stem)
    return (stem.replace("gpt_oss_20b", "gpt-oss:20b").replace("gemma4_31b", "gemma4:31b")
                .replace("qwen3_5", "qwen3.5").replace("llama3_1", "llama3.1")
                .replace("command_r", "command-r"))


if __name__ == "__main__":
    main()
