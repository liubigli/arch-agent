"""Compare the human annotation with the automatic scorer.

Reports Cohen's kappa with a bootstrap confidence interval, the confusion
matrix, and every disagreement - the disagreements are the useful part, because
each one is either a scorer bug or an ambiguous reference entry.

Raw agreement is printed too, but only next to the chance agreement it has to
beat: with an accuracy near 80%, two raters who agree 85% of the time are
barely above what coin flips would produce.

    python compute_kappa.py [--dir benchmark/kappa]
"""

from __future__ import annotations

import argparse
import csv
import io
import random
from collections import Counter
from pathlib import Path

VALID = {"corretto", "non_corretto"}
SKIP = {"incerto", ""}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="benchmark/kappa")
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--verdicts", default=None)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--show-disagreements", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.dir)
    sheet = read_csv(Path(args.sheet) if args.sheet else base / "annotation_sheet.csv")
    verdicts = {r["item_id"]: r for r in read_csv(Path(args.verdicts) if args.verdicts else base / "scorer_verdicts.csv")}

    pairs, skipped, missing = [], [], []
    for row in sheet:
        human = (row.get("verdetto_umano") or "").strip().lower()
        item = verdicts.get(row["item_id"])
        if item is None:
            missing.append(row["item_id"])
            continue
        if human in SKIP:
            skipped.append(row)
            continue
        if human not in VALID:
            raise SystemExit(
                f"{row['item_id']}: verdetto_umano = {human!r}. "
                f"Ammessi: {', '.join(sorted(VALID))}, incerto, oppure vuoto."
            )
        pairs.append((row, human, item["verdetto_scorer"]))

    if missing:
        print(f"WARNING: {len(missing)} item senza verdetto dello scorer: {missing[:5]}")
    annotated = len(pairs)
    if annotated < 2:
        raise SystemExit("Servono almeno due voci annotate.")

    human_labels = [h for _, h, _ in pairs]
    scorer_labels = [s for _, _, s in pairs]

    kappa = cohen_kappa(human_labels, scorer_labels)
    low, high = bootstrap_ci(human_labels, scorer_labels, args.bootstrap, args.seed)
    observed = sum(h == s for h, s in zip(human_labels, scorer_labels)) / annotated
    expected = chance_agreement(human_labels, scorer_labels)

    print(f"Annotate      : {annotated}   (escluse: {len(skipped)} incerte / vuote)")
    print(f"Accordo grezzo: {observed:.1%}")
    print(f"Atteso a caso : {expected:.1%}   <- questa e' la soglia da battere")
    print(f"Cohen's kappa : {kappa:.3f}   IC 95% bootstrap [{low:.3f}, {high:.3f}]")
    print(f"                {interpret(kappa)}")
    print()

    print("Matrice di confusione (righe = umano, colonne = scorer)")
    labels = ["corretto", "non_corretto"]
    counts = Counter(zip(human_labels, scorer_labels))
    print(f"{'':16}" + "".join(f"{c:>16}" for c in labels))
    for row_label in labels:
        cells = "".join(f"{counts[(row_label, c)]:>16}" for c in labels)
        print(f"{row_label:16}{cells}")
    print()

    disagreements = [(row, h, s) for row, h, s in pairs if h != s]
    print(f"Disaccordi: {len(disagreements)}")
    by_family = Counter(row["famiglia"] for row, _, _ in disagreements)
    if by_family:
        print("  per famiglia:", dict(by_family.most_common()))
    print()
    for row, human, scorer in disagreements[: args.show_disagreements]:
        print(f"  {row['item_id']}  umano={human}  scorer={scorer}  [{row['famiglia']}]")
        print(f"    domanda : {row['domanda'][:96]}")
        print(f"    risposta: {row['risposta_del_modello'][:96]}")
        print(f"    atteso  : {row['fatto_atteso'][:96]}")
        print(f"    nota    : {row.get('note', '')[:96]}")
        print()

    print("I disaccordi in cui l'umano dice corretto e lo scorer no sono bug dello")
    print("scorer. Quelli opposti sono di norma errori del gold o del modello che")
    print("lo scorer ha accettato per un controllo troppo permissivo.")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = chance_agreement(a, b)
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def chance_agreement(a: list[str], b: list[str]) -> float:
    n = len(a)
    count_a, count_b = Counter(a), Counter(b)
    return sum((count_a[label] / n) * (count_b[label] / n) for label in set(a) | set(b))


def bootstrap_ci(a: list[str], b: list[str], draws: int, seed: int) -> tuple[float, float]:
    """Percentile interval by resampling items, so no distributional assumption."""
    rng = random.Random(seed)
    n = len(a)
    values = []
    for _ in range(draws):
        index = [rng.randrange(n) for _ in range(n)]
        sample_a = [a[i] for i in index]
        sample_b = [b[i] for i in index]
        if len(set(sample_a)) < 2 or len(set(sample_b)) < 2:
            continue
        values.append(cohen_kappa(sample_a, sample_b))
    if not values:
        return (float("nan"), float("nan"))
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def interpret(kappa: float) -> str:
    if kappa < 0.20:
        return "scarso - i numeri automatici non sono presentabili come accuratezza"
    if kappa < 0.41:
        return "discreto - non presentabile senza revisione"
    if kappa < 0.61:
        return "moderato - presentabile solo con la riserva dichiarata"
    if kappa < 0.81:
        return "sostanziale - presentabile dichiarando il kappa"
    return "quasi perfetto - i numeri automatici reggono da soli"


def read_csv(path: Path) -> list[dict]:
    """Read a sheet back, whatever the annotator's spreadsheet wrote.

    Excel in an Italian locale saves CSV with ";" and a BOM, so the file that
    comes back is not the file that went out. Sniffing the delimiter costs
    nothing and avoids a confusing failure on the last step of the process.
    """
    if not path.exists():
        raise SystemExit(f"File mancante: {path}")
    text = path.read_text(encoding="utf-8-sig")
    header = text.splitlines()[0] if text else ""
    delimiter = ";" if header.count(";") > header.count(",") else ","
    # StringIO, not splitlines: the answers contain newlines inside quoted
    # fields, and splitting on lines first would tear them apart.
    return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))


if __name__ == "__main__":
    main()
