"""Build a blind annotation sheet for validating the automatic scorer.

Writes two files. The sheet is what a person annotates; it deliberately does
not contain the scorer's verdict, because an annotator who can see it is
confirming the scorer rather than validating it. The verdicts go to a separate
file that compute_kappa.py reads afterwards.

Sampling is stratified by model only. Stratifying on the scorer's own verdict
would change the marginal frequencies, and Cohen's kappa is computed from those
marginals - a sample balanced between correct and incorrect would give a kappa
that does not describe the real distribution.

    python make_kappa_sheet.py <raw report json>... --per-model 12
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from arch_agent.benchmark.scoring import CORRECT, score_answer
from arch_agent.benchmark.structured_reference import (
    load_structured_reference,
    reference_for_question,
)

SHEET_FIELDS = [
    "item_id",
    "modello",
    "question_id",
    "famiglia",
    "domanda",
    "risposta_del_modello",
    "fatto_atteso",
    "verdetto_umano",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--reference-file", default="benchmark/references/scena4_VAL_reference_draft.json")
    parser.add_argument("--questions-file", default="benchmark/domande_per_scene.txt")
    parser.add_argument("--per-model", type=int, default=12, help="Items sampled per model.")
    parser.add_argument("--seed", type=int, default=20260920, help="Sampling seed, so the draw is reproducible.")
    parser.add_argument("--output-dir", default="benchmark/kappa")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_structured_reference(args.reference_file)
    questions = load_questions(Path(args.questions_file))
    rng = random.Random(args.seed)

    sheet_rows, hidden_rows = [], []
    for report in sorted(args.reports):
        path = Path(report)
        model, scored = score_report(path, payload, questions)
        if not scored:
            print(f"{path.name}: nothing scoreable, skipped")
            continue
        picked = rng.sample(scored, min(args.per_model, len(scored)))
        for item in picked:
            item_id = f"{model}__q{item['question_id']}"
            sheet_rows.append({
                "item_id": item_id,
                "modello": model,
                "question_id": item["question_id"],
                "famiglia": item["family"],
                "domanda": item["question"],
                "risposta_del_modello": item["answer"],
                "fatto_atteso": item["expected"],
                "verdetto_umano": "",
                "note": "",
            })
            hidden_rows.append({
                "item_id": item_id,
                "verdetto_scorer": item["outcome"],
                "dettaglio_scorer": item["detail"],
            })

    rng.shuffle(sheet_rows)  # models interleaved, so fatigue does not track one model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = output_dir / "annotation_sheet.csv"
    hidden_path = output_dir / "scorer_verdicts.csv"
    write_csv(sheet_path, sheet_rows, SHEET_FIELDS)
    write_csv(hidden_path, hidden_rows, ["item_id", "verdetto_scorer", "dettaglio_scorer"])

    print(f"sheet   : {sheet_path}  ({len(sheet_rows)} items to annotate)")
    print(f"verdicts: {hidden_path}  (do not open before annotating)")
    print()
    print("Fill 'verdetto_umano' with corretto / non_corretto / incerto.")
    print("See the instructions block written next to the sheet.")
    write_instructions(output_dir / "ISTRUZIONI.md", len(sheet_rows))


def score_report(path: Path, payload: dict, questions: list[str]):
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"] if isinstance(data, dict) else data
    model = model_name(path)
    out = []
    for record in records:
        question_id = record.get("question_id")
        spec = reference_for_question(payload, question_id)
        if spec is None:
            continue
        tool_output = "\n".join((c.get("output") or "") for c in (record.get("tool_calls") or []))
        score = score_answer(record.get("final_answer"), spec, tool_output=tool_output)
        if not score.is_scored:
            continue
        out.append({
            "question_id": question_id,
            "family": score.family,
            "question": questions[question_id - 1] if question_id <= len(questions) else record.get("question", ""),
            "answer": (record.get("final_answer") or "").strip(),
            "expected": describe_expectation(spec),
            "outcome": "corretto" if score.outcome == CORRECT else "non_corretto",
            "detail": score.detail,
        })
    return model, out


def describe_expectation(spec: dict) -> str:
    """What the reference says the answer must contain, as readable text."""
    facts = spec.get("resolved_facts") or spec.get("required_facts") or {}
    parts = []
    for key, value in facts.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        parts.append(f"{key} = {value}")
    forbidden = spec.get("forbidden_claims") or []
    if forbidden:
        parts.append("NON deve affermare: " + "; ".join(str(f) for f in forbidden))
    return " | ".join(parts)


def model_name(path: Path) -> str:
    stem = re.sub(r"^benchmark_raw_scena4_VAL_|_test_\d+$", "", path.stem)
    stem = re.sub(r"_think_(true|false)", "", stem)
    return re.sub(r"_\d{8}$", "", stem)


def load_questions(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip().endswith("?")]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_instructions(path: Path, count: int) -> None:
    path.write_text(INSTRUCTIONS.format(count=count), encoding="utf-8")
    print(f"istruzioni: {path}")


INSTRUCTIONS = """# Istruzioni per l'annotazione ({count} voci)

Stai validando lo **scorer automatico**, non i modelli. Il risultato di questo
lavoro e' un numero solo: quanto lo scorer e' d'accordo con una persona.

## La domanda a cui rispondi

Per ogni riga, una sola:

> **Data la scena, questa risposta e' corretta?**

Non stai giudicando se e' scritta bene, se ha usato il tool giusto, se e'
prolissa o se e' in italiano o inglese. Solo se dice il vero.

## Cosa scrivere in `verdetto_umano`

| Valore | Quando |
|---|---|
| `corretto` | la risposta afferma il fatto atteso, comunque formulato, e non afferma nulla che lo contraddica |
| `non_corretto` | afferma un valore sbagliato, inventa un oggetto / una classe / una relazione, oppure non afferma affatto il fatto atteso |
| `incerto` | solo se la risposta e' genuinamente ambigua e non riesci a decidere |

Usa `incerto` con parsimonia: quelle voci vengono **escluse** dal calcolo, e se
sono troppe il risultato perde significato. Se ne usi piu' di 5-6 su {count},
annota nel campo `note` il motivo: probabilmente e' il gold a essere ambiguo, e
quella e' un'informazione preziosa di suo.

## Regole che evitano i disaccordi inutili

1. **Parziale vale `non_corretto`.** Una risposta che azzecca meta' dei conteggi
   non e' corretta. Scrivi `parziale` nelle note: il conteggio dei parziali
   viene riportato a parte.
2. **Informazione in piu' non penalizza.** Se dice il fatto atteso e aggiunge
   altro di vero, e' `corretto`.
3. **La lingua non conta.** Italiano o inglese sono equivalenti.
4. **La formulazione non conta.** "Ci sono 6 colonne", "column: 6" e
   "$\\boxed{{6}}$" sono la stessa affermazione.
5. **Dire che un dato non c'e', quando davvero non c'e', e' `corretto`.**
   Astenersi non e' sbagliare.
6. **Inventare e' sempre `non_corretto`**, anche se il resto della risposta e'
   giusto: un oggetto come `arch_0`, in una scena senza archi, invalida la
   risposta.

## Come lavorare

- Annota **senza** aprire `scorer_verdicts.csv`. Se vedi prima il verdetto
  automatico, stai confermando lo scorer, non validandolo, e il numero che ne
  esce non vale nulla.
- Le righe sono mescolate tra i modelli apposta: non sai quale modello ha
  risposto, e la stanchezza non si concentra su uno solo.
- Se una riga richiede piu' di un minuto, e' probabilmente un `incerto`.

Quando hai finito: `python compute_kappa.py`
"""


if __name__ == "__main__":
    main()
