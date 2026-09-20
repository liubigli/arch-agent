"""Regression cases for the answer scorer.

Run with `python tests/test_scoring.py` - no test framework needed.

Every case is an answer a model could plausibly give, paired with the outcome
the scorer must produce. Cases marked SCORER were real bugs found by running
the scorer on actual benchmark output; they are kept so the fix cannot regress.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arch_agent.benchmark.scoring import (  # noqa: E402
    CORRECT,
    INCORRECT,
    NOT_SCORED,
    PARTIAL,
    aggregate,
    score_answer,
)
from arch_agent.benchmark.structured_reference import (  # noqa: E402
    load_structured_reference,
    reference_for_question,
)

REFERENCE = "benchmark/references/scena4_VAL_reference_draft.json"

# (question id, expected outcome, answer, note)
CASES = [
    (2, CORRECT, "La scena contiene 33 oggetti in totale.", ""),
    (2, INCORRECT, "La scena contiene 34 oggetti in totale.", ""),
    (2, CORRECT, r"The final answer is: $\boxed{33}$", "SCORER: bare number, no total marker"),
    (3, CORRECT, "Le classi presenti sono: column, door_window, floor, moldings, vault, wall.", ""),
    (3, INCORRECT, "Le classi presenti sono: column, wall, roof e stairs.", ""),
    (4, CORRECT, "Le classi assenti sono arch, other, roof e stairs: non sono presenti nella scena.",
     "SCORER: reference marker list has 'assente' but not 'assenti'"),
    (6, PARTIAL, "12 walls, 8 door_windows, 6 columns, 6 moldings, 1 vault and 1 floor, total 34 objects.",
     "SCORER: plural alias 'door_windows' must not be shadowed by 'door_window'"),
    (7, CORRECT,
     "column: 6, wall: 11, floor: 1, roof: 0, vault: 1, arch: 0, stairs: 0, door_window: 8, moldings: 6", ""),
    (7, PARTIAL,
     "column: 6, wall: 9, floor: 1, roof: 0, vault: 1, arch: 0, stairs: 0, door_window: 8, moldings: 6", ""),
    (14, CORRECT, "Nella scena ci sono 6 colonne.", ""),
    (14, INCORRECT, "Nella scena ci sono 8 colonne.", ""),
    (15, CORRECT, "Gli oggetti column sono column_0, column_1, column_2, column_3, column_4, column_5.", ""),
    (15, INCORRECT, "Gli oggetti column sono column_0, column_1 e column_2.", ""),
    (16, CORRECT, "Le colonne sono in Breccia policroma, secondo il CSV di scena.", ""),
    (16, INCORRECT, "Le colonne sono in marmo bianco.", "the hallucination seen on llama3.1"),
    (18, CORRECT, "Si, le colonne sono elementi strutturali.", ""),
    (19, CORRECT, "Le colonne supportano la volta: column supports vault, 6 istanze.", ""),
    (19, INCORRECT, "Le colonne supportano il pavimento.", ""),
    (39, CORRECT, "Non ci sono tetti nella scena: roof 0 oggetti.", ""),
    (39, INCORRECT, "Nella scena ci sono 2 tetti.", ""),
    (40, CORRECT, 'Sembra che non ci siano oggetti della classe "roof" nella scena.',
     "SCORER: subjunctive 'non ci siano' is not in the marker list"),
    (41, CORRECT, "No, le colonne non supportano il tetto: la classe roof e assente.", ""),
    (41, INCORRECT, "Si, le colonne supportano il tetto.", ""),
    (44, INCORRECT, 'Si. La scena contiene un "roof" (copertura).', ""),
    (49, CORRECT, "No: la classe arch e assente, quindi non ci sono archi che formano nervature.", ""),
    (49, INCORRECT, "Si, arch_0 e una nervatura della volta.", "must flag the invented object"),
    (50, INCORRECT, "Si. Ci sono 2 scale nella scena.", ""),
    (56, CORRECT, "Si, ci sono 2 door_window in legno: door_window_3 e door_window_7.", ""),
    (56, PARTIAL, "Si, ce ne sono 4: door_window_0, door_window_1, door_window_3, door_window_7.", ""),
    (57, CORRECT,
     "door_window_3 si trova a 635.53, 811.65, 230.11 e door_window_7 a 630.87, 801.82, 230.16.", ""),
    (60, CORRECT, "I moldings sono elementi ornamentali, non strutturali.",
     "SCORER: a bare 'non' before a role word is not in the marker list"),
    (1, NOT_SCORED, "La scena e una sala con colonne e volte.", "semantic, needs judgement"),
]

EXPECTED_COVERAGE = 49  # 50 once tool output is supplied: Q26 is scored against it


def main() -> int:
    payload = load_structured_reference(REFERENCE)
    failures = []

    for question_id, expected, answer, note in CASES:
        spec = reference_for_question(payload, question_id)
        result = score_answer(answer, spec)
        if result.outcome != expected:
            failures.append(
                f"  Q{question_id} expected {expected}, got {result.outcome}"
                f"{f' [{note}]' if note else ''}\n      {result.detail[:100]}"
            )

    scores = [
        score_answer("risposta generica", reference_for_question(payload, entry["question_id"]))
        for entry in payload["references"]
    ]
    coverage = aggregate(scores)["scored"]
    if coverage != EXPECTED_COVERAGE:
        failures.append(f"  coverage is {coverage}, expected {EXPECTED_COVERAGE}")

    # Q26 is scored entirely against the tool output, so it needs one to count.
    spec26 = reference_for_question(payload, 26)
    tool_text = "wall_0 adjacent_to wall_1; wall_0 supports vault_0"
    grounded = score_answer("wall_0 e adiacente a wall_1 e supporta vault_0.", spec26, tool_output=tool_text)
    invented = score_answer("wall_0 supporta il tetto roof_0.", spec26, tool_output=tool_text)
    if grounded.outcome != CORRECT:
        failures.append(f"  Q26 grounded answer scored {grounded.outcome}: {grounded.detail[:80]}")
    if invented.outcome == CORRECT:
        failures.append("  Q26 accepted a relation absent from the tool output")

    if failures:
        print(f"FAILED {len(failures)} of {len(CASES) + 3} checks\n" + "\n".join(failures))
        return 1
    print(f"OK: {len(CASES)} answer cases, coverage {coverage}/60, tool-grounding both ways")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
