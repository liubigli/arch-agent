"""Score one benchmark answer against the approved structured reference.

This is separate from `grounding_checks`, which flags issues worth a human
look. Scoring answers a narrower question - is this answer right? - for the
50 of 60 questions whose reference is checkable without judgement, and says
`not_scored` for the rest rather than guessing.

Every scorer works on `claims.Claims`, never on raw prose, so a model's
writing style cannot change its score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import claims as C

CORRECT = "correct"
INCORRECT = "incorrect"
PARTIAL = "partial"
NOT_SCORED = "not_scored"

FAMILY_BY_MODE = {
    "exact": "exact_value",
    "exact_absence": "absence",
    "set_exact": "set",
    "set_and_value_exact": "set",
    "set_and_counts_exact": "set",
    "counts_exact": "counts",
    "per_object_exact": "per_object",
    "relationship_exact": "relationship",
    "coordinates": "coordinates",
    "tool_output_grounded": "tool_grounded",
    "semantic": "open",
    "manual_assisted": "open",
}

# Fact keys that carry no expectation of their own.
_META_KEYS = frozenset({"reason", "values_must_match_tool_output", "language_note",
                        "additional_relations_must_exist_in_tool_output",
                        "every_reported_relation_must_exist_in_tool_output"})


@dataclass
class QuestionScore:
    question_id: int | None
    validation_mode: str
    family: str
    outcome: str
    score: float = 0.0
    detail: str = ""
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def is_scored(self) -> bool:
        return self.outcome != NOT_SCORED


def score_answer(
    answer: str | None,
    reference_spec: dict | None,
    tool_output: str = "",
    condition: str = "full",
) -> QuestionScore:
    """Score one answer. `tool_output` is the concatenated tool results.

    Under the "graph" ablation the CSV layer is not loaded, so the questions
    that depend on it have a different correct answer: saying the value is not
    available. Scoring those against the CSV value would fail every model,
    including the ones that behave correctly, and would make the two conditions
    incomparable. The reference declares the expectation per question in
    `expected_without_csv`.
    """
    if reference_spec is None:
        return QuestionScore(None, "", "unknown", NOT_SCORED, detail="no reference entry")

    mode = reference_spec.get("validation_mode") or ""
    qid = reference_spec.get("question_id")
    family = FAMILY_BY_MODE.get(mode, "unknown")
    expectation = reference_spec.get("expected_without_csv")

    withheld = condition != "full" and bool(expectation)
    if withheld and expectation == "manual_review":
        return QuestionScore(qid, mode, "withheld_csv", NOT_SCORED,
                             detail="the question changes meaning without the CSV")
    # The withheld-CSV check comes before the "open" one: an interpretive
    # question about a CSV value still has one checkable answer once the CSV is
    # gone, namely that the value cannot be reached.
    if family == "open" and not withheld:
        return QuestionScore(qid, mode, family, NOT_SCORED, detail="needs judgement")
    if not answer:
        return QuestionScore(qid, mode, "withheld_csv" if withheld else family,
                             INCORRECT, detail="empty answer")

    facts = reference_spec.get("resolved_facts") or reference_spec.get("required_facts") or {}
    parsed = C.extract(answer, reference_spec.get("validation_policy"))

    if withheld:
        return _score_without_csv(qid, mode, parsed, facts, reference_spec, expectation)

    handler = {
        "exact": _score_exact,
        "exact_absence": _score_absence,
        "set_exact": _score_set,
        "set_and_value_exact": _score_set,
        "set_and_counts_exact": _score_set,
        "counts_exact": _score_counts,
        "per_object_exact": _score_per_object,
        "relationship_exact": _score_relationship,
        "coordinates": _score_coordinates,
        "tool_output_grounded": _score_tool_grounded,
    }.get(mode)

    if handler is None:
        return QuestionScore(qid, mode, family, NOT_SCORED, detail=f"no scorer for mode {mode!r}")

    checks = handler(parsed, facts, reference_spec, tool_output)
    inventions = _invention_checks(parsed, facts, reference_spec)
    if mode != "exact_absence":
        checks = checks + inventions
    if not checks:
        return QuestionScore(qid, mode, family, NOT_SCORED,
                             detail="reference facts not in a recognised shape")

    passed = sum(1 for _, ok, _ in checks if ok)
    failed = "; ".join(note for _, ok, note in checks if not ok)

    if inventions:
        # An invention is disqualifying, not a partial miss. An answer that has
        # the role of columns right but explains it through a roof the scene
        # does not contain is not half correct.
        return QuestionScore(qid, mode, family, INCORRECT, 0.0,
                             detail=failed, checks=checks)

    score = passed / len(checks)
    outcome = CORRECT if passed == len(checks) else (INCORRECT if passed == 0 else PARTIAL)
    return QuestionScore(qid, mode, family, outcome, score,
                         detail=failed or "all checks passed", checks=checks)


# --------------------------------------------------------------------------
# per-mode scorers. Each returns [(check name, passed, note)].
# --------------------------------------------------------------------------

ROLE_WORDS = (
    "strutturale", "strutturali", "structural", "portante", "portanti",
    "ornamentale", "ornamentali", "ornamental", "decorativ",
    "superficie di appoggio", "support surface", "piano di calpestio",
    "apertura", "opening", "circolazione", "circulation",
)

# Fact keys that are graph-derived, not CSV-derived: reporting them is
# legitimate even when the CSV layer is withheld.
_GRAPH_FACT_KEYS = frozenset({
    "object_total", "relationship_total", "annotated_count", "count", "answer",
    "objects_required", "must_only_use_present_classes", "values_must_match_tool_output",
})


def _score_without_csv(qid, mode, parsed, facts, spec, expectation) -> QuestionScore:
    """Score a CSV-dependent question in the condition where the CSV is absent.

    The correct answer is that the value cannot be reached. Producing one
    anyway is the invention this ablation exists to measure, and reproducing
    the CSV value itself is the strongest form of it - the model cannot have
    read it.
    """
    family = "withheld_csv"
    leaked = [
        value for key, value in facts.items()
        if key not in _GRAPH_FACT_KEYS and isinstance(value, str) and value.strip()
        and parsed.mentions(value)
    ]
    if leaked:
        return QuestionScore(qid, mode, family, INCORRECT, 0.0,
                             detail="reports a CSV value it could not read: " + "; ".join(leaked),
                             checks=[("no_csv_value", False, "; ".join(leaked))])

    if parsed.abstains:
        return QuestionScore(qid, mode, family, CORRECT, 1.0,
                             detail="declares the value unavailable",
                             checks=[("abstains", True, "")])

    if expectation == "abstention_or_role_only" and any(w in parsed.text for w in ROLE_WORDS):
        return QuestionScore(qid, mode, family, CORRECT, 1.0,
                             detail="restricts itself to the schema role",
                             checks=[("role_only", True, "")])

    return QuestionScore(qid, mode, family, INCORRECT, 0.0,
                         detail="neither declares the value unavailable nor stays on graph facts",
                         checks=[("abstains", False, "")])


def _score_exact(parsed, facts, spec, tool_output):
    checks = []
    for key, expected in facts.items():
        if key in _META_KEYS:
            continue
        checks.extend(_check_fact(parsed, key, expected))
    return checks


def _score_absence(parsed, facts, spec, tool_output):
    checks = []
    for key, expected in facts.items():
        if key in _META_KEYS:
            continue
        if key == "answer":
            # "Si. Gli archi sono assenti in questa scena" opens with a
            # discourse marker, not an affirmative answer. Treating the class
            # as absent is the substance of the denial, so it counts.
            absent_denied = any(
                label in parsed.classes_negated
                for label in (spec.get("scene_absent_classes") or [])
            )
            denied = parsed.polarity is False or parsed.abstains or absent_denied
            checks.append(("answer_is_negative", denied,
                           f"expected a negative answer, polarity={parsed.polarity}"))
        elif key.endswith("_relationships"):
            label = key[: -len("_relationships")]
            claimed = [r for r in parsed.relations if label in (r[0], r[2])]
            checks.append((key, not claimed, f"claims relations for absent {label}: {sorted(claimed)[:3]}"))
        elif key in C.CLASS_ALIASES and expected == 0:
            checks.append((f"{key}_absent", _absent(parsed, key),
                           f"{key} is absent but the answer does not say so"))
        else:
            checks.extend(_check_fact(parsed, key, expected))
    checks.extend(_invention_checks(parsed, facts, spec))
    return checks


def _invention_checks(parsed, facts, spec) -> list[tuple[str, bool, str]]:
    """Asserting something about a class the scene does not contain.

    Three assertions count, in any mode and whatever else the answer gets
    right: naming an object of that class, counting one or more of them, or
    putting the class in a relation. An answer that is right about the role of
    columns but explains it by saying they hold up the roof is not correct.

    A bare mention does not count. The reference already warns that a class
    name needs semantic context: the CSV typology for openings is "apertura ad
    arco", and a vault's function is a "copertura", so the words arch and roof
    appear in perfectly sound answers. Those produce no count and no relation,
    which is what separates them from an invention.

    The absent classes come from the reference scene facts, not from this
    question's required_facts: several absence questions state only
    {"answer": false}, so the class under test is not in the facts at all.
    """
    absent = set(spec.get("scene_absent_classes") or [])
    absent |= {key for key, value in facts.items() if key in C.CLASS_ALIASES and value == 0}
    checks = []
    for label in sorted(absent):
        invented = sorted(o for o in parsed.object_ids if o.rsplit("_", 1)[0] == label)
        if invented:
            checks.append((f"no_invented_{label}", False, f"invented objects: {invented}"))

        counted = parsed.class_counts.get(label)
        if counted:
            checks.append((f"no_count_for_{label}", False,
                           f"counts {counted} objects of absent class {label}"))

        related = sorted(r for r in parsed.relations if label in (r[0], r[2]))
        if related:
            checks.append((f"no_relation_with_{label}", False,
                           f"puts absent class {label} in a relation: {related[:2]}"))
    return checks


def _score_set(parsed, facts, spec, tool_output):
    checks = []
    for key, expected in facts.items():
        if key in _META_KEYS:
            continue
        if key == "present_classes":
            checks.append(_set_check("present_classes", parsed.classes_affirmed, set(expected)))
        elif key == "absent_classes":
            # "other" is a pseudo-class. The reference's own validation_policy
            # warns that the ordinary word is not the class, and no model names
            # it among the absent ones - but an answer that does is not wrong
            # either, so it is excluded from the comparison both ways.
            checks.append(_set_check(
                "absent_classes",
                parsed.classes_negated - {"other"},
                set(expected) - {"other"},
            ))
        elif key == "class_counts":
            checks.extend(_count_checks(parsed, expected))
        elif key == "annotated_count" and isinstance(expected, int):
            # Naming exactly that many objects states the count without
            # writing the number out.
            named = len(parsed.object_ids)
            checks.append((key, expected in parsed.numbers or named == expected,
                           f"neither states {expected} nor names {expected} objects (named {named})"))
        elif key == "unannotated_objects":
            missing = [o for o in expected if o not in parsed.object_ids]
            checks.append(("unannotated_objects", not missing, f"does not name {missing}"))
        elif isinstance(expected, list) and expected and _looks_like_ids(expected):
            checks.append(_set_check(key, parsed.object_ids & _id_universe(expected, parsed), set(expected)))
        elif isinstance(expected, list):
            checks.append(_set_check(key, parsed.classes_affirmed, set(expected)))
        else:
            checks.extend(_check_fact(parsed, key, expected))
    return checks


def _score_counts(parsed, facts, spec, tool_output):
    checks = []
    for key, expected in facts.items():
        if key in _META_KEYS:
            continue
        if isinstance(expected, dict):
            checks.extend(_count_checks(parsed, expected))
        else:
            checks.extend(_check_fact(parsed, key, expected))
    return checks


def _score_per_object(parsed, facts, spec, tool_output):
    checks = []
    for key, expected in facts.items():
        if key in _META_KEYS:
            continue
        if isinstance(expected, dict):
            for object_id, value in expected.items():
                named = object_id in parsed.object_ids
                stated = parsed.mentions(value)
                checks.append((f"{object_id}", named and stated,
                               f"{object_id}: named={named}, value_reported={stated}"))
        elif key == "objects_required" and isinstance(expected, int):
            checks.append(("objects_required", len(parsed.object_ids) >= expected,
                           f"named {len(parsed.object_ids)} objects, expected {expected}"))
        elif key == "fields_required":
            continue
        else:
            checks.extend(_check_fact(parsed, key, expected))
    return checks


def _score_relationship(parsed, facts, spec, tool_output):
    spec_rel = facts.get("relationship")
    if not isinstance(spec_rel, dict):
        return _score_exact(parsed, facts, spec, tool_output)
    source = spec_rel.get("source_class")
    target = spec_rel.get("target_class")
    relation = spec_rel.get("type")
    stated = any(
        r == relation and _matches_entity(s, source) and _matches_entity(t, target)
        for s, r, t in parsed.relations
    )
    return [(f"{source} {relation} {target}", stated,
             f"does not state {source} {relation} {target}")]


def _score_coordinates(parsed, facts, spec, tool_output):
    tolerance = float(spec.get("coordinate_tolerance_m", 0.02))
    centroids = facts.get("centroids") or {}
    checks = []
    for object_id, expected in centroids.items():
        close = any(
            all(abs(a - b) <= tolerance for a, b in zip(found, expected))
            for found in parsed.coordinates
        )
        checks.append((object_id, close, f"no coordinate within {tolerance} m of {expected}"))
    objects = facts.get("objects") or []
    if objects:
        checks.append(_set_check("objects", parsed.object_ids & _id_universe(objects, parsed), set(objects)))
    return checks


def _score_tool_grounded(parsed, facts, spec, tool_output):
    """Every relation stated must appear in what the tools actually returned."""
    tool_text = C.normalize(tool_output)
    checks = []
    for phrase in facts.get("must_include") or []:
        checks.append((f"includes:{phrase}", _phrase_stated(parsed, phrase),
                       f"does not state {phrase!r}"))
    if not tool_text:
        return checks
    unsupported = [
        (s, r, t) for s, r, t in parsed.relations
        if not (_token_in(tool_text, s) and _token_in(tool_text, t))
    ]
    checks.append(("relations_supported_by_tool_output", not unsupported,
                   f"not in tool output: {sorted(unsupported)[:3]}"))
    return checks


# --------------------------------------------------------------------------
# shared checks
# --------------------------------------------------------------------------

def _check_fact(parsed, key: str, expected) -> list[tuple[str, bool, str]]:
    """Handle the fact-key conventions used across the reference."""
    if key == "object_total" and isinstance(expected, int):
        stated = parsed.total_count
        if stated is None and set(parsed.numbers) == {expected}:
            # A bare answer such as "$\boxed{33}$" carries no total marker but
            # states one unambiguous number.
            stated = expected
        return [("object_total", stated == expected,
                 f"states total {parsed.total_count}, expected {expected}")]

    if key in C.CLASS_ALIASES and isinstance(expected, int):
        if expected == 0:
            return [(f"{key}_absent", _absent(parsed, key), f"does not report {key} as absent")]
        stated = parsed.class_counts.get(key)
        return [(f"{key}_count", stated == expected, f"states {key}={stated}, expected {expected}")]

    if isinstance(expected, bool):
        if key == "answer":
            match = parsed.polarity is expected
            return [("answer", match, f"polarity={parsed.polarity}, expected {expected}")]
        if key == "structural":
            claims_structural = "structural" in parsed.roles
            if "structural" in parsed.roles_negated:
                claims_structural = False
            return [("structural", claims_structural is expected,
                     f"claims structural={claims_structural}, expected {expected}")]
        return []

    if key.endswith("_role") and isinstance(expected, str):
        stated = expected in parsed.roles
        return [(key, stated, f"does not state role {expected!r}")]

    if isinstance(expected, str):
        return [(key, parsed.mentions(expected), f"does not report {key}={expected!r}")]

    if isinstance(expected, int):
        return [(key, expected in parsed.numbers, f"does not report {key}={expected}")]

    if isinstance(expected, list) and _looks_like_ids(expected):
        return [_set_check(key, parsed.object_ids & _id_universe(expected, parsed), set(expected))]

    return []


def _count_checks(parsed, expected: dict) -> list[tuple[str, bool, str]]:
    checks = []
    for label, value in expected.items():
        if value == 0:
            checks.append((f"{label}=0", _absent(parsed, label), f"does not report {label} as absent"))
        else:
            stated = parsed.class_counts.get(label)
            checks.append((f"{label}={value}", stated == value,
                           f"states {label}={stated}, expected {value}"))
    return checks


def _set_check(name: str, stated: set, expected: set) -> tuple[str, bool, str]:
    missing = sorted(expected - stated)
    extra = sorted(stated - expected)
    return (name, not missing and not extra, f"{name}: missing={missing}, unexpected={extra}")


def _absent(parsed, label: str) -> bool:
    """The answer treats `label` as absent: negated, counted zero, or unmentioned."""
    if label in parsed.classes_negated or parsed.class_counts.get(label) == 0:
        return True
    return label not in parsed.classes_affirmed


def _looks_like_ids(values: list) -> bool:
    return all(isinstance(v, str) and C.OBJECT_ID_RE.fullmatch(v) for v in values)


def _id_universe(expected: list[str], parsed) -> set[str]:
    """Only compare ids of the classes the question is about."""
    prefixes = {value.rsplit("_", 1)[0] for value in expected}
    return {o for o in parsed.object_ids if o.rsplit("_", 1)[0] in prefixes}


def _matches_entity(value: str, target: str | None) -> bool:
    if target is None:
        return False
    return value == target or value.rsplit("_", 1)[0] == target


def _phrase_stated(parsed, phrase: str) -> bool:
    """`must_include` phrases read like "column supports vault"."""
    tokens = C.normalize(phrase).split()
    labels = [t for t in tokens if t.rstrip("s") in C.CLASS_ALIASES or t in C.CLASS_ALIASES]
    relations = [r for r, aliases in C.RELATION_ALIASES.items() if any(a in C.normalize(phrase) for a in aliases)]
    if len(labels) >= 2 and relations:
        source, target = _canonical(labels[0]), _canonical(labels[-1])
        return any(
            r in relations and _matches_entity(s, source) and _matches_entity(t, target)
            for s, r, t in parsed.relations
        )
    return parsed.mentions(phrase)


def _canonical(token: str) -> str:
    token = token.rstrip(".,;:")
    for label, aliases in C.CLASS_ALIASES.items():
        if token == label or token in aliases or token.rstrip("s") == label:
            return label
    return token


def _token_in(text: str, value: str) -> bool:
    if value in C.CLASS_ALIASES:
        return any(alias in text for alias in C.CLASS_ALIASES[value])
    return value in text


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def aggregate(scores: list[QuestionScore]) -> dict:
    """Headline metrics for one run."""
    scored = [s for s in scores if s.is_scored]
    by_family: dict[str, dict] = {}
    for score in scored:
        bucket = by_family.setdefault(score.family, {"n": 0, "correct": 0, "partial": 0, "score": 0.0})
        bucket["n"] += 1
        bucket["correct"] += score.outcome == CORRECT
        bucket["partial"] += score.outcome == PARTIAL
        bucket["score"] += score.score

    for bucket in by_family.values():
        bucket["accuracy"] = round(bucket["correct"] / bucket["n"], 4) if bucket["n"] else 0.0
        bucket["partial_credit"] = round(bucket["score"] / bucket["n"], 4) if bucket["n"] else 0.0
        bucket.pop("score")

    correct = sum(1 for s in scored if s.outcome == CORRECT)
    return {
        "questions": len(scores),
        "scored": len(scored),
        "not_scored": len(scores) - len(scored),
        "coverage": round(len(scored) / len(scores), 4) if scores else 0.0,
        "accuracy": round(correct / len(scored), 4) if scored else 0.0,
        "partial_credit": round(sum(s.score for s in scored) / len(scored), 4) if scored else 0.0,
        "by_family": by_family,
    }
