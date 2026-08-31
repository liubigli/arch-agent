"""Deterministic groundedness checks for benchmark answers.

These are closed-vocabulary heuristics, not a general hallucination
detector: they only work because the scene graph domain is fully
enumerable (a fixed list of semantic classes, a fixed list of relation
types, and object names with a known naming pattern). They catch the
concrete failure mode observed in practice — the model's final answer
contradicts a fact that could have been read directly from the scene
graph — not open-ended factual errors about the world.

False negatives are expected (free text is not fully parseable); treat
flagged issues as "needs human review", not as a certified error count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..agent import (
    _asks_for_class_count,
    _asks_for_count,
    _extract_semantic_label,
    _normalize_text,
)
from ..pipeline.pipeline import SceneContext
from ..settings import get_config

VALID_RELATION_TYPES = {
    "near", "adjacent_to", "above", "below",
    "supports", "rests_on",
    "has_part", "part_of", "is_opening_in", "is_rib_of",
    "is_ornament_of", "is_attached_to", "is_placed_on", "is_connected_to",
}
INVALID_RELATION_TYPES = {"contains", "inside"}

_ABSENCE_MARKER_PATTERNS = tuple(
    re.compile(rf"\b{re.escape(marker)}")
    for marker in (
        "assent",  # assente/assenti
        "absent",
        "non presente", "non sono presenti", "not present",
        "mancano", "manca",
        "nessun",
        "none",
    )
) + (re.compile(r"\bno\b"),)
_OBJECT_NAME_RE = re.compile(r"\b([a-z_]+_\d+)\b")
_NUMBER_RE = re.compile(r"\d+")

ABSENT_CLASS_OK_PATTERNS = (
    "classe non presente",
    "classe assente",
    "non presente",
    "assente",
    "not present",
    "absent",
    "not found",
)

ABSENT_CLASS_BAD_PATTERNS = (
    "supporta",
    "supportano",
    "supportato",
    "supportata",
    "supported",
    "adjacent",
    "adiacente",
    "above",
    "below",
    "sopra",
    "sotto",
    "materiale",
    "material",
    "funzione",
    "function",
)

SUBSTITUTION_PATTERNS = (
    "se con roof intendi",
    "se per roof intendi",
    "roof-like",
    "equivale",
    "puo essere interpretata come",
    "può essere interpretata come",
)


@dataclass
class GroundingIssue:
    kind: str
    detail: str


def check_groundedness(
    ctx: SceneContext,
    question: str,
    final_answer: str | None,
) -> list[GroundingIssue]:
    if not final_answer:
        return []
    issues: list[GroundingIssue] = []
    issues.extend(_check_absent_class_claimed_present(ctx, final_answer))
    issues.extend(_check_invalid_relation_types(final_answer))
    issues.extend(_check_unknown_object_names(ctx, final_answer))
    issues.extend(_check_class_count_number(ctx, question, final_answer))
    issues.extend(_check_per_class_count_number(ctx, question, final_answer))
    issues.extend(_check_total_object_count(ctx, question, final_answer))
    return issues


def _present_classes(ctx: SceneContext) -> set[str]:
    return {obj["semantic_label"] for obj in ctx.objects.values()}


def _sentences(text: str) -> list[str]:
    return re.split(r"[.\n;]", text)


def _check_absent_class_claimed_present(
    ctx: SceneContext,
    answer: str,
) -> list[GroundingIssue]:
    present = _present_classes(ctx)
    absent = set(get_config()["semantic_classes"]["names"]) - present
    if not absent:
        return []

    normalized_answer = _normalize_text(answer)
    issues: list[GroundingIssue] = []
    for label in sorted(absent):
        label_pattern = re.compile(rf"\b{re.escape(label)}\b")
        if not label_pattern.search(normalized_answer):
            continue

        object_pattern = re.compile(rf"\b{re.escape(label)}_\d+\b")
        invents_objects = object_pattern.search(normalized_answer) is not None
        reports_absence = any(
            pattern in normalized_answer for pattern in ABSENT_CLASS_OK_PATTERNS
        ) or any(
            pattern.search(normalized_answer) for pattern in _ABSENCE_MARKER_PATTERNS
        )
        reports_zero = (
            f"{label}: 0" in normalized_answer
            or f"{label}=0" in normalized_answer
            or f"{label} = 0" in normalized_answer
            or f"0 {label}" in normalized_answer
            or f"nessun oggetto {label}" in normalized_answer
            or f"non ci sono {label}" in normalized_answer
            or f"no {label}" in normalized_answer
        )
        asserts_properties = any(
            label in sentence
            and any(bad in sentence for bad in ABSENT_CLASS_BAD_PATTERNS)
            for sentence in _sentences(normalized_answer)
        )
        substitutes_class = any(
            pattern in normalized_answer for pattern in SUBSTITUTION_PATTERNS
        )

        if (
            (reports_absence or reports_zero)
            and not invents_objects
            and not asserts_properties
            and not substitutes_class
        ):
            continue

        issues.append(
            GroundingIssue(
                kind="absent_class_claimed_present",
                detail=(
                    f"Class '{label}' is absent, but the answer treats it as "
                    "present or substitutes it with another class."
                ),
            )
        )
    return issues


def _check_invalid_relation_types(answer: str) -> list[GroundingIssue]:
    normalized = _normalize_text(answer)
    issues = []
    for relation_type in sorted(INVALID_RELATION_TYPES):
        if re.search(rf"\b{re.escape(relation_type)}\b", normalized):
            issues.append(
                GroundingIssue(
                    kind="invalid_relation_type",
                    detail=(
                        f"Answer uses relation type '{relation_type}', which does "
                        "not exist in this scene graph's relation vocabulary."
                    ),
                )
            )
    return issues


def _check_unknown_object_names(
    ctx: SceneContext,
    answer: str,
) -> list[GroundingIssue]:
    known_names = set(ctx.objects)
    known_labels = set(get_config()["semantic_classes"]["names"])
    issues = []
    seen = set()
    for match in _OBJECT_NAME_RE.finditer(answer.lower()):
        name = match.group(1)
        if name in known_names or name in seen:
            continue
        prefix = name.rsplit("_", 1)[0]
        if prefix not in known_labels:
            continue
        seen.add(name)
        issues.append(
            GroundingIssue(
                kind="unknown_object_name",
                detail=f"Answer references object '{name}', which is not in the scene.",
            )
        )
    return issues


def _first_number(text: str) -> int | None:
    match = _NUMBER_RE.search(text)
    return int(match.group()) if match else None


def _check_class_count_number(
    ctx: SceneContext,
    question: str,
    answer: str,
) -> list[GroundingIssue]:
    text = _normalize_text(question)
    if not _asks_for_class_count(text):
        return []
    expected = len(_present_classes(ctx))
    stated = _first_number(answer)
    if stated is not None and stated != expected:
        return [
            GroundingIssue(
                kind="class_count_mismatch",
                detail=f"Answer states {stated} classes, but the scene has {expected}.",
            )
        ]
    return []


def _check_per_class_count_number(
    ctx: SceneContext,
    question: str,
    answer: str,
) -> list[GroundingIssue]:
    text = _normalize_text(question)
    if not _asks_for_count(text):
        return []
    label = _extract_semantic_label(text)
    if label is None:
        return []
    expected = sum(1 for obj in ctx.objects.values() if obj["semantic_label"] == label)
    stated = _first_number(answer)
    if stated is not None and stated != expected:
        return [
            GroundingIssue(
                kind="object_count_mismatch",
                detail=(
                    f"Answer states {stated} objects of class '{label}', "
                    f"but the scene has {expected}."
                ),
            )
        ]
    return []


def _check_total_object_count(
    ctx: SceneContext,
    question: str,
    answer: str,
) -> list[GroundingIssue]:
    text = _normalize_text(question)
    if not _asks_for_count(text) or _extract_semantic_label(text) is not None:
        return []
    if not any(term in text for term in ("oggetti", "objects", "elementi", "elements")):
        return []
    expected = len(ctx.objects)
    stated = _first_number(answer)
    if stated is not None and stated != expected:
        return [
            GroundingIssue(
                kind="object_total_count_mismatch",
                detail=f"Answer states {stated} total objects, but the scene has {expected}.",
            )
        ]
    return []
