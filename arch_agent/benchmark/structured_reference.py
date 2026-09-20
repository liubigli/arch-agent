"""Loading and resolving per-question benchmark reference specifications."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_structured_reference(path: str | Path) -> dict[str, Any]:
    reference_path = Path(path)
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if payload.get("status") != "approved":
        raise ValueError(
            f"Structured reference is not approved: {reference_path} "
            f"(status={payload.get('status')!r})"
        )
    references = payload.get("references") or []
    ids = [item.get("question_id") for item in references]
    if len(references) != payload.get("questions_expected"):
        raise ValueError(
            f"Expected {payload.get('questions_expected')} reference entries, "
            f"found {len(references)}"
        )
    if len(ids) != len(set(ids)):
        raise ValueError("Structured reference contains duplicate question IDs.")
    return payload


def reference_for_question(
    payload: dict[str, Any] | None,
    question_id: int | None,
) -> dict[str, Any] | None:
    if payload is None or question_id is None:
        return None
    for entry in payload.get("references", []):
        if entry.get("question_id") == question_id:
            resolved = deepcopy(entry)
            resolved["resolved_facts"] = _resolve_required_facts(payload, entry)
            resolved["validation_policy"] = deepcopy(payload.get("validation_policy") or {})
            return resolved
    return None


def _resolve_required_facts(
    payload: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    facts = deepcopy(entry.get("required_facts") or {})
    for dotted_path in entry.get("required_facts_from") or []:
        value = _dotted_get(payload, dotted_path)
        facts[dotted_path.rsplit(".", 1)[-1]] = deepcopy(value)
    return facts


def _dotted_get(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Unknown structured-reference path: {dotted_path}")
        value = value[part]
    return value
