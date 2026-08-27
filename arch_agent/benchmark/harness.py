"""Benchmark harness for measuring tool-calling reliability of the scene agent.

Unlike the interactive CLI (`arch_agent.agent.run_agent`), every question here is
sent straight to the LLM+tools graph — the deterministic pattern-matching layer
(`_try_answer_deterministic`) is never used to short-circuit the answer. It is
only used, separately, to compute a reference answer for comparison.

Reports also include language compliance fields, so English questions can be
checked for English final answers and Italian questions for Italian answers.
"""

from __future__ import annotations

import csv
import json
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..agent import (
    _needs_language_repair,
    _response_language,
    _try_answer_deterministic,
    create_agent,
)
from ..pipeline.pipeline import SceneContext
from .grounding_checks import GroundingIssue, check_groundedness


@dataclass
class ToolCall:
    name: str
    args: dict
    output: str | None
    reasoning: str | None = None


@dataclass
class BenchmarkResult:
    question: str
    model: str
    expected_language: str | None = None
    language_ok: bool | None = None
    language_issue: str | None = None
    chain_of_thought: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str | None = None
    reference_answer: str | None = None
    grounding_issues: list[GroundingIssue] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None

    @property
    def num_tool_calls(self) -> int:
        return len(self.tool_calls)

    @property
    def reliability(self) -> str:
        """Coarse reliability verdict for `final_answer`, mutually exclusive:

        - "error": the LLM/tool pipeline itself failed (exception, unreachable Ollama).
        - "unanswered": no final answer text was produced at all.
        - "ungrounded": an answer was produced but contradicts data verifiable
          from the scene graph (see `grounding_checks.py`).
        - "unverified_no_tool": an answer was produced without calling any
          tool, so groundedness could not even be checked against fresh data.
        - "grounded": an answer was produced, at least one tool was called,
          and no contradiction was found.

        This is NOT a correctness score against `reference_answer` — see the
        module docstring: reference answers for interpretive questions
        (typology, hierarchy, dominant element, ...) are one heuristic
        reading among several plausible ones, not ground truth.
        """
        if self.error:
            return "error"
        if not self.final_answer:
            return "unanswered"
        if self.grounding_issues:
            return "ungrounded"
        if self.num_tool_calls == 0:
            return "unverified_no_tool"
        return "grounded"


def load_questions(path: str | Path) -> list[str]:
    """Extract one question per line from a plain-text question list.

    Lines are kept if they end with '?' once stripped of leading list
    markers ('-', digits, dots). Duplicate questions (common across the
    "secche si/no" and "sequenza consigliata" sections of these files) are
    dropped, keeping the first occurrence.
    """
    text = Path(path).read_text(encoding="utf-8")
    seen: set[str] = set()
    questions: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-").strip()
        if not line.endswith("?"):
            continue
        line = _strip_leading_ordinal(line)
        key = _normalize(line)
        if key in seen:
            continue
        seen.add(key)
        questions.append(line)
    return questions


def _strip_leading_ordinal(line: str) -> str:
    head, _, rest = line.partition(".")
    if head.strip().isdigit() and rest.strip():
        return rest.strip()
    return line


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(c for c in normalized if not unicodedata.combining(c))
    return " ".join(without_accents.split())


def run_question(agent, ctx: SceneContext, model: str, question: str) -> BenchmarkResult:
    result = BenchmarkResult(question=question, model=model)
    result.expected_language = _response_language(question)
    result.reference_answer = _try_answer_deterministic(ctx, question)

    start = time.monotonic()
    try:
        outcome = agent.invoke({"messages": [HumanMessage(content=question)]})
    except Exception as exc:  # noqa: BLE001 - benchmark must survive per-question failures
        result.error = str(exc)
        result.latency_s = time.monotonic() - start
        return result
    result.latency_s = time.monotonic() - start
    result.chain_of_thought = outcome.get("reasoning")

    messages = outcome["messages"]
    tool_outputs = {
        message.tool_call_id: message.content
        for message in messages
        if isinstance(message, ToolMessage)
    }
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        reasoning = message.content.strip() if isinstance(message.content, str) else None
        for tool_call in message.tool_calls or []:
            result.tool_calls.append(
                ToolCall(
                    name=tool_call["name"],
                    args=tool_call.get("args", {}),
                    output=tool_outputs.get(tool_call.get("id")),
                    reasoning=reasoning or None,
                )
            )

    if messages and isinstance(messages[-1], AIMessage):
        result.final_answer = messages[-1].content

    _set_language_check(result)
    result.grounding_issues = check_groundedness(ctx, question, result.final_answer)

    return result


def _set_language_check(result: BenchmarkResult) -> None:
    if result.error:
        return
    if not result.final_answer:
        result.language_issue = "No final answer to check."
        return

    expected = result.expected_language or _response_language(result.question)
    result.language_ok = not _needs_language_repair(result.final_answer, expected)
    if result.language_ok:
        result.language_issue = None
        return

    language_name = "English" if expected == "en" else "Italian"
    result.language_issue = f"Expected final answer in {language_name}."


def build_trace(result: BenchmarkResult) -> list[str]:
    """Reconstruct a step-by-step trace from real execution data only.

    Unlike `chain_of_thought` (a prediction made before any tool ran, which
    can only guess at tool names and invent results), this is built purely
    from what actually happened — the real tool calls, their real
    arguments, their real output, and the real final answer — so it carries
    no additional hallucination risk and scales to any number of cascaded
    tool calls.

    Returns one string per step (not pre-joined), so JSON output keeps each
    step as a separate list item and stays easy to follow; CSV output joins
    them into a single cell.
    """
    steps = [f"1. Domanda: {result.question}"]
    step = 2
    if result.expected_language:
        steps.append(f"{step}. Lingua attesa: {result.expected_language}")
        step += 1
    if result.error:
        steps.append(f"{step}. Errore: {result.error}")
        return steps
    if not result.tool_calls:
        steps.append(f"{step}. Nessun tool chiamato.")
        step += 1
    for call in result.tool_calls:
        args_text = ", ".join(f"{key}={value!r}" for key, value in call.args.items())
        steps.append(f"{step}. Tool chiamato: {call.name}({args_text})")
        step += 1
        steps.append(f"{step}. Risultato del tool: {call.output or '(nessun output)'}")
        step += 1
    steps.append(f"{step}. Risposta finale: {result.final_answer or '(nessuna risposta)'}")
    step += 1
    if result.language_ok is not None:
        status = "ok" if result.language_ok else "errore"
        detail = f" ({result.language_issue})" if result.language_issue else ""
        steps.append(f"{step}. Controllo lingua: {status}{detail}")
    return steps


def run_benchmark(
    ctx: SceneContext,
    model: str,
    questions: list[str],
    capture_reasoning: bool = False,
    on_result=None,
) -> list[BenchmarkResult]:
    agent = create_agent(ctx, model=model, capture_reasoning=capture_reasoning)
    results = []
    for question in questions:
        result = run_question(agent, ctx, model, question)
        results.append(result)
        if on_result:
            on_result(result)
    return results


def write_csv_report(results: list[BenchmarkResult], path: str | Path) -> None:
    fieldnames = [
        "question",
        "model",
        "expected_language",
        "language_ok",
        "language_issue",
        "chain_of_thought",
        "trace",
        "num_tool_calls",
        "tool_names",
        "tool_call_reasoning",
        "final_answer",
        "reference_answer",
        "grounding_issues",
        "reliability",
        "latency_s",
        "error",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "question": result.question,
                    "model": result.model,
                    "expected_language": result.expected_language or "",
                    "language_ok": (
                        "" if result.language_ok is None else str(result.language_ok)
                    ),
                    "language_issue": result.language_issue or "",
                    "chain_of_thought": result.chain_of_thought or "",
                    "trace": "\n".join(build_trace(result)),
                    "num_tool_calls": result.num_tool_calls,
                    "tool_names": ", ".join(call.name for call in result.tool_calls),
                    "tool_call_reasoning": " | ".join(
                        f"{call.name}: {call.reasoning}"
                        for call in result.tool_calls
                        if call.reasoning
                    ),
                    "final_answer": result.final_answer or "",
                    "reference_answer": result.reference_answer or "",
                    "grounding_issues": " | ".join(
                        f"{issue.kind}: {issue.detail}" for issue in result.grounding_issues
                    ),
                    "reliability": result.reliability,
                    "latency_s": f"{result.latency_s:.3f}",
                    "error": result.error or "",
                }
            )


def write_json_report(results: list[BenchmarkResult], path: str | Path) -> None:
    payload = []
    for result in results:
        item = asdict(result)
        item["trace"] = build_trace(result)
        item["reliability"] = result.reliability
        payload.append(item)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize(results: list[BenchmarkResult]) -> str:
    total = len(results)
    errors = sum(1 for r in results if r.error)
    zero_tool_calls = sum(1 for r in results if not r.error and r.num_tool_calls == 0)
    language_checked = sum(1 for r in results if r.language_ok is not None)
    language_mismatches = sum(1 for r in results if r.language_ok is False)
    reliability_counts: dict[str, int] = {}
    for result in results:
        reliability_counts[result.reliability] = reliability_counts.get(result.reliability, 0) + 1
    scoreable = total - errors
    grounded_rate = reliability_counts.get("grounded", 0) / scoreable if scoreable else 0.0
    with_chain_of_thought = sum(1 for r in results if r.chain_of_thought)
    flagged = sum(1 for r in results if r.grounding_issues)
    issue_kinds: dict[str, int] = {}
    for result in results:
        for issue in result.grounding_issues:
            issue_kinds[issue.kind] = issue_kinds.get(issue.kind, 0) + 1
    tool_usage: dict[str, int] = {}
    for result in results:
        for call in result.tool_calls:
            tool_usage[call.name] = tool_usage.get(call.name, 0) + 1
    total_calls = sum(tool_usage.values())
    reasoned_calls = sum(
        1 for r in results for call in r.tool_calls if call.reasoning
    )
    avg_calls = total_calls / total if total else 0.0
    avg_latency = sum(r.latency_s for r in results) / total if total else 0.0
    reasoning_rate = reasoned_calls / total_calls if total_calls else 0.0

    lines = [
        f"Questions: {total}",
        f"Errors (LLM/tool invocation failed): {errors}",
        f"Answered with zero tool calls: {zero_tool_calls}",
        f"Language mismatches: {language_mismatches}/{language_checked}",
        f"Questions with a captured chain-of-thought: {with_chain_of_thought}/{total}",
        f"Average tool calls per question: {avg_calls:.2f}",
        f"Tool calls with stated reasoning: {reasoned_calls}/{total_calls} ({reasoning_rate:.0%})",
        f"Average latency per question: {avg_latency:.2f}s",
        f"Answers flagged by groundedness checks: {flagged}/{total}",
        f"Grounded rate (grounded / (total - errors)): {grounded_rate:.0%}",
        "Reliability breakdown:",
    ]
    for label, count in sorted(reliability_counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"  - {label}: {count}")
    lines.append("Tool usage:")
    for name, count in sorted(tool_usage.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"  - {name}: {count}")
    if issue_kinds:
        lines.append("Grounding issue breakdown:")
        for kind, count in sorted(issue_kinds.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"  - {kind}: {count}")
    return "\n".join(lines)
