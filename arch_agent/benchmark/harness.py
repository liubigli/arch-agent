"""Benchmark harness for measuring tool-calling reliability of the scene agent.

Unlike the interactive CLI (`arch_agent.agent.run_agent`), every question here is
sent straight to the LLM+tools graph — the deterministic pattern-matching layer
(`_try_answer_deterministic`) is never used to short-circuit the answer. It is
only used, separately, to compute a reference answer for comparison.
"""

from __future__ import annotations

import csv
import json
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..agent import _try_answer_deterministic, create_agent
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
    result.reference_answer = _try_answer_deterministic(ctx, question)

    start = time.monotonic()
    try:
        outcome = agent.invoke({"messages": [HumanMessage(content=question)]})
    except Exception as exc:  # noqa: BLE001 - benchmark must survive per-question failures
        result.error = str(exc)
        result.latency_s = time.monotonic() - start
        return result
    result.latency_s = time.monotonic() - start

    messages = outcome["messages"]
    ai_messages = [message for message in messages if isinstance(message, AIMessage)]
    if len(ai_messages) > 1 and not ai_messages[0].tool_calls:
        leading_content = ai_messages[0].content
        if isinstance(leading_content, str) and leading_content.strip():
            result.chain_of_thought = leading_content.strip()

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

    result.grounding_issues = check_groundedness(ctx, question, result.final_answer)

    return result


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
        "chain_of_thought",
        "num_tool_calls",
        "tool_names",
        "tool_call_reasoning",
        "final_answer",
        "reference_answer",
        "grounding_issues",
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
                    "chain_of_thought": result.chain_of_thought or "",
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
                    "latency_s": f"{result.latency_s:.3f}",
                    "error": result.error or "",
                }
            )


def write_json_report(results: list[BenchmarkResult], path: str | Path) -> None:
    payload = [asdict(result) for result in results]
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize(results: list[BenchmarkResult]) -> str:
    total = len(results)
    errors = sum(1 for r in results if r.error)
    zero_tool_calls = sum(1 for r in results if not r.error and r.num_tool_calls == 0)
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
        f"Questions with a captured chain-of-thought: {with_chain_of_thought}/{total}",
        f"Average tool calls per question: {avg_calls:.2f}",
        f"Tool calls with stated reasoning: {reasoned_calls}/{total_calls} ({reasoning_rate:.0%})",
        f"Average latency per question: {avg_latency:.2f}s",
        f"Answers flagged by groundedness checks: {flagged}/{total}",
        "Tool usage:",
    ]
    for name, count in sorted(tool_usage.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"  - {name}: {count}")
    if issue_kinds:
        lines.append("Grounding issue breakdown:")
        for kind, count in sorted(issue_kinds.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"  - {kind}: {count}")
    return "\n".join(lines)