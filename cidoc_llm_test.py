"""Standalone LLM test for CIDOC knowledge-graph tool routing."""

import argparse
import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from arch_agent.model_profiles import resolve_model_profile
from arch_agent.pipeline.pipeline import PipelineParams, run_pipeline
from arch_agent.tools.cidoc_tools import create_cidoc_tools


SYSTEM_PROMPT = """You answer questions about one architectural scene.
You have exactly one tool: query_cidoc_knowledge_graph.

Always call the tool before answering. Base every factual claim exclusively on
the returned CIDOC triples. Never replace a missing triple with architectural
common knowledge or raw spatial evidence. If the graph has no matching triple,
say so clearly.

Predicate map:
- material: P45 / crm:P45_consists_of
- typology or type: P2 / crm:P2_has_type
- function or intended use: P103 / crm:P103_was_intended_for
- wall bearing a molding feature: P56 / crm:P56_bears_feature
- opening forming part of a wall: P46i / crm:P46i_forms_part_of

Argument rules:
- object_name is only for an exact instance id such as column_0 or wall_3.
- semantic_label is only for a canonical scene class. Use these exact values:
  arch, column, door_window, floor, moldings, other, roof, stairs, vault, wall.
- Never put an Italian or English common noun such as muro, finestra, apertura,
  modanatura, wall, or window in object_name unless it is an exact id.
- outgoing means that the selected object/class is the triple subject/source.
- incoming means that the selected object/class is the triple object/target.

Exact routing examples:
- "Which moldings are features of walls?" or "Quali modanature sono feature
  dei muri?" -> semantic_label="wall", predicate="P56", direction="outgoing".
- "Which openings are part of walls?" or "Quali aperture fanno parte dei
  muri?" -> semantic_label="door_window", predicate="P46i",
  direction="outgoing".
- "Which windows are part of walls?" -> use semantic_label="door_window";
  door and window instances share that scene class.

CIDOC direction constraints:
- P46i has the opening as source and the wall as target. Therefore every
  question asking which doors/windows/openings form part of walls MUST use
  semantic_label="door_window" and direction="outgoing". Never select wall
  as the outgoing class for P46i.
- P56 has the wall as source and the molding as target. Therefore questions
  asking which moldings are features of walls use semantic_label="wall" and
  direction="outgoing".
- near, adjacent_to, above, and below belong to the spatial graph and are not
  CIDOC predicates in this standalone test. Do not reinterpret adjacency as
  P46i or P56. After checking the available CIDOC triples, state that spatial
  adjacency cannot be established with this tool.
- Resolve short follow-up questions such as "quali?", "which ones?", or
  pronouns from the immediately preceding user question and answer. Repeat
  that most recent graph filter, not an older filter from the conversation.
- When listing element-to-element triples, always report both endpoints using
  their exact segmented object_name values when the tool provides them, for
  example `door_window_3 -> wall_4`. Do not list only the CIDOC display label.
- Never collapse repeated source labels into "twice" or "four times". The same
  source may be connected to different targets. List every distinct
  source-predicate-target triple so that apparent duplicates remain auditable.

For all semantic information or all triples concerning an object, omit the
predicate filter. Preserve the language of the user's question. Keep the final
answer concise and distinguish graph facts from missing information.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test an Ollama model using only the standalone CIDOC tool."
    )
    parser.add_argument("point_cloud_path")
    parser.add_argument("--annotation-csv", required=True)
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--distance-threshold", type=float, default=2.0)
    parser.add_argument("--think", choices=("auto", "true", "false"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = run_pipeline(
        PipelineParams(
            point_cloud_path=args.point_cloud_path,
            annotation_csv_path=args.annotation_csv,
            distance_threshold=args.distance_threshold,
        )
    )
    tools = create_cidoc_tools(ctx)
    tools_by_name = {item.name: item for item in tools}
    profile = resolve_model_profile(args.model)
    llm_kwargs = {
        "model": args.model,
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        "temperature": 0.0,
        "num_ctx": profile.num_ctx,
    }
    think = profile.think if args.think == "auto" else args.think == "true"
    if think is not None:
        llm_kwargs["think"] = think
    if profile.num_predict is not None:
        llm_kwargs["num_predict"] = profile.num_predict

    llm = ChatOllama(**llm_kwargs)
    llm_with_tool = llm.bind_tools(tools)
    llm_with_required_tool = llm.bind_tools(tools, tool_choice="any")

    print("=" * 64)
    print(f"  Standalone CIDOC LLM Test | model: {args.model}")
    print("  Only query_cidoc_knowledge_graph is available")
    print("  Type 'quit' to exit.")
    print("=" * 64)

    conversation = [SystemMessage(content=SYSTEM_PROMPT)]
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            return
        if not question:
            continue

        conversation.append(HumanMessage(content=question))
        final_text = ""
        for step in range(5):
            runnable = llm_with_required_tool if step == 0 else llm_with_tool
            response = runnable.invoke(conversation)
            conversation.append(response)
            if not response.tool_calls:
                final_text = _message_text(response.content)
                break
            for call in response.tool_calls:
                print(f"[tool call] {call['name']}({call['args']})")
                tool_item = tools_by_name.get(call["name"])
                if tool_item is None:
                    result = f"Unknown tool: {call['name']}"
                else:
                    result = tool_item.invoke(call["args"])
                conversation.append(
                    ToolMessage(content=str(result), tool_call_id=call["id"])
                )
        if not final_text:
            final_text = "The model did not produce a final answer after the tool call limit."
        print(f"\nAgent: {final_text}")


def _message_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content).strip()


if __name__ == "__main__":
    main()
