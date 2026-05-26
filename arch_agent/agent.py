from typing import Annotated

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from .pipeline.pipeline import SceneContext
from .tools.scene_tools import create_scene_tools

_SYSTEM_PROMPT = """\
You are an expert architectural analyst. You help users explore and understand \
3D architectural scenes described by a scene graph built from semantic point clouds.

The scene graph contains:
- NODES: detected architectural elements — arch, column, moldings, floor, \
door_window, wall, stairs, vault, roof, other
- EDGES: spatial relationships — near, adjacent, above, below, contains, inside

Element categories:
  Structural : arch, column, wall, vault, roof
  Finishing  : moldings, floor, door_window, stairs, other

Rules:
- Always use the available tools to retrieve data before answering.
- Do not assume or invent object names or counts.
- When the user asks a general question about the scene for the first time, \
start by calling get_scene_statistics.
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def create_agent(ctx: SceneContext, model: str = "llama3"):
    tools = create_scene_tools(ctx)
    llm = ChatOllama(model=model, base_url="http://localhost:11434", temperature=0.7)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def chat_node(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=_SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    graph = StateGraph(AgentState)
    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    return graph.compile()


def run_agent(ctx: SceneContext, model: str = "llama3") -> None:
    agent = create_agent(ctx, model=model)

    print("=" * 60)
    print("  Architectural Scene Agent  |  model: " + model)
    print("=" * 60)
    print(f"  Scene : {ctx.params.csv_path}")
    print(f"  Objects: {len(ctx.objects)}  |  Relationships: {len(ctx.relationships)}")
    print("  Type 'quit' to exit.\n")

    messages: list[BaseMessage] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        print(f"\nAgent: {messages[-1].content}\n")
