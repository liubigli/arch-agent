from collections import Counter
import re
from typing import Annotated
import unicodedata

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from pathlib import Path

from .pipeline.pipeline import SceneContext
from .pipeline.relationships import (
    RELATIONSHIP_LAYER_NAMES,
    RELATIONSHIP_LAYER_ORDER,
    architectural_role,
    mereological_relation_type,
    supports_label_pair,
)
from .tools.scene_tools import create_scene_tools
from .settings import get_config

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system.md"


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def create_agent(ctx: SceneContext, model: str = "llama3"):
    tools = create_scene_tools(ctx)
    llm = ChatOllama(model=model, base_url="http://localhost:11434", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def chat_node(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=_load_system_prompt())] + state["messages"]
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
    print(f"  Scene : {ctx.params.point_cloud_path}")
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

        deterministic_answer = _try_answer_deterministic(ctx, user_input)
        if deterministic_answer is not None:
            print(f"\nAgent: {deterministic_answer}\n")
            continue

        messages.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        print(f"\nAgent: {messages[-1].content}\n")


def _try_answer_deterministic(ctx: SceneContext, user_input: str) -> str | None:
    text = _normalize_text(user_input)
    requested_facts = _format_requested_facts(ctx, text)
    if requested_facts is not None:
        return requested_facts

    distance_answer = _try_answer_distance(ctx, text)
    if distance_answer is not None:
        return distance_answer

    if "incongruen" in text or "contraddizion" in text or "contraddittor" in text:
        return _format_relationship_inconsistencies(ctx)

    if _asks_for_relationships(text):
        return _format_relationships(ctx, level=_extract_relationship_level(text))

    if "bounding box" in text or "boundingn box" in text:
        return _format_point_cloud_info(ctx)

    if "pointcloud" in text or "point cloud" in text or "nuvola" in text:
        if "punti" in text or "points" in text or "bounding" in text:
            return _format_point_cloud_info(ctx)

    if "volume" in text and ("stanza" in text or "room" in text):
        return _format_room_volume(ctx)

    if "volume" in text and "bounding" in text:
        return _format_point_cloud_info(ctx)

    if "rgb" in text or "colore" in text or "color" in text:
        label = _extract_semantic_label(text)
        return _format_color_summary(ctx, semantic_label=label)

    if "maggior numero di punti" in text or "piu punti" in text:
        return _format_top_object(ctx, metric="point_count")

    if "volume maggiore" in text or "maggior volume" in text or "piu volume" in text:
        return _format_top_object(ctx, metric="volume")

    if "piu compatto" in text or "geometricamente compatto" in text:
        return _format_top_object(ctx, metric="compactness", reverse=False)

    if _asks_for_scene_inventory(text):
        return _format_scene_inventory(ctx)

    return None


def _format_requested_facts(ctx: SceneContext, text: str) -> str | None:
    sections: list[tuple[str, str]] = []

    if _asks_for_scene_inventory(text):
        sections.append(("Inventario", _format_scene_inventory(ctx)))
    if "maggior numero di punti" in text or "piu punti" in text:
        sections.append(("Elemento con piu punti", _format_top_object(ctx, metric="point_count")))
    if "volume maggiore" in text or "maggior volume" in text or "piu volume" in text:
        sections.append(("Elemento con volume maggiore", _format_top_object(ctx, metric="volume")))
    if "piu compatto" in text or "geometricamente compatto" in text:
        sections.append(("Elemento piu compatto", _format_top_object(ctx, metric="compactness", reverse=False)))
    if "volume" in text and ("stanza" in text or "room" in text):
        sections.append(("Volume stanza", _format_room_volume(ctx)))
    if "bounding box" in text or "boundingn box" in text:
        sections.append(("Point cloud", _format_point_cloud_info(ctx)))

    if not sections:
        return None

    return "\n\n".join(f"{title}\n{body}" for title, body in sections)


def _try_answer_distance(ctx: SceneContext, text: str) -> str | None:
    distance_terms = ("distanza", "dista", "distano", "vicino", "vicini", "nearest", "closest")
    if not any(term in text for term in distance_terms):
        return None

    object_names = _extract_object_names(text, ctx.objects)
    if len(object_names) >= 2:
        return _format_distance(ctx, object_names[0], object_names[1])

    if len(object_names) == 1 and any(term in text for term in ("vicino", "vicini", "nearest", "closest")):
        semantic_label = _extract_semantic_label(text)
        if semantic_label == ctx.objects[object_names[0]]["semantic_label"]:
            semantic_label = None
        return _format_nearest_objects(ctx, object_names[0], semantic_label=semantic_label)

    return None


def _extract_object_names(text: str, objects: dict) -> list[str]:
    found = []
    for match in re.finditer(r"\b[a-z]+(?:_[a-z]+)*_\d+\b", text):
        name = match.group(0)
        if name in objects and name not in found:
            found.append(name)
    return found


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _asks_for_relationships(text: str) -> bool:
    relationship_words = ("relazione", "relazioni", "relationship", "relationships")
    list_words = ("tutte", "tutti", "lista", "elenco", "fornisc", "elenca", "l1", "l2", "l3")
    return any(word in text for word in relationship_words) and any(
        word in text for word in list_words
    )


def _extract_relationship_level(text: str) -> str:
    if "l1" in text or "geometric" in text or "geometrich" in text:
        return "L1"
    if "l2" in text or "structural" in text or "struttural" in text:
        return "L2"
    if "l3" in text or "mereolog" in text:
        return "L3"
    return "all"


def _format_relationships(ctx: SceneContext, level: str = "all", limit: int = 200) -> str:
    if level == "all":
        return _format_relationships_cascade(ctx, limit=limit)

    relationships = ctx.relationship_layers.get(level, [])
    lines = [f"Relazioni {level}: {len(relationships)}"]
    type_counts = Counter(rel_type for _, _, rel_type, _ in relationships)
    if type_counts:
        lines.append("Distribuzione per tipo:")
        lines.extend(f"  - {rel_type}: {count}" for rel_type, count in sorted(type_counts.items()))

    shown = relationships[:limit]
    lines.append(f"Prime {len(shown)} relazioni:")
    for src, tgt, rel_type, rel_level in shown:
        lines.append(f"  - {src} --[{rel_level}:{rel_type}]--> {tgt}")
    if len(relationships) > limit:
        lines.append(
            f"  ... {len(relationships) - limit} relazioni non mostrate per evitare "
            "di saturare il contesto della chat."
        )
    elif not relationships:
        lines.append("  Nessuna relazione trovata.")
    return "\n".join(lines)


def _format_relationships_cascade(ctx: SceneContext, limit: int = 200) -> str:
    max_rows = max(1, min(int(limit), 1000))
    total = sum(len(ctx.relationship_layers.get(level, [])) for level in RELATIONSHIP_LAYER_ORDER)
    lines = [
        f"Relazioni all: {total}",
        "Ordine di analisi: L1/geometric -> L2/structural -> L3/mereological",
    ]

    remaining = max_rows
    hidden = 0
    for level in RELATIONSHIP_LAYER_ORDER:
        relationships = ctx.relationship_layers.get(level, [])
        layer_name = RELATIONSHIP_LAYER_NAMES.get(level, level)
        lines.append(f"{level}/{layer_name}: {len(relationships)}")

        type_counts = Counter(rel_type for _, _, rel_type, _ in relationships)
        if type_counts:
            lines.append(
                "  Tipi: "
                + ", ".join(f"{rel_type}={count}" for rel_type, count in sorted(type_counts.items()))
            )

        shown = relationships[:remaining] if remaining > 0 else []
        for src, tgt, rel_type, rel_level in shown:
            lines.append(f"  - {src} --[{rel_level}:{rel_type}]--> {tgt}")

        hidden += max(0, len(relationships) - len(shown))
        remaining -= len(shown)

    if hidden:
        lines.append(
            f"  ... {hidden} relazioni non mostrate per evitare "
            "di saturare il contesto della chat."
        )
    elif total == 0:
        lines.append("  Nessuna relazione trovata.")
    return "\n".join(lines)


def _format_relationship_inconsistencies(ctx: SceneContext) -> str:
    pair_relations: dict[frozenset[str], list[tuple[str, str, str, str]]] = {}
    for src, tgt, rel_type, rel_level in ctx.relationships:
        pair_relations.setdefault(frozenset((src, tgt)), []).append(
            (src, tgt, rel_type, rel_level)
        )

    issues = []
    suspicious = []
    for pair, rels in pair_relations.items():
        if len(pair) != 2:
            continue
        objects = list(pair)
        a, b = objects[0], objects[1]
        rel_set = {(src, tgt, rel_type, rel_level) for src, tgt, rel_type, rel_level in rels}

        if (
            (a, b, "above", "geometric") in rel_set
            and (b, a, "above", "geometric") in rel_set
        ):
            issues.append(f"{a} e {b}: entrambi risultano 'above' l'uno rispetto all'altro.")
        if (
            (a, b, "below", "geometric") in rel_set
            and (b, a, "below", "geometric") in rel_set
        ):
            issues.append(f"{a} e {b}: entrambi risultano 'below' l'uno rispetto all'altro.")
        if (
            (a, b, "supports", "structural") in rel_set
            and (b, a, "supports", "structural") in rel_set
        ):
            issues.append(f"{a} e {b}: entrambi risultano supportarsi reciprocamente.")
        if (
            (a, b, "rests_on", "structural") in rel_set
            and (b, a, "rests_on", "structural") in rel_set
        ):
            issues.append(f"{a} e {b}: entrambi risultano appoggiati l'uno sull'altro.")

        for src, tgt, rel_type, rel_level in rels:
            src_label = ctx.objects.get(src, {}).get("semantic_label")
            tgt_label = ctx.objects.get(tgt, {}).get("semantic_label")
            if rel_type in {"contains", "inside"}:
                issues.append(
                    f"{src} -> {tgt}: relazione '{rel_type}' non prevista dal modello relazionale corrente."
                )
            if rel_type == "is_opening_in" and not (src_label == "door_window" and tgt_label == "wall"):
                issues.append(
                    f"{src} -> {tgt}: is_opening_in non valida per classi {src_label}->{tgt_label}."
                )
            if rel_type == "has_part" and src_label == "floor" and tgt_label == "door_window":
                issues.append(
                    f"{src} -> {tgt}: un floor non dovrebbe contenere una door_window."
                )
            if rel_type == "supports" and not supports_label_pair(src_label, tgt_label):
                issues.append(
                    f"{src} -> {tgt}: supports non ammessa per classi {src_label}->{tgt_label}."
                )
            if rel_type == "rests_on" and not supports_label_pair(tgt_label, src_label):
                issues.append(
                    f"{src} -> {tgt}: rests_on non ammessa per classi {src_label}->{tgt_label}."
                )
            if rel_level == "mereological":
                if rel_type == "has_part":
                    expected = mereological_relation_type(tgt_label, src_label)
                    if expected is None:
                        issues.append(
                            f"{src} -> {tgt}: has_part senza regola mereologica inversa "
                            f"per classi {src_label}->{tgt_label}."
                        )
                else:
                    expected = mereological_relation_type(src_label, tgt_label)
                    if expected is None:
                        issues.append(
                            f"{src} -> {tgt}: relazione mereologica '{rel_type}' non ammessa "
                            f"per classi {src_label}->{tgt_label}."
                        )
                    elif rel_type != expected:
                        issues.append(
                            f"{src} -> {tgt}: relazione mereologica '{rel_type}' diversa "
                            f"da quella attesa '{expected}' per classi {src_label}->{tgt_label}."
                        )
            if rel_type == "is_placed_on" and not (src_label == "stairs" and tgt_label == "floor"):
                suspicious.append(
                    f"{src} -> {tgt}: is_placed_on inattesa per classi {src_label}->{tgt_label}."
                )

    if not issues and not suspicious:
        return (
            "Non emergono contraddizioni dirette nel grafo calcolato "
            "(es. above reciproco, below reciproco, supports reciproco, contains/inside non previsti)."
        )

    lines = [f"Incongruenze dirette trovate: {len(issues)}"]
    lines.extend(f"  - {issue}" for issue in issues[:200])
    if len(issues) > 200:
        lines.append(f"  ... {len(issues) - 200} incongruenze non mostrate.")
    if suspicious:
        lines.append(f"Anomalie sospette: {len(suspicious)}")
        lines.extend(f"  - {issue}" for issue in suspicious[:100])
        if len(suspicious) > 100:
            lines.append(f"  ... {len(suspicious) - 100} anomalie non mostrate.")
    return "\n".join(lines)


def _format_point_cloud_info(ctx: SceneContext) -> str:
    if ctx.df is None or ctx.df.empty:
        return "Non sono disponibili dati della point cloud."

    mins = ctx.df[["x", "y", "z"]].min()
    maxs = ctx.df[["x", "y", "z"]].max()
    dims = maxs - mins
    volume = float(dims["x"] * dims["y"] * dims["z"])
    class_counts = ctx.df["semantic_label"].value_counts().sort_index()

    lines = [
        f"La point cloud contiene {len(ctx.df):,} punti.",
        "Bounding box:",
        f"  Min (x,y,z): ({mins['x']:.2f}, {mins['y']:.2f}, {mins['z']:.2f})",
        f"  Max (x,y,z): ({maxs['x']:.2f}, {maxs['y']:.2f}, {maxs['z']:.2f})",
        f"  Dimensioni (x,y,z): ({dims['x']:.2f}, {dims['y']:.2f}, {dims['z']:.2f}) m",
        f"  Volume AABB: {volume:.3f} m3",
        "Classi nei punti:",
    ]
    lines.extend(f"  - {label}: {count:,}" for label, count in class_counts.items())
    lines.append(
        "RGB: " + ("disponibile" if _has_rgb(ctx.df) else "non disponibile")
    )
    return "\n".join(lines)


def _format_distance(ctx: SceneContext, object_a: str, object_b: str) -> str:
    metrics = _distance_metrics(ctx.objects[object_a], ctx.objects[object_b])
    return "\n".join([
        f"Distanza tra {object_a} e {object_b}:",
        f"  Distanza tra centroidi: {metrics['centroid_distance']:.3f} m",
        f"  Gap tra bounding box: {metrics['bbox_gap']:.3f} m",
        f"  Gap per asse (x,y,z): ({metrics['gap_x']:.3f}, {metrics['gap_y']:.3f}, {metrics['gap_z']:.3f}) m",
        f"  Gap verticale: {metrics['vertical_gap']:.3f} m",
        f"  Overlap XY: {metrics['xy_overlap_ratio']:.3f}",
        "  Bounding box a contatto/sovrapposte: "
        + ("si" if metrics["touching_or_overlapping"] else "no"),
    ])


def _format_nearest_objects(
    ctx: SceneContext,
    object_name: str,
    semantic_label: str | None = None,
    limit: int = 10,
) -> str:
    rows = []
    for candidate_name, candidate in ctx.objects.items():
        if candidate_name == object_name:
            continue
        if semantic_label and candidate["semantic_label"] != semantic_label:
            continue
        metrics = _distance_metrics(ctx.objects[object_name], candidate)
        rows.append((candidate_name, candidate["semantic_label"], metrics))

    rows.sort(key=lambda row: (row[2]["bbox_gap"], row[2]["centroid_distance"]))
    lines = [
        f"Oggetti piu vicini a {object_name}"
        + (f" filtrati per classe {semantic_label}" if semantic_label else "")
        + f": {len(rows)} candidati"
    ]
    for candidate_name, label, metrics in rows[:limit]:
        lines.append(
            f"  - {candidate_name} ({label}): "
            f"bbox_gap={metrics['bbox_gap']:.3f} m, "
            f"centroide={metrics['centroid_distance']:.3f} m, "
            f"gap_verticale={metrics['vertical_gap']:.3f} m, "
            f"overlap_xy={metrics['xy_overlap_ratio']:.3f}"
        )
    if not rows:
        lines.append("  Nessun oggetto corrispondente trovato.")
    return "\n".join(lines)


def _distance_metrics(obj_a: dict, obj_b: dict) -> dict:
    c_a = obj_a["centroid"]
    c_b = obj_b["centroid"]
    gaps = _axis_gaps(obj_a["bounds"], obj_b["bounds"])
    bbox_gap = _norm(gaps)
    return {
        "centroid_distance": _norm(c_a - c_b),
        "bbox_gap": bbox_gap,
        "gap_x": gaps[0],
        "gap_y": gaps[1],
        "gap_z": gaps[2],
        "vertical_gap": _signed_vertical_gap(obj_a["bounds"], obj_b["bounds"]),
        "xy_overlap_ratio": _overlap_xy_ratio(obj_a["bounds"], obj_b["bounds"]),
        "touching_or_overlapping": bbox_gap == 0.0,
    }


def _axis_gaps(bounds_a: dict, bounds_b: dict) -> list[float]:
    return [
        _axis_gap(
            float(bounds_a["min"][axis]),
            float(bounds_a["max"][axis]),
            float(bounds_b["min"][axis]),
            float(bounds_b["max"][axis]),
        )
        for axis in range(3)
    ]


def _axis_gap(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    if max_a < min_b:
        return float(min_b - max_a)
    if max_b < min_a:
        return float(min_a - max_b)
    return 0.0


def _signed_vertical_gap(bounds_a: dict, bounds_b: dict) -> float:
    if bounds_a["max"][2] < bounds_b["min"][2]:
        return float(bounds_b["min"][2] - bounds_a["max"][2])
    if bounds_b["max"][2] < bounds_a["min"][2]:
        return float(bounds_a["min"][2] - bounds_b["max"][2])
    return 0.0


def _overlap_xy_ratio(bounds_a: dict, bounds_b: dict) -> float:
    x_overlap = max(
        0.0,
        min(bounds_a["max"][0], bounds_b["max"][0])
        - max(bounds_a["min"][0], bounds_b["min"][0]),
    )
    y_overlap = max(
        0.0,
        min(bounds_a["max"][1], bounds_b["max"][1])
        - max(bounds_a["min"][1], bounds_b["min"][1]),
    )
    reference_area = min(_xy_area(bounds_a), _xy_area(bounds_b))
    if reference_area <= 0:
        return 0.0
    return float((x_overlap * y_overlap) / reference_area)


def _norm(values) -> float:
    return sum(float(value) ** 2 for value in values) ** 0.5


def _format_room_volume(ctx: SceneContext) -> str:
    room_volume = ctx.scene_features.get("room_volume", {})
    if not room_volume:
        return (
            "Non posso stimare il volume della stanza: la feature "
            "scene_features['room_volume'] non e disponibile. Servono almeno "
            "un floor e un elemento tra wall, column, roof o vault."
        )
    floor_dims = room_volume["floor_base_dimensions"]
    return "\n".join([
        "Volume stimato della stanza come box contenitore:",
        f"  Floor usato: {room_volume['floor_object']}",
        f"  Dimensioni base floor (AABB XY): {floor_dims[0]:.3f} x {floor_dims[1]:.3f} m",
        f"  Superficie base: {room_volume['floor_base_area']:.3f} m2",
        f"  Quota inferiore: top del floor = {room_volume['lower_z']:.3f} m",
        f"  Quota superiore: max Z di wall/column/roof/vault = {room_volume['upper_z']:.3f} m",
        f"  Altezza box: {room_volume['height']:.3f} m",
        f"  Volume: {room_volume['volume']:.3f} m3",
        "  Feature: scene_features['room_volume']",
        "  Formula: area base del floor x altezza del box.",
    ])


def _format_color_summary(ctx: SceneContext, semantic_label: str | None = None) -> str:
    if semantic_label:
        frames = [
            obj["points"] for obj in ctx.objects.values()
            if obj["semantic_label"] == semantic_label
        ]
        if not frames:
            return f"Non sono stati trovati oggetti di classe '{semantic_label}'."
        import pandas as pd

        df = pd.concat(frames, ignore_index=True)
        name = semantic_label
    else:
        df = ctx.df
        name = "scena"

    color = _mean_rgb(df)
    if color is None:
        return f"Il valore RGB non e disponibile per {name}."
    raw, rgb8 = color
    return "\n".join([
        f"Colore medio per {name}:",
        f"  RGB raw: ({raw[0]:.1f}, {raw[1]:.1f}, {raw[2]:.1f})",
        f"  RGB 8-bit: ({rgb8[0]}, {rgb8[1]}, {rgb8[2]})",
    ])


def _format_top_object(ctx: SceneContext, metric: str, reverse: bool = True) -> str:
    if metric == "point_count":
        if not ctx.objects:
            return "Non sono disponibili oggetti nella scena."
        name, obj = max(ctx.objects.items(), key=lambda item: item[1]["point_count"])
        return (
            "Elemento con il maggior numero di punti:\n"
            f"  - {name} ({obj['semantic_label']}): {obj['point_count']:,} punti"
        )

    candidates = [
        (name, ctx.objects[name]["semantic_label"], ctx.features[name][metric])
        for name in ctx.objects
        if name in ctx.features and metric in ctx.features[name]
    ]
    if not candidates:
        return f"Non sono disponibili valori per la metrica '{metric}'."

    name, label, value = sorted(candidates, key=lambda item: item[2], reverse=reverse)[0]
    metric_label = {
        "volume": "volume AABB maggiore",
        "compactness": "compactness minore",
    }.get(metric, metric)
    return "\n".join([
        f"Elemento con {metric_label}:",
        f"  - {name} ({label}): {value:.3f}",
    ])


def _format_scene_inventory(ctx: SceneContext) -> str:
    class_counts = Counter(obj["semantic_label"] for obj in ctx.objects.values())
    all_classes = set(get_config()["semantic_classes"]["names"])
    role_counts = Counter()
    for label, count in class_counts.items():
        role_counts[architectural_role(label)] += count
    absent = sorted(all_classes - set(class_counts))

    lines = [f"La scena contiene {len(ctx.objects)} oggetti."]
    lines.append("Classi presenti:")
    lines.extend(f"  - {label}: {count}" for label, count in sorted(class_counts.items()))
    lines.append(
        "Classi assenti: " + (", ".join(absent) if absent else "nessuna")
    )
    lines.append("Ruoli architettonici:")
    lines.extend(f"  - {role}: {count}" for role, count in sorted(role_counts.items()))
    return "\n".join(lines)


def _asks_for_scene_inventory(text: str) -> bool:
    inventory_terms = (
        "quanti oggetti",
        "quanti elementi",
        "che tipi",
        "classi",
        "classe",
        "inventario",
        "assenti",
        "strutturali",
        "decorativi",
        "finishing",
    )
    return any(term in text for term in inventory_terms)


def _extract_semantic_label(text: str) -> str | None:
    aliases = {
        "colonne": "column",
        "colonna": "column",
        "columns": "column",
        "column": "column",
        "muri": "wall",
        "muro": "wall",
        "pareti": "wall",
        "wall": "wall",
        "pavimento": "floor",
        "floor": "floor",
        "tetto": "roof",
        "roof": "roof",
        "volta": "vault",
        "vault": "vault",
        "porte": "door_window",
        "finestre": "door_window",
        "door": "door_window",
        "window": "door_window",
    }
    for word, label in aliases.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return label
    return None


def _has_rgb(df) -> bool:
    return df is not None and all(column in df.columns for column in ["R", "G", "B"])


def _mean_rgb(df) -> tuple[tuple[float, float, float], tuple[int, int, int]] | None:
    if not _has_rgb(df) or df.empty:
        return None
    raw = tuple(float(value) for value in df[["R", "G", "B"]].mean().to_numpy())
    max_channel = max(float(df[["R", "G", "B"]].max().max()), 1.0)
    divisor = 257.0 if max_channel > 255 else 1.0
    rgb8 = tuple(int(round(min(max(value / divisor, 0), 255))) for value in raw)
    return raw, rgb8


def _xy_area(bounds: dict) -> float:
    dims = bounds["max"][:2] - bounds["min"][:2]
    return float(max(dims[0], 0.0) * max(dims[1], 0.0))
