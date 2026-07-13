from collections import Counter, defaultdict
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

from .evaluation_answers import answer_evaluation_prompt
from .pipeline.point_metrics import (
    format_material_summary as format_point_material_summary,
    format_rgb_summary as format_point_rgb_summary,
    format_roughness_summary,
    has_rgb,
)
from .pipeline.pipeline import SceneContext
from .pipeline.relationships import (
    Relationship,
    RELATIONSHIP_LAYER_NAMES,
    RELATIONSHIP_LAYER_ORDER,
    architectural_role,
    mereological_relation_type,
    supports_label_pair,
)
from .tools.scene_tools import create_scene_tools
from .settings import get_config

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system.md"

_SEMANTIC_ALIASES = (
    ("porta finestra", "door_window"),
    ("porta-finestra", "door_window"),
    ("porte finestre", "door_window"),
    ("porte-finestre", "door_window"),
    ("archi", "arch"),
    ("arco", "arch"),
    ("arches", "arch"),
    ("arch", "arch"),
    ("colonne", "column"),
    ("colonna", "column"),
    ("columns", "column"),
    ("column", "column"),
    ("aperture", "door_window"),
    ("apertura", "door_window"),
    ("muri", "wall"),
    ("muro", "wall"),
    ("pareti", "wall"),
    ("parete", "wall"),
    ("walls", "wall"),
    ("wall", "wall"),
    ("pavimenti", "floor"),
    ("pavimento", "floor"),
    ("floors", "floor"),
    ("floor", "floor"),
    ("tetti", "roof"),
    ("tetto", "roof"),
    ("coperture", "roof"),
    ("copertura", "roof"),
    ("roofs", "roof"),
    ("roof", "roof"),
    ("volte", "vault"),
    ("volta", "vault"),
    ("vaults", "vault"),
    ("vault", "vault"),
    ("scale", "stairs"),
    ("scala", "stairs"),
    ("stairs", "stairs"),
    ("stair", "stairs"),
    ("modanature", "moldings"),
    ("modanatura", "moldings"),
    ("moldings", "moldings"),
    ("molding", "moldings"),
    ("porte", "door_window"),
    ("porta", "door_window"),
    ("finestre", "door_window"),
    ("finestra", "door_window"),
    ("doors", "door_window"),
    ("door", "door_window"),
    ("windows", "door_window"),
    ("window", "door_window"),
    ("altro", "other"),
    ("other", "other"),
)


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
    last_semantic_label: str | None = None

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

        text = _normalize_text(user_input)
        labels_in_input = _extract_semantic_labels(text)
        deterministic_answer = _try_answer_deterministic(
            ctx,
            user_input,
            default_label=last_semantic_label,
        )
        if labels_in_input:
            last_semantic_label = labels_in_input[0]
        if deterministic_answer is not None:
            print(f"\nAgent: {deterministic_answer}\n")
            continue

        messages.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        print(f"\nAgent: {messages[-1].content}\n")


def _try_answer_deterministic(
    ctx: SceneContext,
    user_input: str,
    default_label: str | None = None,
) -> str | None:
    language = _response_language(user_input)
    evaluation_answer = answer_evaluation_prompt(ctx, user_input)
    if evaluation_answer is not None:
        return evaluation_answer

    text = _normalize_text(user_input)
    semantic_count = _try_answer_semantic_count(ctx, text, language=language)
    if semantic_count is not None:
        return _format_grounded_answer(
            observed=semantic_count,
            relations=_phrase(
                language,
                it="Nessuna relazione L1/L2/L3 usata: risposta basata sul conteggio degli oggetti per classe semantica.",
                en="No L1/L2/L3 relationship used: answer based on the semantic-class object count.",
            ),
            inference=_phrase(
                language,
                it="Conteggio diretto degli oggetti segmentati; non implica valutazioni su funzione, importanza o qualita della segmentazione.",
                en="Direct count of segmented objects; it does not imply function, importance, or segmentation quality.",
            ),
            confidence=_phrase(
                language,
                it="alta per il conteggio nel grafo corrente; media se la segmentazione semantica e incerta.",
                en="high for the count in the current graph; medium if semantic segmentation is uncertain.",
            ),
            language=language,
        )

    class_count = _try_answer_class_count(ctx, text, language=language)
    if class_count is not None:
        return _format_grounded_answer(
            observed=class_count,
            relations=_phrase(
                language,
                it="Nessuna relazione L1/L2/L3 usata: risposta basata sulle classi semantiche presenti.",
                en="No L1/L2/L3 relationship used: answer based on the semantic classes present.",
            ),
            inference=_phrase(
                language,
                it="Conteggio delle classi riconosciute nella segmentazione; non descrive relazioni o funzioni.",
                en="Count of classes recognized by segmentation; it does not describe relationships or functions.",
            ),
            confidence=_phrase(
                language,
                it="alta per le label presenti nel grafo corrente; media se la segmentazione semantica e incerta.",
                en="high for labels present in the current graph; medium if semantic segmentation is uncertain.",
            ),
            language=language,
        )

    support_answer = _try_answer_support_between_classes(ctx, text, language=language)
    if support_answer is not None:
        return support_answer

    open_support_answer = _try_answer_open_support_question(ctx, text, language=language)
    if open_support_answer is not None:
        return open_support_answer

    class_relationships = _try_answer_class_relationships(
        ctx,
        text,
        default_label=default_label,
        language=language,
    )
    if class_relationships is not None:
        return class_relationships

    requested_facts = _format_requested_facts(ctx, text)
    if requested_facts is not None:
        return _format_grounded_answer(
            observed=requested_facts,
            relations="Nessuna relazione L1/L2/L3 usata: risposta basata su conteggi, classi o feature.",
            inference="Sintesi descrittiva derivata dai dati disponibili, senza interpretazioni strutturali aggiuntive.",
            confidence="alta: i valori provengono direttamente dal contesto della scena.",
        )

    distance_answer = _try_answer_distance(ctx, text)
    if distance_answer is not None:
        return _format_grounded_answer(
            observed=distance_answer,
            relations="L1/geometric: metriche di distanza, gap tra bounding box e overlap XY.",
            inference="La vicinanza e una relazione geometrica; non implica da sola contatto, supporto o appartenenza.",
            confidence="alta per le misure geometriche; media per eventuali interpretazioni spaziali.",
        )

    if "incongruen" in text or "contraddizion" in text or "contraddittor" in text:
        return _format_grounded_answer(
            observed=_format_relationship_inconsistencies(ctx),
            relations="Controllo incrociato su L1/geometric, L2/structural e L3/mereological.",
            inference="Le anomalie indicano conflitti logici o relazioni non coerenti con le regole architettoniche.",
            confidence="media: dipende dalla qualita della segmentazione e dalle soglie geometriche.",
        )

    if _asks_for_relationships(text):
        level = _extract_relationship_level(text)
        return _format_grounded_answer(
            observed=_format_relationships(ctx, level=level),
            relations=_relationship_usage_text(level),
            inference="Nessuna inferenza aggiuntiva: elenco delle relazioni calcolate nel grafo.",
            confidence="alta per le relazioni elencate; media per il loro significato architettonico se basato solo su L1.",
        )

    if "bounding box" in text or "boundingn box" in text:
        return _format_grounded_answer(
            observed=_format_point_cloud_info(ctx),
            relations="Nessuna relazione usata: risposta basata sulla point cloud e sulla bounding box globale.",
            inference="Il volume AABB descrive l'estensione geometrica, non il volume architettonico abitabile.",
            confidence="alta: dati calcolati direttamente dalle coordinate della point cloud.",
        )

    if "pointcloud" in text or "point cloud" in text or "nuvola" in text:
        if "punti" in text or "points" in text or "bounding" in text:
            return _format_grounded_answer(
                observed=_format_point_cloud_info(ctx),
                relations="Nessuna relazione usata: risposta basata sulla point cloud.",
                inference="Descrizione geometrica globale della nuvola, senza interpretazione architettonica.",
                confidence="alta: dati letti direttamente dal dataframe della point cloud.",
            )

    if "volume" in text and ("stanza" in text or "room" in text):
        return _format_grounded_answer(
            observed=_format_room_volume(ctx),
            relations="Relazioni non usate direttamente: stima basata su floor ed envelope verticale della scena.",
            inference="Il volume stanza e una stima semplificata come box contenitore.",
            confidence="media: dipende dalla qualita dei floor e degli elementi verticali rilevati.",
        )

    if "volume" in text and "bounding" in text:
        return _format_grounded_answer(
            observed=_format_point_cloud_info(ctx),
            relations="Nessuna relazione usata: volume calcolato dalla bounding box della point cloud.",
            inference="Volume puramente geometrico, non equivalente al volume funzionale dello spazio.",
            confidence="alta per il calcolo AABB; bassa se interpretato come volume architettonico.",
        )

    if _asks_for_material(text):
        object_names = _extract_object_names(text, ctx.objects)
        object_name = object_names[0] if object_names else None
        label = _extract_semantic_label(text)
        if object_name:
            label = ctx.objects[object_name]["semantic_label"]
        return _format_grounded_answer(
            observed=_format_material_summary(
                ctx,
                semantic_label=label,
                object_name=object_name,
                language=language,
            ),
            relations=_phrase(
                language,
                it="Nessuna relazione L1/L2/L3 usata: risposta basata su classe semantica, RGB e rugosita locale.",
                en="No L1/L2/L3 relationship used: answer based on semantic class, RGB, and local roughness.",
            ),
            inference=_phrase(
                language,
                it=(
                    "Il materiale e proposto come candidato probabilistico: colore e rugosita "
                    "possono dipendere da illuminazione, acquisizione, rumore o degrado."
                ),
                en=(
                    "Material is proposed as a probabilistic candidate: color and roughness "
                    "can depend on lighting, acquisition, noise, or decay."
                ),
            ),
            confidence=_phrase(
                language,
                it="media-bassa: inferenza euristica, non analisi materica calibrata.",
                en="medium-low: heuristic inference, not calibrated material analysis.",
            ),
            language=language,
        )

    if _asks_for_surface_roughness(text):
        object_names = _extract_object_names(text, ctx.objects)
        object_name = object_names[0] if object_names else None
        label = None if object_name else _extract_semantic_label(text)
        return _format_grounded_answer(
            observed=_format_surface_roughness_summary(
                ctx,
                semantic_label=label,
                object_name=object_name,
                language=language,
            ),
            relations=_phrase(
                language,
                it="Nessuna relazione L1/L2/L3 usata: risposta basata sui punti XYZ della point cloud.",
                en="No L1/L2/L3 relationship used: answer based on XYZ point-cloud coordinates.",
            ),
            inference=_phrase(
                language,
                it=(
                    "La rugosita stimata descrive lo scarto locale dei punti da un piano; "
                    "puo includere rumore, curvatura e artefatti di segmentazione."
                ),
                en=(
                    "Estimated roughness describes local point residuals from a plane; "
                    "it can include noise, curvature, and segmentation artifacts."
                ),
            ),
            confidence=_phrase(
                language,
                it="media: metrica geometrica automatica, non misura materica assoluta.",
                en="medium: automatic geometric metric, not an absolute material measurement.",
            ),
            language=language,
        )

    if "rgb" in text or "colore" in text or "color" in text:
        object_names = _extract_object_names(text, ctx.objects)
        object_name = object_names[0] if object_names else None
        label = _extract_semantic_label(text)
        return _format_grounded_answer(
            observed=_format_color_summary(
                ctx,
                semantic_label=None if object_name else label,
                object_name=object_name,
                language=language,
            ),
            relations=_phrase(
                language,
                it="Nessuna relazione usata: risposta basata sui canali RGB dei punti.",
                en="No relationship used: answer based on point RGB channels.",
            ),
            inference=_phrase(
                language,
                it="Il colore e una feature visiva, non una prova funzionale o strutturale.",
                en="Color is a visual feature, not functional or structural evidence.",
            ),
            confidence=_phrase(
                language,
                it="alta se RGB e disponibile; altrimenti non disponibile.",
                en="high if RGB is available; otherwise unavailable.",
            ),
            language=language,
        )

    if "maggior numero di punti" in text or "piu punti" in text:
        return _format_grounded_answer(
            observed=_format_top_object(ctx, metric="point_count"),
            relations="Nessuna relazione usata: confronto basato sul numero di punti degli oggetti.",
            inference="Un alto numero di punti puo indicare dominanza geometrica o maggiore copertura, non importanza architettonica certa.",
            confidence="alta per il ranking numerico; media per l'interpretazione di dominanza.",
        )

    if "volume maggiore" in text or "maggior volume" in text or "piu volume" in text:
        return _format_grounded_answer(
            observed=_format_top_object(ctx, metric="volume"),
            relations="Nessuna relazione usata: confronto basato sul volume AABB degli oggetti.",
            inference="Il volume AABB misura ingombro geometrico, non necessariamente importanza funzionale.",
            confidence="alta per il ranking geometrico; media per l'interpretazione architettonica.",
        )

    if "piu compatto" in text or "geometricamente compatto" in text:
        return _format_grounded_answer(
            observed=_format_top_object(ctx, metric="compactness", reverse=False),
            relations="Nessuna relazione usata: confronto basato sulla metrica di compactness.",
            inference="La compactness e una proprieta geometrica, non una classificazione semantica.",
            confidence="media: dipende dalla qualita della superficie stimata.",
        )

    if _asks_for_scene_inventory(text):
        return _format_grounded_answer(
            observed=_format_scene_inventory(ctx),
            relations="Nessuna relazione usata: inventario basato sulle classi semantiche degli oggetti.",
            inference="I ruoli architettonici derivano dall'ontologia Python, non da una nuova osservazione geometrica.",
            confidence="alta per conteggi e label; media per i ruoli se la segmentazione e incerta.",
        )

    return None


def _format_grounded_answer(
    observed: str,
    relations: str,
    inference: str,
    confidence: str,
    language: str = "it",
) -> str:
    if language == "en":
        return "\n".join([
            "Observed data:",
            observed,
            "",
            "Relationships used:",
            relations,
            "",
            "Inference:",
            inference,
            "",
            "Confidence:",
            confidence,
        ])

    return "\n".join([
        "Osservato dai dati:",
        observed,
        "",
        "Relazioni usate:",
        relations,
        "",
        "Inferenza:",
        inference,
        "",
        "Confidenza:",
        confidence,
    ])


def _response_language(text: str) -> str:
    normalized = _normalize_text(text)
    english_markers = (
        "how many",
        "what",
        "which",
        "does",
        "do ",
        "is ",
        "are ",
        "support",
        "supported",
        "relationship",
        "relationships",
        "inside",
        "outside",
        "mixed",
        "load-bearing",
        "typology",
        "material",
        "materials",
        "roughness",
        "surface",
        "rgb",
        "color",
    )
    italian_markers = (
        "quante",
        "quanti",
        "quali",
        "cosa",
        "che ",
        "scena",
        "relazioni",
        "supporta",
        "sostiene",
        "sorregge",
        "intern",
        "estern",
        "mista",
        "portant",
        "tipologia",
        "material",
        "materic",
        "rugos",
        "ruvid",
        "asperit",
        "colore",
    )
    english_score = sum(marker in normalized for marker in english_markers)
    italian_score = sum(marker in normalized for marker in italian_markers)
    return "en" if english_score > italian_score else "it"


def _phrase(language: str, *, it: str, en: str) -> str:
    return en if language == "en" else it


def _relationship_usage_text(level: str) -> str:
    if level == "L1":
        return "L1/geometric: near, adjacent_to, above, below."
    if level == "L2":
        return "L2/structural: supports, rests_on, filtrate dalle regole architettoniche."
    if level == "L3":
        return "L3/mereological: has_part, is_opening_in, is_ornament_of, is_attached_to e relazioni parte-tutto."
    return "Cascata completa: prima L1/geometric, poi L2/structural, infine L3/mereological."


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


def _try_answer_semantic_count(
    ctx: SceneContext,
    text: str,
    language: str = "it",
) -> str | None:
    if not _asks_for_count(text):
        return None

    label = _extract_semantic_label(text)
    if label is None:
        return None

    names = sorted(
        name
        for name, obj in ctx.objects.items()
        if obj["semantic_label"] == label
    )
    lines = [
        _phrase(
            language,
            it=f"Oggetti di classe '{label}': {len(names)}.",
            en=f"Objects of class '{label}': {len(names)}.",
        )
    ]
    if names:
        lines.append(
            _phrase(
                language,
                it="Istanze: ",
                en="Instances: ",
            )
            + ", ".join(names)
        )
    return "\n".join(lines)


def _asks_for_count(text: str) -> bool:
    count_terms = (
        "quante",
        "quanti",
        "numero di",
        "conteggio",
        "count",
        "how many",
    )
    return any(term in text for term in count_terms)


def _try_answer_class_count(
    ctx: SceneContext,
    text: str,
    language: str = "it",
) -> str | None:
    if not _asks_for_class_count(text):
        return None

    class_counts = Counter(obj["semantic_label"] for obj in ctx.objects.values())
    if not class_counts:
        return _phrase(
            language,
            it="Non sono presenti classi semantiche nella scena.",
            en="No semantic classes are present in the scene.",
        )

    class_list = ", ".join(sorted(class_counts))
    lines = [
        _phrase(
            language,
            it=f"Classi semantiche presenti: {len(class_counts)} ({class_list}).",
            en=f"Semantic classes present: {len(class_counts)} ({class_list}).",
        ),
        _phrase(language, it="Distribuzione per classe:", en="Class distribution:"),
    ]
    lines.extend(
        f"  - {label}: {count}"
        for label, count in sorted(class_counts.items())
    )
    return "\n".join(lines)


def _asks_for_class_count(text: str) -> bool:
    count_terms = ("quante", "quanti", "numero di", "conteggio", "how many")
    class_terms = ("classi", "classe", "semantic classes", "semantic class")
    return any(term in text for term in count_terms) and any(
        term in text for term in class_terms
    )


def _try_answer_support_between_classes(
    ctx: SceneContext,
    text: str,
    language: str = "it",
) -> str | None:
    if not _asks_for_support(text):
        return None

    labels = _extract_semantic_labels(text)
    if len(labels) < 2:
        return None

    if _is_passive_support_question(text):
        upper_label, lower_label = labels[0], labels[1]
    else:
        lower_label, upper_label = labels[0], labels[1]

    supports = [
        rel for rel in ctx.relationship_layers.get("L2", [])
        if rel[2] == "supports"
        and ctx.objects.get(rel[0], {}).get("semantic_label") == lower_label
        and ctx.objects.get(rel[1], {}).get("semantic_label") == upper_label
    ]

    if supports:
        return _phrase(
            language,
            it=f"Si: {len(supports)} relazioni L2 supports {lower_label} -> {upper_label}.",
            en=f"Yes: {len(supports)} L2 supports relationships {lower_label} -> {upper_label}.",
        )

    if supports_label_pair(lower_label, upper_label):
        return _phrase(
            language,
            it=f"No: nessuna relazione L2 supports {lower_label} -> {upper_label} nella scena.",
            en=f"No: no L2 supports relationship {lower_label} -> {upper_label} in the scene.",
        )

    return _phrase(
        language,
        it=f"No: l'ontologia non ammette {lower_label} -> {upper_label} come supporto.",
        en=f"No: the ontology does not allow {lower_label} -> {upper_label} as a support relation.",
    )


def _try_answer_open_support_question(
    ctx: SceneContext,
    text: str,
    language: str = "it",
) -> str | None:
    if not _asks_for_support(text):
        return None

    labels = _extract_semantic_labels(text)
    if len(labels) != 1:
        return None

    label = labels[0]
    if _is_passive_support_question(text) or _asks_what_supports_subject(text):
        supports = [
            rel for rel in ctx.relationship_layers.get("L2", [])
            if rel[2] == "supports"
            and ctx.objects.get(rel[1], {}).get("semantic_label") == label
        ]
        return _format_open_support_brief(ctx, label, supports, direction="in", language=language)
    elif _asks_what_subject_supports(text):
        supports = [
            rel for rel in ctx.relationship_layers.get("L2", [])
            if rel[2] == "supports"
            and ctx.objects.get(rel[0], {}).get("semantic_label") == label
        ]
        return _format_open_support_brief(ctx, label, supports, direction="out", language=language)
    else:
        return None


def _format_open_support_brief(
    ctx: SceneContext,
    label: str,
    supports: list[Relationship],
    direction: str,
    language: str = "it",
) -> str:
    if not supports:
        if direction == "out":
            return _phrase(
                language,
                it=f"{label} non supporta nessuna classe tramite relazioni L2.",
                en=f"{label} does not support any class through L2 relationships.",
            )
        return _phrase(
            language,
            it=f"{label} non e supportato da nessuna classe tramite relazioni L2.",
            en=f"{label} is not supported by any class through L2 relationships.",
        )

    class_index = 1 if direction == "out" else 0
    class_counts = Counter(
        ctx.objects.get(rel[class_index], {}).get("semantic_label", "unknown")
        for rel in supports
    )
    summary = ", ".join(
        f"{class_label}={count}"
        for class_label, count in sorted(class_counts.items())
    )
    if direction == "out":
        return _phrase(
            language,
            it=f"{label} supporta: {summary} (L2 supports).",
            en=f"{label} supports: {summary} (L2 supports).",
        )
    return _phrase(
        language,
        it=f"{label} e supportato da: {summary} (L2 supports).",
        en=f"{label} is supported by: {summary} (L2 supports).",
    )


def _asks_what_subject_supports(text: str) -> bool:
    patterns = (
        "cosa supportano",
        "che cosa supportano",
        "cosa sostengono",
        "che cosa sostengono",
        "cosa sorreggono",
        "che cosa sorreggono",
        "what do",
        "what does",
        "which elements do",
        "which elements does",
    )
    return any(pattern in text for pattern in patterns)


def _asks_what_supports_subject(text: str) -> bool:
    patterns = (
        "da cosa",
        "da che cosa",
        "chi support",
        "cosa supporta",
        "cosa sostiene",
        "cosa sorregge",
        "what supports",
        "what is supporting",
        "which elements support",
    )
    return any(pattern in text for pattern in patterns)


def _format_support_targets_for_label(ctx: SceneContext, label: str) -> str:
    object_names = _objects_with_semantic_label(ctx, label)
    supports = [
        rel for rel in ctx.relationship_layers.get("L2", [])
        if rel[2] == "supports"
        and ctx.objects.get(rel[0], {}).get("semantic_label") == label
    ]
    return _format_support_relationships(
        ctx,
        label,
        object_names,
        supports,
        direction="out",
    )


def _format_support_sources_for_label(ctx: SceneContext, label: str) -> str:
    object_names = _objects_with_semantic_label(ctx, label)
    supports = [
        rel for rel in ctx.relationship_layers.get("L2", [])
        if rel[2] == "supports"
        and ctx.objects.get(rel[1], {}).get("semantic_label") == label
    ]
    return _format_support_relationships(
        ctx,
        label,
        object_names,
        supports,
        direction="in",
    )


def _format_support_relationships(
    ctx: SceneContext,
    label: str,
    object_names: list[str],
    supports: list[Relationship],
    direction: str,
) -> str:
    lines = [
        f"Oggetti di classe '{label}': {len(object_names)}"
        + (f" ({', '.join(object_names)})" if object_names else ""),
    ]
    if not supports:
        relation_text = "in uscita" if direction == "out" else "in ingresso"
        lines.append(f"Nessuna relazione L2 supports {relation_text} trovata.")
        return "\n".join(lines)

    target_index = 1 if direction == "out" else 0
    class_counts = Counter(
        ctx.objects.get(rel[target_index], {}).get("semantic_label", "unknown")
        for rel in supports
    )
    lines.append(f"Relazioni L2 supports trovate: {len(supports)}.")
    lines.append(
        "Classi coinvolte: "
        + ", ".join(f"{class_label}={count}" for class_label, count in sorted(class_counts.items()))
    )
    for src, tgt, _, _ in supports[:30]:
        lines.append(f"  - {src} --[structural:supports]--> {tgt}")
    if len(supports) > 30:
        lines.append(f"  ... {len(supports) - 30} non mostrate.")
    return "\n".join(lines)


def _try_answer_class_relationships(
    ctx: SceneContext,
    text: str,
    default_label: str | None = None,
    language: str = "it",
) -> str | None:
    if not _asks_for_class_relationships(text):
        return None

    labels = _extract_semantic_labels(text)
    label = labels[0] if labels else default_label
    if label is None:
        return None

    observed = _format_class_relationship_summary(ctx, label, language=language)
    return _format_grounded_answer(
        observed=observed,
        relations=_phrase(
            language,
            it=(
                "Cascata L1->L2->L3: riepilogo delle relazioni che coinvolgono "
                f"oggetti di classe '{label}', raggruppate per altra classe, tipo e direzione."
            ),
            en=(
                "L1->L2->L3 cascade: summary of relationships involving "
                f"objects of class '{label}', grouped by other class, type, and direction."
            ),
        ),
        inference=_phrase(
            language,
            it=(
                "Le relazioni L1 descrivono vicinanza, adiacenza e sopra/sotto; "
                "solo L2 supports/rests_on viene trattato come evidenza strutturale. "
                "L3, se presente, resta una relazione parte-tutto o di appartenenza."
            ),
            en=(
                "L1 relationships describe proximity, adjacency, and above/below; "
                "only L2 supports/rests_on is treated as structural evidence. "
                "L3, when present, remains a part-whole or belonging relationship."
            ),
        ),
        confidence=_phrase(
            language,
            it=(
                "alta per i conteggi del grafo; media per l'interpretazione architettonica "
                "perche dipende dalle soglie geometriche e dalla segmentazione."
            ),
            en=(
                "high for graph counts; medium for architectural interpretation "
                "because it depends on geometric thresholds and segmentation."
            ),
        ),
        language=language,
    )


def _asks_for_class_relationships(text: str) -> bool:
    relationship_terms = ("relazione", "relazioni", "relationship", "relationships")
    if not any(term in text for term in relationship_terms):
        return False
    if _extract_semantic_labels(text):
        return True

    class_terms = (
        "classe",
        "classi",
        "altre classi",
        "semantic",
        "semantich",
        "con le altre",
        "con gli altri",
        "con altri",
    )
    return any(term in text for term in class_terms)


def _format_class_relationship_summary(
    ctx: SceneContext,
    label: str,
    limit: int = 40,
    language: str = "it",
) -> str:
    object_names = set(_objects_with_semantic_label(ctx, label))
    if not object_names:
        return _phrase(
            language,
            it=f"Nessun oggetto di classe '{label}' trovato nella scena.",
            en=f"No object of class '{label}' found in the scene.",
        )

    counts: Counter[tuple[str, str, str, str, str]] = Counter()
    examples: dict[tuple[str, str, str, str, str], list[Relationship]] = defaultdict(list)

    for level in RELATIONSHIP_LAYER_ORDER:
        for relationship in ctx.relationship_layers.get(level, []):
            src, tgt, rel_type, rel_level = relationship
            src_is_label = src in object_names
            tgt_is_label = tgt in object_names
            if not src_is_label and not tgt_is_label:
                continue

            other_name = tgt if src_is_label else src
            other_label = ctx.objects.get(other_name, {}).get("semantic_label", "unknown")
            if other_label == label:
                continue

            direction = "out" if src_is_label else "in"
            key = (level, rel_level, rel_type, direction, other_label)
            counts[key] += 1
            if len(examples[key]) < 3:
                examples[key].append(relationship)

    lines = [
        _phrase(
            language,
            it=f"Oggetti di classe '{label}': {len(object_names)} ({', '.join(sorted(object_names))}).",
            en=f"Objects of class '{label}': {len(object_names)} ({', '.join(sorted(object_names))}).",
        ),
    ]
    if not counts:
        lines.append(
            _phrase(
                language,
                it="Nessuna relazione con altre classi trovata.",
                en="No relationship with other classes found.",
            )
        )
        return "\n".join(lines)

    lines.append(_phrase(language, it="Relazioni con altre classi:", en="Relationships with other classes:"))
    for index, ((level, rel_level, rel_type, direction, other_label), count) in enumerate(
        sorted(counts.items(), key=lambda item: (item[0][0], item[0][4], item[0][2], item[0][3])),
        start=1,
    ):
        if index > limit:
            lines.append(
                _phrase(
                    language,
                    it=f"  ... {len(counts) - limit} gruppi non mostrati.",
                    en=f"  ... {len(counts) - limit} groups not shown.",
                )
            )
            break
        arrow = f"{label} -> {other_label}" if direction == "out" else f"{other_label} -> {label}"
        lines.append(f"  - {level}/{rel_level}: {arrow}, {rel_type} = {count}")
        for src, tgt, example_type, example_level in examples[
            (level, rel_level, rel_type, direction, other_label)
        ]:
            lines.append(f"      es. {src} --[{example_level}:{example_type}]--> {tgt}")
    return "\n".join(lines)


def _asks_for_support(text: str) -> bool:
    support_terms = (
        "support",
        "supports",
        "supported",
        "supporting",
        "supporta",
        "supportano",
        "supportato",
        "supportata",
        "supportati",
        "supportate",
        "sostiene",
        "sostengono",
        "sostenuto",
        "sostenuta",
        "sostenuti",
        "sostenute",
        "sorregge",
        "sorreggono",
        "sorretto",
        "sorretta",
        "sorretti",
        "sorrette",
        "regge",
        "reggono",
        "hold up",
        "holds up",
        "held up",
    )
    return any(term in text for term in support_terms)


def _is_passive_support_question(text: str) -> bool:
    passive_patterns = (
        "supportato da",
        "supportata da",
        "supportati da",
        "supportate da",
        "sostenuto da",
        "sostenuta da",
        "sostenuti da",
        "sostenute da",
        "sorretto da",
        "sorretta da",
        "sorretti da",
        "sorrette da",
        "supported by",
        "held up by",
    )
    return any(pattern in text for pattern in passive_patterns)


def _objects_with_semantic_label(ctx: SceneContext, label: str) -> list[str]:
    return sorted(
        name
        for name, obj in ctx.objects.items()
        if obj["semantic_label"] == label
    )


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


def _asks_for_material(text: str) -> bool:
    material_terms = (
        "materiale",
        "materiali",
        "materico",
        "materica",
        "material",
        "materials",
        "stone",
        "pietra",
        "marmo",
        "calcare",
        "brick",
        "bricks",
        "laterizio",
        "mattoni",
        "intonaco",
        "plaster",
        "stucco",
        "wood",
        "legno",
        "metal",
        "metallo",
        "glass",
        "vetro",
        "terracotta",
        "tile",
    )
    return any(term in text for term in material_terms)


def _asks_for_surface_roughness(text: str) -> bool:
    roughness_terms = (
        "roughness",
        "surface roughness",
        "rugosita",
        "rugos",
        "ruvid",
        "asperita",
        "asperit",
        "irregolarita superficiale",
        "superficie irregolare",
        "surface texture",
    )
    return any(term in text for term in roughness_terms)


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


def _format_material_summary(
    ctx: SceneContext,
    semantic_label: str | None = None,
    object_name: str | None = None,
    language: str = "it",
) -> str:
    query = _point_frame_for_query(ctx, semantic_label=semantic_label, object_name=object_name)
    if isinstance(query, str):
        return query
    name, df = query
    label = semantic_label
    if object_name and object_name in ctx.objects:
        label = ctx.objects[object_name]["semantic_label"]
    return format_point_material_summary(
        name,
        df,
        semantic_label=label,
        language=language,
    )


def _format_surface_roughness_summary(
    ctx: SceneContext,
    semantic_label: str | None = None,
    object_name: str | None = None,
    language: str = "it",
) -> str:
    query = _point_frame_for_query(ctx, semantic_label=semantic_label, object_name=object_name)
    if isinstance(query, str):
        return query
    name, df = query
    return format_roughness_summary(name, df, language=language)


def _format_color_summary(
    ctx: SceneContext,
    semantic_label: str | None = None,
    object_name: str | None = None,
    language: str = "it",
) -> str:
    query = _point_frame_for_query(ctx, semantic_label=semantic_label, object_name=object_name)
    if isinstance(query, str):
        return query
    name, df = query
    return format_point_rgb_summary(name, df, language=language)


def _point_frame_for_query(
    ctx: SceneContext,
    semantic_label: str | None = None,
    object_name: str | None = None,
) -> tuple[str, object] | str:
    if object_name:
        if object_name not in ctx.objects:
            return f"Oggetto '{object_name}' non trovato."
        return object_name, ctx.objects[object_name]["points"]

    if semantic_label:
        frames = [
            obj["points"] for obj in ctx.objects.values()
            if obj["semantic_label"] == semantic_label
        ]
        if not frames:
            return f"Non sono stati trovati oggetti di classe '{semantic_label}'."
        import pandas as pd

        return semantic_label, pd.concat(frames, ignore_index=True)

    return "scena", ctx.df


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
    labels = _extract_semantic_labels(text)
    return labels[0] if labels else None


def _extract_semantic_labels(text: str) -> list[str]:
    matches: list[tuple[int, int, str]] = []
    for word, label in _SEMANTIC_ALIASES:
        for match in re.finditer(rf"\b{re.escape(word)}\b", text):
            matches.append((match.start(), -(match.end() - match.start()), label))

    labels: list[str] = []
    occupied: set[int] = set()
    for start, negative_length, label in sorted(matches):
        length = -negative_length
        span = set(range(start, start + length))
        if occupied & span:
            continue
        occupied.update(span)
        labels.append(label)
    return labels


def _has_rgb(df) -> bool:
    return has_rgb(df)


def _xy_area(bounds: dict) -> float:
    dims = bounds["max"][:2] - bounds["min"][:2]
    return float(max(dims[0], 0.0) * max(dims[1], 0.0))
