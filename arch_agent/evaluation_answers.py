from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Callable
import unicodedata

from .evaluation_prompts import PROMPT_EXAMPLES
from .pipeline.relationships import (
    RELATIONSHIP_LAYER_NAMES,
    RELATIONSHIP_LAYER_ORDER,
    architectural_role,
)

if TYPE_CHECKING:
    from .pipeline.pipeline import SceneContext


AnswerBuilder = Callable[["SceneContext"], str]
Relationship = tuple[str, str, str, str]


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    without_accents = "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )
    return " ".join(without_accents.split())


_PROMPT_ID_BY_TEXT = {
    _normalize_text(example["prompt"]): example["id"]
    for example in PROMPT_EXAMPLES
}


def answer_evaluation_prompt(ctx: "SceneContext", user_input: str) -> str | None:
    text = _normalize_text(user_input)
    language = _response_language(user_input)
    prompt_id = _PROMPT_ID_BY_TEXT.get(text) or _infer_prompt_id(text)
    if prompt_id is None:
        return None

    builders: dict[int, AnswerBuilder] = {
        1: _answer_scene_summary,
        2: _answer_dominant_element,
        3: _answer_high_confidence_elements,
        4: _answer_inside_outside,
        5: _answer_boundaries,
        6: _answer_organizing_elements,
        7: _answer_adjacencies,
        8: _answer_above_below,
        9: _answer_intersections,
        10: _answer_supports,
        11: _answer_construction_systems,
        12: _answer_bearing_vs_non_bearing,
        13: _answer_structural_function,
        14: _answer_circulation_access,
        15: _answer_hierarchy,
        16: _answer_evident_spatial_relations,
        17: _answer_ambiguities,
        18: _answer_observation_inference_check,
        19: _answer_relation_quality_check,
        20: _answer_typology,
    }
    answer = builders[prompt_id](ctx)
    if language == "en":
        return _translate_grounded_answer_to_english(answer)
    return answer


def _infer_prompt_id(text: str) -> int | None:
    if _has_any(
        text,
        "elemento dominante",
        "oggetto dominante",
        "elemento piu importante",
        "elemento più importante",
        "dominant element",
        "most important element",
    ):
        return 2

    if _has_any(text, "interna", "interno", "inside", "internal") and _has_any(
        text,
        "esterna",
        "esterno",
        "mista",
        "mixed",
        "outside",
        "external",
    ):
        return 4

    if _has_any(text, "confini", "confine", "boundar", "limiti", "limite"):
        return 5

    if _has_any(text, "delimit", "contain", "organize", "organise", "conten", "organizz") and _has_any(
        text,
        "spazio",
        "space",
        "scena",
        "scene",
        "elementi",
        "elements",
        "oggetti",
        "objects",
    ):
        return 6

    if _has_any(text, "adiacent", "adjacent"):
        return 7

    if _has_any(text, "sopra", "sotto", "above", "below"):
        return 8

    if _has_any(text, "intersec", "sovrapp", "overlap", "intersect"):
        return 9

    if _has_any(text, "support", "sosteng", "sostiene", "sosten", "sorregg") and _has_any(
        text,
        "quali elementi",
        "quali oggetti",
        "elementi sembrano",
        "oggetti sembrano",
    ):
        return 10

    if _has_any(text, "sistema costruttivo", "sistemi costruttivi"):
        return 11

    if _has_any(text, "portanti", "portante", "non portanti", "non portante", "load-bearing"):
        return 12

    if _has_any(text, "funzione strutturale", "funzioni strutturali"):
        return 13

    if _has_any(text, "distributiv", "passaggio", "accesso", "circolazione"):
        return 14

    if _has_any(text, "gerarchia", "principali", "secondari"):
        return 15

    if _has_any(text, "relazioni spaziali", "spatial relationships"):
        return 16

    if _has_any(text, "ambigui", "ambigu", "difficili da interpretare"):
        return 17

    if _has_any(text, "tipologia", "tipologica", "etichetta tipologica"):
        return 20

    return None


def _has_any(text: str, *patterns: str) -> bool:
    return any(pattern in text for pattern in patterns)


def _response_language(text: str) -> str:
    normalized = _normalize_text(text)
    english_markers = (
        "what",
        "which",
        "how",
        "does",
        "do ",
        "is ",
        "are ",
        "inside",
        "outside",
        "mixed",
        "boundary",
        "boundaries",
        "delimit",
        "load-bearing",
        "typology",
        "support",
        "relationship",
    )
    italian_markers = (
        "quale",
        "quali",
        "quante",
        "quanti",
        "cosa",
        "scena",
        "intern",
        "estern",
        "mista",
        "confini",
        "delimit",
        "portant",
        "tipologia",
        "supporta",
        "relazioni",
    )
    english_score = sum(marker in normalized for marker in english_markers)
    italian_score = sum(marker in normalized for marker in italian_markers)
    return "en" if english_score > italian_score else "it"


def _translate_grounded_answer_to_english(answer: str) -> str:
    replacements = (
        ("Osservato dai dati:", "Observed data:"),
        ("Relazioni usate:", "Relationships used:"),
        ("Inferenza:", "Inference:"),
        ("Confidenza:", "Confidence:"),
        ("Indizi presenti:", "Observed cues:"),
        ("Relazioni considerate in cascata", "Relationships considered in cascade"),
        ("La scena sembra coperta o semi-interna, ma i limiti laterali sono incompleti.", "The scene appears covered or semi-internal, but the lateral boundaries are incomplete."),
        ("La scena è probabilmente interna o coperta.", "The scene is probably internal or covered."),
        ("La distinzione interno/esterno resta ambigua dai soli oggetti disponibili.", "The internal/external distinction remains ambiguous from the available objects alone."),
        ("Limiti principali", "Main boundaries"),
        ("Colonne perimetrali candidate", "Candidate perimeter columns"),
        ("Colonne interne / organizzatrici", "Internal / organizing columns"),
        ("Elementi distributivi", "Circulation elements"),
        ("Ruoli:", "Roles:"),
        ("Potenzialmente portanti", "Potentially load-bearing"),
        ("Superfici di appoggio", "Support surfaces"),
        ("Non portanti o non determinati", "Non-load-bearing or undetermined"),
        ("Etichetta tipologica sintetica", "Synthetic typological label"),
        ("spazio coperto colonnato / portico o padiglione", "covered colonnaded space / portico or pavilion"),
        ("sono presenti molte colonne, un piano di base e una copertura.", "many columns, a base plane, and a roof are present."),
        ("media", "medium"),
        ("alta", "high"),
        ("bassa", "low"),
        ("nessuno", "none"),
        ("nessuna", "none"),
        ("floor come piano inferiore", "floor as lower plane"),
        ("roof/vault come copertura superiore", "roof/vault as upper cover"),
        ("wall come possibile limite laterale", "wall as possible lateral boundary"),
        ("column associate a copertura", "columns associated with a cover"),
    )
    translated = answer
    for old, new in replacements:
        translated = translated.replace(old, new)
    return translated


def _grounded(
    observed: str,
    relations: str,
    inference: str,
    confidence: str,
) -> str:
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


def _answer_scene_summary(ctx: "SceneContext") -> str:
    return _grounded(
        observed="\n".join([
            _inventory_summary(ctx),
            _relationship_layer_summary(ctx),
            _top_metric_line(ctx, "point_count", "Oggetto più campionato"),
        ]),
        relations=(
            "Cascata L1->L2->L3 usata solo come sintesi quantitativa; "
            "la descrizione degli elementi deriva dalle classi semantiche."
        ),
        inference=_scene_type_inference(ctx),
        confidence=(
            "media-alta se le classi principali sono ben rappresentate; "
            "media se pochi oggetti dominano la scena o mancano L2/L3."
        ),
    )


def _answer_dominant_element(ctx: "SceneContext") -> str:
    by_points = _top_objects(ctx, "point_count", limit=3)
    by_volume = _top_objects(ctx, "volume", limit=3)
    by_degree = _top_by_degree(ctx, limit=3)
    dominant = _dominant_candidates(by_points, by_volume, by_degree)

    observed_lines = [
        _format_rank("Ranking per numero di punti", by_points, value_suffix=" punti"),
        _format_rank("Ranking per volume AABB", by_volume, precision=3),
        _format_rank("Ranking per grado relazionale", by_degree, value_suffix=" relazioni"),
    ]
    if dominant:
        inference = (
            "L'elemento dominante è probabilmente "
            + ", ".join(dominant)
            + ": compare ai primi posti in più metriche. "
            "La dominanza è geometrica/relazionale, non automaticamente tipologica."
        )
        confidence = "media: la dominanza dipende dalla metrica scelta."
    else:
        inference = (
            "Non emerge un unico elemento dominante: punto-count, volume e centralità "
            "non convergono sullo stesso oggetto."
        )
        confidence = "media-bassa: serve scegliere esplicitamente il criterio di dominanza."

    return _grounded(
        observed="\n\n".join(observed_lines),
        relations="L1/L2/L3 usate solo per il grado relazionale; punti e volume non usano relazioni.",
        inference=inference,
        confidence=confidence,
    )


def _answer_high_confidence_elements(ctx: "SceneContext") -> str:
    class_counts = _class_counts(ctx)
    top_classes = class_counts.most_common(5)
    robust_objects = _top_objects(ctx, "point_count", limit=5)
    labels = ", ".join(f"{label} ({count})" for label, count in top_classes) or "nessuna classe"

    return _grounded(
        observed="\n".join([
            f"Classi più ricorrenti: {labels}.",
            _format_rank("Oggetti con più punti", robust_objects, value_suffix=" punti"),
        ]),
        relations=(
            "Nessuna relazione necessaria per la confidenza di riconoscimento; "
            "la stima usa label semantiche e densità/campionamento degli oggetti."
        ),
        inference=(
            "Gli elementi identificabili con maggiore sicurezza sono quelli con label ripetute "
            "o molti punti. Le classi con un solo frammento piccolo vanno considerate più incerte."
        ),
        confidence="media: la confidenza reale dipende anche dalla qualità della segmentazione.",
    )


def _answer_inside_outside(ctx: "SceneContext") -> str:
    class_counts = _class_counts(ctx)
    labels = set(class_counts)
    cues = []
    if "floor" in labels:
        cues.append("floor come piano inferiore")
    if "roof" in labels or "vault" in labels:
        cues.append("roof/vault come copertura superiore")
    if "wall" in labels:
        cues.append("wall come possibile limite laterale")
    if "column" in labels and ("roof" in labels or "vault" in labels):
        cues.append("column associate a copertura")

    if "floor" in labels and ({"roof", "vault"} & labels) and class_counts["wall"] >= 2:
        inference = "La scena è probabilmente interna o coperta."
        confidence = "media-alta: floor, wall e copertura sono presenti."
    elif "floor" in labels and ({"roof", "vault"} & labels):
        inference = "La scena sembra coperta o semi-interna, ma i limiti laterali sono incompleti."
        confidence = "media: manca una chiusura laterale completa."
    elif "wall" in labels and "floor" in labels:
        inference = "La scena potrebbe essere interna o di facciata, ma la copertura non è esplicita."
        confidence = "media-bassa: evidenza parziale."
    else:
        inference = "La distinzione interno/esterno resta ambigua dai soli oggetti disponibili."
        confidence = "bassa: mancano indizi architettonici completi."

    return _grounded(
        observed="Indizi presenti: " + (", ".join(cues) if cues else "nessun indizio forte."),
        relations=(
            "Relazioni considerate in cascata: L1 per sopra/sotto e adiacenze, "
            "L2 solo se esistono supporti coerenti."
        ),
        inference=inference,
        confidence=confidence,
    )


def _answer_boundaries(ctx: "SceneContext") -> str:
    boundaries = _objects_with_labels(ctx, {"floor", "wall", "roof", "vault"})
    groups = _group_names_by_label(ctx, boundaries)

    return _grounded(
        observed=_format_grouped_objects(groups, "Possibili confini rilevati"),
        relations=(
            "L1/geometric sopra-sotto e adiacenze possono indicare posizione dei confini; "
            "nessuna relazione di contenimento è definita nel grafo corrente."
        ),
        inference=(
            "Il floor può agire come limite inferiore, roof/vault come limite superiore, "
            "wall come limite laterale. Questa è una lettura architettonica dei ruoli, "
            "non una relazione 'inside/contains'."
        ),
        confidence="media: i confini sono plausibili se gli oggetti sono continui e ben segmentati.",
    )


def _answer_organizing_elements(ctx: "SceneContext") -> str:
    labels = _class_counts(ctx)
    delimiters = _objects_with_labels(ctx, {"floor", "wall", "roof", "vault"})
    outer_columns = _outer_column_names(ctx)
    inner_columns = [
        name for name in _objects_with_labels(ctx, {"column"})
        if name not in set(outer_columns)
    ]
    circulation = _objects_with_labels(ctx, {"stairs"})

    return _grounded(
        observed="\n".join([
            _format_grouped_objects(_group_names_by_label(ctx, delimiters), "Limiti principali"),
            _format_object_list("Colonne perimetrali candidate", outer_columns),
            _format_object_list("Colonne interne / organizzatrici", inner_columns),
            _format_object_list("Elementi distributivi", circulation),
            _role_summary(labels),
        ]),
        relations=(
            "L1/geometric per posizione planimetrica, adiacenza e sopra/sotto; "
            "colonne perimetrali stimate dal convex hull XY dei centroidi; "
            "L2/structural solo per eventuali supporti."
        ),
        inference=(
            "Floor, wall e roof/vault definiscono i limiti principali; le colonne "
            "più esterne possono contribuire al perimetro o alla scansione del bordo, "
            "mentre le colonne interne organizzano e/o supportano lo spazio. "
            "Non viene inferita una chiusura completa senza pareti continue."
        ),
        confidence=(
            "media: le colonne perimetrali sono stimate geometricamente dai centroidi "
            "e dipendono da segmentazione e completezza della scena."
        ),
    )


def _answer_adjacencies(ctx: "SceneContext") -> str:
    adjs = _unique_undirected(_relationships(ctx, level="L1", rel_type="adjacent_to"))
    examples = _format_relationship_examples(adjs, limit=20)

    return _grounded(
        observed=examples,
        relations="L1/geometric: adjacent_to. Le relazioni reciproche sono deduplicate nella lista.",
        inference=(
            "L'adiacenza indica vicinanza/contatto geometrico tra bounding box; "
            "non implica da sola supporto, appartenenza o funzione."
        ),
        confidence="alta per l'elenco L1; media per il significato architettonico.",
    )


def _answer_above_below(ctx: "SceneContext") -> str:
    above = _relationships(ctx, level="L1", rel_type="above")
    examples = _format_relationship_examples(above, limit=20)

    return _grounded(
        observed=examples,
        relations="L1/geometric: above/below. La direzione below è l'inverso di above.",
        inference=(
            "Le relazioni sopra/sotto descrivono ordine verticale. Non sono una prova "
            "di supporto strutturale se non compaiono anche relazioni L2 coerenti."
        ),
        confidence="alta per la geometria verticale; media-bassa per interpretazioni strutturali.",
    )


def _answer_intersections(ctx: "SceneContext") -> str:
    overlap_like = _overlap_candidates(ctx, limit=20)
    observed = (
        "Il grafo corrente non definisce relazioni esplicite 'intersects', 'overlaps', "
        "'inside' o 'contains'."
    )
    if overlap_like:
        observed += "\nCoppie con bounding box sovrapposte o a contatto:\n" + "\n".join(
            f"  - {a} / {b}" for a, b in overlap_like
        )

    return _grounded(
        observed=observed,
        relations="Controllo geometrico indiretto su bounding box; nessuna relazione L1 dedicata all'intersezione.",
        inference=(
            "Si possono segnalare contatti o sovrapposizioni di bounding box, ma non "
            "affermare una vera intersezione fisica senza una relazione o un test geometrico più fine."
        ),
        confidence="bassa per l'assenza di intersezioni; media per eventuali overlap AABB.",
    )


def _answer_supports(ctx: "SceneContext") -> str:
    supports = _relationships(ctx, level="L2", rel_type="supports")
    rests_on = _relationships(ctx, level="L2", rel_type="rests_on")

    return _grounded(
        observed="\n\n".join([
            _format_relationship_examples(supports, title="Supporti L2", limit=20),
            _format_relationship_examples(rests_on, title="Appoggi L2", limit=20),
        ]),
        relations="L2/structural: supports e rests_on, già filtrate da regole architettoniche di classe.",
        inference=(
            "Gli elementi che supportano sono solo quelli presenti come sorgente di 'supports'. "
            "Le relazioni L1 'above' non vengono trasformate automaticamente in supporto."
        ),
        confidence="media-alta se L2 non è vuoto; media se il supporto dipende da soglie di contatto.",
    )


def _answer_construction_systems(ctx: "SceneContext") -> str:
    structural = _objects_by_role(ctx, "structural")
    support_surface = _objects_by_role(ctx, "support_surface")
    ornamental = _objects_by_role(ctx, "ornamental")
    openings = _objects_by_role(ctx, "opening")
    supports = _relationships(ctx, level="L2", rel_type="supports")

    observed = "\n".join([
        _format_object_list("Sistema strutturale potenziale", structural),
        _format_object_list("Superfici di appoggio", support_surface),
        _format_object_list("Elementi ornamentali", ornamental),
        _format_object_list("Aperture", openings),
        _format_relationship_examples(supports, title="Connessioni L2 rilevate", limit=15),
    ])

    return _grounded(
        observed=observed,
        relations="L2 per sistema resistente; L3 per elementi parte-di/decorativi se presenti.",
        inference=(
            "Gli oggetti dello stesso sistema costruttivo sono raggruppati per ruolo "
            "architettonico e, quando disponibile, per relazioni L2/L3. Le ripetizioni "
            "di columns indicano un possibile sistema modulare."
        ),
        confidence="media: il sistema costruttivo è una sintesi, non una label osservata direttamente.",
    )


def _answer_bearing_vs_non_bearing(ctx: "SceneContext") -> str:
    labels = _class_counts(ctx)
    structural = _objects_by_role(ctx, "structural")
    support_surface = _objects_by_role(ctx, "support_surface")
    non_bearing = (
        _objects_by_role(ctx, "ornamental")
        + _objects_by_role(ctx, "opening")
        + _objects_by_role(ctx, "circulation")
        + _objects_by_role(ctx, "unknown")
    )

    return _grounded(
        observed="\n".join([
            _role_summary(labels),
            _format_object_list("Potenzialmente portanti", structural),
            _format_object_list("Superfici di appoggio", support_surface),
            _format_object_list("Non portanti o non determinati", non_bearing),
        ]),
        relations=(
            "L2/structural rafforza la lettura portante quando compaiono supports/rests_on; "
            "la classificazione base deriva dall'ontologia delle classi."
        ),
        inference=(
            "Arch, column, wall, vault e roof sono trattati come strutturali. "
            "Moldings e door_window sono non portanti; stairs e other non vanno considerati "
            "portanti senza evidenza aggiuntiva."
        ),
        confidence="media: la distinzione è semantica, non una verifica meccanica.",
    )


def _answer_structural_function(ctx: "SceneContext") -> str:
    structural = _objects_by_role(ctx, "structural")
    supports = _relationships(ctx, level="L2", rel_type="supports")

    return _grounded(
        observed="\n".join([
            _format_object_list("Elementi con ruolo strutturale", structural),
            _format_relationship_examples(supports, title="Supporti strutturali rilevati", limit=20),
        ]),
        relations="L2/structural per supporti; ruoli architettonici per la lista degli elementi strutturali.",
        inference=(
            "Gli elementi con funzione strutturale sono quelli dell'ontologia strutturale; "
            "una funzione portante effettiva è più solida quando compare una relazione L2."
        ),
        confidence="media-alta per i ruoli; media per la funzione effettiva se L2 è scarso.",
    )


def _answer_circulation_access(ctx: "SceneContext") -> str:
    stairs = _objects_with_labels(ctx, {"stairs"})
    openings = _objects_with_labels(ctx, {"door_window"})
    floors = _objects_with_labels(ctx, {"floor"})

    return _grounded(
        observed="\n".join([
            _format_object_list("Scale / distribuzione verticale", stairs),
            _format_object_list("Aperture / accessi potenziali", openings),
            _format_object_list("Floor / piano percorribile potenziale", floors),
        ]),
        relations="L3 se stairs is_placed_on floor o door_window is_opening_in wall; altrimenti solo ruoli semantici.",
        inference=(
            "Stairs indicano distribuzione verticale; door_window indica possibile accesso o apertura; "
            "floor può essere piano di percorrenza ma non definisce da solo un percorso."
        ),
        confidence="media se stairs o door_window sono presenti; bassa se resta solo floor.",
    )


def _answer_hierarchy(ctx: "SceneContext") -> str:
    main = _dominant_candidates(
        _top_objects(ctx, "point_count", limit=3),
        _top_objects(ctx, "volume", limit=3),
        _top_by_degree(ctx, limit=3),
    )
    secondary = (
        _objects_by_role(ctx, "ornamental")
        + _objects_by_role(ctx, "opening")
        + _objects_by_role(ctx, "unknown")
    )
    supports = _relationships(ctx, level="L2", rel_type="supports")

    observed = "\n".join([
        "Elementi principali candidati: " + (", ".join(main) if main else "non univoci"),
        _format_object_list("Elementi secondari candidati", secondary),
        _format_relationship_examples(supports, title="Gerarchia L2 disponibile", limit=15),
    ])

    if main or supports:
        inference = (
            "La scena mostra una gerarchia parziale: elementi strutturali e oggetti dominanti "
            "possono essere principali, mentre ornamentazioni/aperture/frammenti restano secondari."
        )
        confidence = "media: la gerarchia è supportata da metriche e relazioni, ma non da una tipologia completa."
    else:
        inference = "Non emerge una gerarchia chiara tra elementi principali e secondari."
        confidence = "bassa: mancano convergenza metrica e relazioni L2."

    return _grounded(
        observed=observed,
        relations="Metriche oggetto + L2/structural; L1 sopra/sotto non basta per definire gerarchia.",
        inference=inference,
        confidence=confidence,
    )


def _answer_evident_spatial_relations(ctx: "SceneContext") -> str:
    geometric = ctx.relationship_layers.get("L1", [])
    type_counts = Counter(rel_type for _, _, rel_type, _ in geometric)
    top_types = ", ".join(
        f"{rel_type}={count}" for rel_type, count in type_counts.most_common()
    ) or "nessuna relazione L1"

    return _grounded(
        observed=f"Distribuzione L1/geometric: {top_types}.",
        relations="L1/geometric: near, adjacent_to, above, below.",
        inference=(
            "Le relazioni spaziali più evidenti sono quelle con conteggio maggiore. "
            "Sono evidenze geometriche e non vanno lette automaticamente come struttura o funzione."
        ),
        confidence="alta per i conteggi; media per la loro interpretazione architettonica.",
    )


def _answer_ambiguities(ctx: "SceneContext") -> str:
    low_point_objects = _low_point_objects(ctx, limit=8)
    unknown = _objects_by_role(ctx, "unknown")
    l1_count = len(ctx.relationship_layers.get("L1", []))
    l2_count = len(ctx.relationship_layers.get("L2", []))
    l3_count = len(ctx.relationship_layers.get("L3", []))
    notes = []
    if unknown:
        notes.append("oggetti 'other' o ruolo unknown")
    if low_point_objects:
        notes.append("oggetti con pochi punti rispetto alla scena")
    if l1_count and not l2_count:
        notes.append("molte relazioni geometriche senza conferma strutturale L2")
    if not l3_count:
        notes.append("assenza di relazioni mereologiche L3")

    observed = "\n".join([
        _format_object_list("Oggetti unknown", unknown),
        _format_rank("Oggetti meno campionati", low_point_objects, value_suffix=" punti"),
        f"Relazioni per livello: L1={l1_count}, L2={l2_count}, L3={l3_count}.",
    ])

    return _grounded(
        observed=observed,
        relations="Confronto tra L1, L2 e L3 per individuare dove l'interpretazione è più debole.",
        inference=(
            "Ambiguità principali: "
            + (", ".join(notes) if notes else "nessuna ambiguità forte rilevata dai criteri automatici.")
        ),
        confidence="media: è un controllo automatico, non una revisione visiva della point cloud.",
    )


def _answer_observation_inference_check(ctx: "SceneContext") -> str:
    return _grounded(
        observed=(
            "Questa domanda valuta una risposta del modello, ma nel contesto corrente "
            "non è presente una risposta precedente da analizzare."
        ),
        relations="Nessuna relazione di scena usata direttamente.",
        inference=(
            "La risposta corretta deve separare dati osservati, relazioni L1/L2/L3, "
            "interpretazioni architettoniche e confidenza. Il formato attuale dell'agente "
            "impone proprio queste quattro sezioni."
        ),
        confidence="alta come criterio di valutazione; non valutabile su una risposta assente.",
    )


def _answer_relation_quality_check(ctx: "SceneContext") -> str:
    l1 = len(ctx.relationship_layers.get("L1", []))
    l2 = len(ctx.relationship_layers.get("L2", []))
    l3 = len(ctx.relationship_layers.get("L3", []))

    return _grounded(
        observed=f"Relazioni disponibili: L1={l1}, L2={l2}, L3={l3}.",
        relations="Controllo del bilanciamento L1/L2/L3, non di una risposta testuale precedente.",
        inference=(
            "Senza una risposta del modello da confrontare non posso dire se quella risposta "
            "sia troppo generica. Posso però segnalare il rischio: se L2/L3 sono pochi o assenti, "
            "le conclusioni strutturali e tipologiche devono restare caute."
        ),
        confidence="alta sul criterio; non valutabile sulla qualità di una risposta assente.",
    )


def _answer_typology(ctx: "SceneContext") -> str:
    label, reason, confidence = _typology_label(ctx)
    return _grounded(
        observed="\n".join([
            _inventory_summary(ctx),
            _relationship_layer_summary(ctx),
        ]),
        relations="Cascata L1->L2->L3 usata come supporto; la tipologia resta una inferenza.",
        inference=f"Etichetta tipologica sintetica: {label}. Motivo: {reason}",
        confidence=confidence,
    )


def _class_counts(ctx: "SceneContext") -> Counter[str]:
    return Counter(obj["semantic_label"] for obj in ctx.objects.values())


def _role_counts(ctx: "SceneContext") -> Counter[str]:
    counts: Counter[str] = Counter()
    for label, count in _class_counts(ctx).items():
        counts[architectural_role(label)] += count
    return counts


def _inventory_summary(ctx: "SceneContext") -> str:
    class_counts = _class_counts(ctx)
    role_counts = _role_counts(ctx)
    class_text = ", ".join(
        f"{label}={count}" for label, count in sorted(class_counts.items())
    ) or "nessuna classe"
    role_text = ", ".join(
        f"{role}={count}" for role, count in sorted(role_counts.items())
    ) or "nessun ruolo"
    return f"Oggetti totali: {len(ctx.objects)}. Classi: {class_text}. Ruoli: {role_text}."


def _relationship_layer_summary(ctx: "SceneContext") -> str:
    parts = []
    for level in RELATIONSHIP_LAYER_ORDER:
        layer_name = RELATIONSHIP_LAYER_NAMES.get(level, level)
        parts.append(f"{level}/{layer_name}={len(ctx.relationship_layers.get(level, []))}")
    return "Relazioni: " + ", ".join(parts) + "."


def _role_summary(class_counts: Counter[str]) -> str:
    role_counts: Counter[str] = Counter()
    for label, count in class_counts.items():
        role_counts[architectural_role(label)] += count
    return "Ruoli: " + (
        ", ".join(f"{role}={count}" for role, count in sorted(role_counts.items()))
        if role_counts
        else "nessun ruolo"
    )


def _objects_with_labels(ctx: "SceneContext", labels: set[str]) -> list[str]:
    return [
        name for name, obj in sorted(ctx.objects.items())
        if obj["semantic_label"] in labels
    ]


def _outer_column_names(ctx: "SceneContext") -> list[str]:
    columns = _objects_with_labels(ctx, {"column"})
    if len(columns) <= 3:
        return columns

    points = [
        (
            float(ctx.objects[name]["centroid"][0]),
            float(ctx.objects[name]["centroid"][1]),
            name,
        )
        for name in columns
    ]
    hull_names = _convex_hull_point_names(points)
    return sorted(hull_names)


def _convex_hull_point_names(points: list[tuple[float, float, str]]) -> list[str]:
    unique: dict[tuple[float, float], str] = {}
    for x, y, name in points:
        unique.setdefault((x, y), name)

    sorted_points = sorted((x, y, name) for (x, y), name in unique.items())
    if len(sorted_points) <= 3:
        return [name for _, _, name in sorted_points]

    lower: list[tuple[float, float, str]] = []
    for point in sorted_points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float, str]] = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    return [name for _, _, name in hull]


def _cross(
    origin: tuple[float, float, str],
    a: tuple[float, float, str],
    b: tuple[float, float, str],
) -> float:
    return (
        (a[0] - origin[0]) * (b[1] - origin[1])
        - (a[1] - origin[1]) * (b[0] - origin[0])
    )


def _objects_by_role(ctx: "SceneContext", role: str) -> list[str]:
    return [
        name for name, obj in sorted(ctx.objects.items())
        if architectural_role(obj["semantic_label"]) == role
    ]


def _group_names_by_label(ctx: "SceneContext", names: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name in names:
        label = ctx.objects[name]["semantic_label"]
        grouped.setdefault(label, []).append(name)
    return grouped


def _format_grouped_objects(groups: dict[str, list[str]], title: str) -> str:
    if not groups:
        return f"{title}: nessun oggetto rilevato."
    lines = [f"{title}:"]
    for label, names in sorted(groups.items()):
        lines.append(f"  - {label}: {', '.join(names)}")
    return "\n".join(lines)


def _format_object_list(title: str, names: list[str]) -> str:
    if not names:
        return f"{title}: nessuno."
    return f"{title}: {', '.join(names)}."


def _relationships(
    ctx: "SceneContext",
    level: str | None = None,
    rel_type: str | None = None,
) -> list[Relationship]:
    if level is None:
        relationships = list(ctx.relationships)
    else:
        relationships = list(ctx.relationship_layers.get(level, []))
    if rel_type is not None:
        relationships = [rel for rel in relationships if rel[2] == rel_type]
    return relationships


def _unique_undirected(relationships: list[Relationship]) -> list[Relationship]:
    deduped: list[Relationship] = []
    seen: set[tuple[str, str, str]] = set()
    for src, tgt, rel_type, rel_level in relationships:
        key = tuple(sorted((src, tgt))) + (rel_type,)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((src, tgt, rel_type, rel_level))
    return deduped


def _format_relationship_examples(
    relationships: list[Relationship],
    title: str = "Relazioni",
    limit: int = 20,
) -> str:
    if not relationships:
        return f"{title}: nessuna relazione trovata."
    shown = relationships[:limit]
    lines = [f"{title}: {len(relationships)} totali; prime {len(shown)}:"]
    for src, tgt, rel_type, rel_level in shown:
        lines.append(f"  - {src} --[{rel_level}:{rel_type}]--> {tgt}")
    if len(relationships) > limit:
        lines.append(f"  ... {len(relationships) - limit} non mostrate.")
    return "\n".join(lines)


def _top_objects(
    ctx: "SceneContext",
    metric: str,
    limit: int = 5,
) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for name, obj in ctx.objects.items():
        if metric == "point_count":
            value = float(obj.get("point_count", 0))
        else:
            value = float(ctx.features.get(name, {}).get(metric, 0.0))
        rows.append((name, obj["semantic_label"], value))
    rows.sort(key=lambda row: row[2], reverse=True)
    return rows[:limit]


def _top_by_degree(ctx: "SceneContext", limit: int = 5) -> list[tuple[str, str, float]]:
    degree: Counter[str] = Counter()
    for src, tgt, _, _ in ctx.relationships:
        degree[src] += 1
        degree[tgt] += 1
    rows = [
        (name, obj["semantic_label"], float(degree[name]))
        for name, obj in ctx.objects.items()
    ]
    rows.sort(key=lambda row: row[2], reverse=True)
    return rows[:limit]


def _top_metric_line(ctx: "SceneContext", metric: str, title: str) -> str:
    rows = _top_objects(ctx, metric, limit=1)
    if not rows:
        return f"{title}: nessuno."
    name, label, value = rows[0]
    if metric == "point_count":
        return f"{title}: {name} ({label}), {int(value):,} punti."
    return f"{title}: {name} ({label}), {value:.3f}."


def _format_rank(
    title: str,
    rows: list[tuple[str, str, float]],
    value_suffix: str = "",
    precision: int = 0,
) -> str:
    if not rows:
        return f"{title}: nessun dato."
    lines = [f"{title}:"]
    for name, label, value in rows:
        if precision == 0:
            value_text = f"{int(value):,}"
        else:
            value_text = f"{value:.{precision}f}"
        lines.append(f"  - {name} ({label}): {value_text}{value_suffix}")
    return "\n".join(lines)


def _dominant_candidates(
    by_points: list[tuple[str, str, float]],
    by_volume: list[tuple[str, str, float]],
    by_degree: list[tuple[str, str, float]],
) -> list[str]:
    votes: Counter[str] = Counter()
    for ranking in (by_points, by_volume, by_degree):
        for rank, (name, _, _) in enumerate(ranking[:3], start=1):
            votes[name] += 4 - rank
    if not votes:
        return []
    max_vote = max(votes.values())
    if max_vote < 4:
        return []
    return [name for name, vote in votes.items() if vote == max_vote]


def _low_point_objects(ctx: "SceneContext", limit: int = 8) -> list[tuple[str, str, float]]:
    rows = [
        (name, obj["semantic_label"], float(obj.get("point_count", 0)))
        for name, obj in ctx.objects.items()
    ]
    rows.sort(key=lambda row: row[2])
    return rows[:limit]


def _overlap_candidates(ctx: "SceneContext", limit: int = 20) -> list[tuple[str, str]]:
    names = sorted(ctx.objects)
    candidates: list[tuple[str, str]] = []
    for index, name_a in enumerate(names):
        for name_b in names[index + 1:]:
            bounds_a = ctx.objects[name_a]["bounds"]
            bounds_b = ctx.objects[name_b]["bounds"]
            if _bounds_overlap_or_touch(bounds_a, bounds_b):
                candidates.append((name_a, name_b))
                if len(candidates) >= limit:
                    return candidates
    return candidates


def _bounds_overlap_or_touch(bounds_a: dict, bounds_b: dict) -> bool:
    for axis in range(3):
        if bounds_a["max"][axis] < bounds_b["min"][axis]:
            return False
        if bounds_b["max"][axis] < bounds_a["min"][axis]:
            return False
    return True


def _scene_type_inference(ctx: "SceneContext") -> str:
    label, reason, _ = _typology_label(ctx)
    return f"Tipologia probabile: {label}. {reason}"


def _typology_label(ctx: "SceneContext") -> tuple[str, str, str]:
    counts = _class_counts(ctx)
    has_cover = counts["roof"] > 0 or counts["vault"] > 0
    has_floor = counts["floor"] > 0
    has_wall = counts["wall"] > 0
    has_many_columns = counts["column"] >= 4
    has_arch_or_vault = counts["arch"] > 0 or counts["vault"] > 0
    has_stairs = counts["stairs"] > 0

    if has_many_columns and has_cover and has_floor:
        return (
            "spazio coperto colonnato / portico o padiglione",
            "sono presenti molte colonne, un piano di base e una copertura.",
            "media: la lettura tipologica è plausibile ma non distingue portico, aula o padiglione.",
        )
    if has_wall and has_floor and has_cover:
        return (
            "spazio interno coperto",
            "floor, wall e copertura compongono un envelope architettonico minimo.",
            "media-alta: gli indizi principali sono presenti, ma la tipologia specifica resta aperta.",
        )
    if has_arch_or_vault:
        return (
            "spazio voltato o sistema ad archi",
            "arch/vault sono elementi tipologicamente caratterizzanti.",
            "media: servono continuità geometrica e relazioni L2/L3 per maggiore certezza.",
        )
    if has_stairs:
        return (
            "spazio di distribuzione verticale",
            "stairs è l'elemento funzionale più specifico rilevato.",
            "media: dipende dal rapporto con floor, wall e aperture.",
        )
    return (
        "scena architettonica parziale",
        "le classi presenti non bastano per una tipologia più specifica.",
        "bassa-media: l'etichetta resta descrittiva.",
    )
