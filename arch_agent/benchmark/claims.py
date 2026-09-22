"""Turn a free-text answer into a normalised set of claims.

Scoring prose directly is what made the earlier validator unreliable: a regex
over the whole answer read "8 door_window" as "8 objects in total" and treated
"roof: 0" as a claim that roof is present. The fix is to stop comparing prose
with facts. An answer is first reduced to the claims it actually makes, and
only those claims are compared with the reference.

Rule-based extraction works here because the domain is closed: ten semantic
classes with known aliases, a fixed relation vocabulary, and object ids that
match a single pattern. An LLM extractor would add a second, unmeasured source
of error.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

OBJECT_ID_RE = re.compile(r"\b([a-z_]+_\d+)\b")
# Answers are routinely formatted as markdown lists - "- **column**: 6" - so
# the separator between a class name and its number may carry punctuation.
_MARKUP = r"[\s*_`~:=]*"
# A list bullet ends the previous item, so it must not be crossed when looking
# backwards: in "- **wall**: 11 - **column**: 6" the 11 belongs to wall.
_NUMBER_BEFORE_RE = re.compile(
    r"(\d+)\s*(?:oggetti|oggetto|objects?|istanze|istanza|instances?|elementi|elemento|elements?)?"
    + _MARKUP + r"$"
)
# No bare "e" connector: normalisation strips accents, so Italian "e" (and)
# and "e" (is) are indistinguishable, and "6 colonne e 11 muri" would give the
# columns the walls' number.
_NUMBER_AFTER_RE = re.compile(r"^" + _MARKUP + r"(?:sono|is|are)?" + _MARKUP + r"(\d+)")

CLASS_ALIASES: dict[str, tuple[str, ...]] = {
    "arch": ("arch", "arches", "arco", "archi"),
    "column": ("column", "columns", "colonna", "colonne"),
    "moldings": ("moldings", "molding", "modanatura", "modanature", "cornice", "cornici", "lesena", "lesene"),
    # Aliases are matched longest-first so a plural is not shadowed by its stem.
    "floor": ("floor", "floors", "pavimento", "pavimenti"),
    "door_window": ("door_windows", "door_window", "door windows", "door window",
                    "porta", "porte", "finestra", "finestre", "apertura", "aperture"),
    "wall": ("wall", "walls", "muro", "muri", "parete", "pareti"),
    "stairs": ("stairs", "staircase", "scala", "scale", "gradini"),
    "vault": ("vault", "vaults", "volta", "volte"),
    "roof": ("roof", "roofs", "tetto", "tetti", "copertura"),
    "other": ("other",),
}

RELATION_ALIASES: dict[str, tuple[str, ...]] = {
    "supports": ("supports", "support", "supporta", "supportano", "sostiene", "sostengono"),
    "rests_on": ("rests_on", "rests on", "poggia", "poggiano", "appoggia", "appoggiano"),
    "above": ("above", "sopra", "al di sopra"),
    "below": ("below", "sotto", "al di sotto"),
    "adjacent_to": ("adjacent_to", "adjacent to", "adiacente", "adiacenti"),
    "near": ("near", "vicino", "vicini", "prossimo"),
    "part_of": ("part_of", "part of", "parte di"),
    "has_part": ("has_part", "has part"),
    "is_opening_in": ("is_opening_in", "opening in", "apertura in", "apertura nel", "apertura nella"),
    "is_ornament_of": ("is_ornament_of", "ornament of", "ornamento di"),
    "is_attached_to": ("is_attached_to", "attached to", "attaccato", "addossato"),
    "is_connected_to": ("is_connected_to", "connected to", "collegato", "collega"),
    "is_placed_on": ("is_placed_on", "placed on", "posato", "collocato"),
    "is_rib_of": ("is_rib_of", "rib of", "nervatura"),
}

ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "structural": ("structural", "strutturale", "strutturali", "portante", "portanti"),
    "ornamental": ("ornamental", "ornamentale", "ornamentali", "decorativo", "decorativa", "decorativi"),
    "support_surface": ("support_surface", "support surface", "superficie di appoggio", "piano di calpestio"),
    "opening": ("opening", "apertura", "aperture"),
    "circulation": ("circulation", "circolazione", "collegamento verticale"),
}

DEFAULT_NEGATION_MARKERS = (
    "non e presente", "non sono presenti", "non ci sono", "non risulta",
    "nessun", "nessuna", "assente", "assenti", "privo", "manca", "mancano",
    "is not present", "are not present", "no objects", "none", "absent",
    "not found", "there are no", "there is no",
)
DEFAULT_TOTAL_MARKERS = (
    "totale", "in tutto", "complessivamente", "in totale",
    "total", "in total", "overall", "altogether",
)
DEFAULT_ABSTENTION_MARKERS = (
    "non disponibile", "non e disponibile", "non risulta", "non specificato",
    "nessuna informazione", "non presente nel grafo", "dato assente",
    "non posso determinare", "not available", "no information",
    "not specified", "cannot determine", "unknown", "no data",
)

# Marker lists cannot keep up with conjugation: the reference has "non ci sono"
# but answers also write "non ci siano", "non vi sono", "non risultano". This
# covers the construction instead of enumerating its forms.
NEGATION_RE = re.compile(
    r"\b(?:non|no)\s+(?:ci\s+|vi\s+|ne\s+)?"
    r"(?:sono|siano|e|sia|era|erano|risulta|risultano|esiste|esistono|"
    r"compare|compaiono|presenta|presentano|contiene|contengono)\b"
    r"|\b(?:there\s+(?:is|are)\s+no|does\s+not\s+(?:contain|include)|"
    r"are\s+not\s+present|is\s+not\s+present)\b"
)

_AFFIRM = ("si", "yes", "esatto", "corretto", "true", "confermo", "sono", "supportano")
_DENY = ("no", "non", "not", "false", "nessun", "assente")


def normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace. Keeps digits and _ intact."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", folded)


@dataclass
class Claims:
    """What an answer asserts, reduced to comparable facts."""

    text: str = ""
    total_count: int | None = None
    class_counts: dict[str, int] = field(default_factory=dict)
    classes_affirmed: set[str] = field(default_factory=set)
    classes_negated: set[str] = field(default_factory=set)
    object_ids: set[str] = field(default_factory=set)
    relations: set[tuple[str, str, str]] = field(default_factory=set)
    roles: set[str] = field(default_factory=set)
    roles_negated: set[str] = field(default_factory=set)
    numbers: list[int] = field(default_factory=list)
    coordinates: list[tuple[float, float, float]] = field(default_factory=list)
    polarity: bool | None = None
    abstains: bool = False

    def mentions(self, value: str) -> bool:
        """True when the answer contains `value`, compared after normalisation."""
        needle = normalize(str(value)).strip()
        return bool(needle) and needle in self.text


def extract(answer: str | None, policy: dict | None = None) -> Claims:
    """Reduce one answer to the claims it makes."""
    if not answer:
        return Claims()

    policy = policy or {}
    text = normalize(answer)
    negation = _markers(policy, "negation_markers", DEFAULT_NEGATION_MARKERS)
    totals = _markers(policy, "total_markers", DEFAULT_TOTAL_MARKERS)
    abstention = _markers(policy, "abstention_markers", DEFAULT_ABSTENTION_MARKERS)

    counts = _class_counts(text)
    affirmed, negated = _class_polarity(text, counts, negation)

    return Claims(
        text=text,
        total_count=_total_count(text, totals),
        class_counts=counts,
        classes_affirmed=affirmed,
        classes_negated=negated,
        object_ids=set(OBJECT_ID_RE.findall(text)),
        relations=_relations(text, negation),
        roles=_roles(text, negation, negated=False),
        roles_negated=_roles(text, negation, negated=True),
        numbers=[int(n) for n in re.findall(r"\b\d+\b", text)],
        coordinates=_coordinates(text),
        polarity=_polarity(text),
        abstains=any(marker in text for marker in abstention),
    )


def _markers(policy: dict, prefix: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    """Reference markers plus the built-in ones.

    The reference list is deliberately not treated as exhaustive: it records
    "assente" but not "assenti", and "non e presente" but not "non sono
    presenti", so replacing the defaults with it would miss ordinary phrasings
    and mark correct answers wrong.
    """
    found = tuple(policy.get(f"{prefix}_it", ())) + tuple(policy.get(f"{prefix}_en", ()))
    return tuple(dict.fromkeys(tuple(normalize(m) for m in found) + fallback))


def _sorted_aliases(aliases: tuple[str, ...]) -> tuple[str, ...]:
    """Longest first: "door_windows" must be tried before "door_window"."""
    return tuple(sorted(aliases, key=len, reverse=True))


def _any_alias(text: str, aliases: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases)


def _roles(text: str, negation: tuple[str, ...], negated: bool) -> set[str]:
    """Roles the answer asserts, split by polarity.

    "ornamentali, non strutturali" states one role and denies another; reading
    both as asserted would fail a correct answer.
    """
    found: set[str] = set()
    for role, aliases in ROLE_ALIASES.items():
        positions = [
            m.start()
            for alias in aliases
            for m in re.finditer(rf"\b{re.escape(alias)}\b", text)
        ]
        if not positions:
            continue
        denied = all(
            _negated_at(text, pos, negation, window=24) or _directly_negated(text, pos)
            for pos in positions
        )
        if denied == negated:
            found.add(role)
    return found


_BARE_NEGATOR_RE = re.compile(r"\b(?:non|not|ne|never|mai)\s+(?:\w+\s+){0,2}$")


def _directly_negated(text: str, position: int, window: int = 22) -> bool:
    """Catch a bare negator just before the word, as in "non strutturali".

    The reference marker list holds whole phrases ("non e presente", "non ci
    sono"); a plain "non" immediately before a role or class word is the other
    common way an answer denies something.
    """
    return bool(_BARE_NEGATOR_RE.search(text[max(0, position - window):position]))


def _class_counts(text: str) -> dict[str, int]:
    """Bind each number to the class name nearest to it.

    The previous validator compared numbers against the scene total whatever
    they were attached to, so "8 door_window" was read as a claim of 8 objects
    in the scene. A number counts for a class only when it sits next to that
    class name.
    """
    counts: dict[str, int] = {}
    for label, aliases in CLASS_ALIASES.items():
        for alias in _sorted_aliases(aliases):
            for match in re.finditer(rf"\b{re.escape(alias)}\b", text):
                value = _number_beside(text, match.start(), match.end())
                if value is not None and label not in counts:
                    counts[label] = value
    return counts


def _number_beside(text: str, start: int, end: int, window: int = 26) -> int | None:
    """The number attached to the class name at [start, end).

    "label: N" is tried first because it is how lists are written; "N labels"
    is the fallback. Looking backwards stops at a bullet or comma, so one list
    item cannot claim its neighbour's number.
    """
    match = _NUMBER_AFTER_RE.match(text[end:end + window])
    if match:
        return int(match.group(1))

    before = text[max(0, start - window):start]
    before = re.split(r"[-\u2013\u2014,;]", before)[-1]
    match = _NUMBER_BEFORE_RE.search(before)
    if match:
        return int(match.group(1))
    return None


def _class_polarity(
    text: str,
    counts: dict[str, int],
    negation: tuple[str, ...],
) -> tuple[set[str], set[str]]:
    """Split mentioned classes into asserted-present and asserted-absent.

    A class named with count 0, or inside a negation, is a claim of absence.
    Reading such a mention as a presence claim was the single largest source of
    false positives in the September run.
    """
    affirmed: set[str] = set()
    negated: set[str] = set()
    for label, aliases in CLASS_ALIASES.items():
        positions = [
            m.start()
            for alias in aliases
            for m in re.finditer(rf"\b{re.escape(alias)}\b", text)
        ]
        if not positions:
            continue
        if counts.get(label) == 0:
            negated.add(label)
            continue
        # Any denial counts. A long answer repeats a class name several times -
        # "no objects of the arch label ... the count for arch is 0" - and
        # requiring every occurrence to sit next to a marker made one stray
        # mention flip the whole answer to an assertion.
        if any(_negated_at(text, pos, negation) for pos in positions):
            negated.add(label)
        else:
            affirmed.add(label)
    return affirmed, negated


_SENTENCE_SPLIT_RE = re.compile(r"[.;!?\n]")


def _negated_at(text: str, position: int, negation: tuple[str, ...], window: int = 48) -> bool:
    """Is the mention at `position` inside a denial?

    Scope is the sentence, not a character window: "le classi assenti sono
    arch, stairs e roof" denies all three, but the last one sits further from
    the marker than any fixed window that does not also leak into neighbouring
    sentences.
    """
    starts = [m.end() for m in _SENTENCE_SPLIT_RE.finditer(text) if m.end() <= position]
    sentence_start = starts[-1] if starts else 0
    ends = [m.start() for m in _SENTENCE_SPLIT_RE.finditer(text) if m.start() > position]
    sentence_end = ends[0] if ends else len(text)

    clause = text[sentence_start:position]
    if any(marker in clause for marker in negation) or NEGATION_RE.search(clause):
        return True

    # The window never leaves the sentence. "modanatura, volta e muro. Le
    # classi assenti sono arco" would otherwise let the next sentence's
    # "assenti" deny the three classes the first sentence affirms.
    context = text[max(sentence_start, position - window):min(sentence_end, position + window)]
    return any(marker in context for marker in negation) or bool(NEGATION_RE.search(context))


def _relations(text: str, negation: tuple[str, ...] = ()) -> set[tuple[str, str, str]]:
    """Collect (subject, relation, object) triples over the closed vocabulary.

    Denied relations are left out: "le colonne non supportano il tetto" states
    that the relation does not hold, and recording it as asserted would turn a
    correct denial into a claim about a class the scene does not contain.
    """
    found: set[tuple[str, str, str]] = set()

    # Models often list relations as "supports (floor -> column)", where the
    # subject follows the relation instead of preceding it. Read that form
    # first: looking backwards for a subject would pick up the previous list
    # item.
    consumed: list[tuple[int, int]] = []
    for match in _ARROW_FORM_RE.finditer(text):
        relation = _canonical_relation(match.group("rel"))
        source, target = _canonical_class(match.group("src")), _canonical_class(match.group("tgt"))
        if relation and source and target and not _directly_negated(text, match.start()):
            found.add((source, relation, target))
            consumed.append(match.span())

    for relation, aliases in RELATION_ALIASES.items():
        for alias in _sorted_aliases(aliases):
            for match in re.finditer(rf"\b{re.escape(alias)}\b", text):
                # The arrow form already read this span; re-reading it with the
                # positional heuristic invents triples out of its own operands.
                if any(start <= match.start() < end for start, end in consumed):
                    continue
                if _directly_negated(text, match.start()) or (
                    negation and _negated_at(text, match.start(), negation, window=30)
                ):
                    continue
                subject = _nearest_entity(text, match.start(), backwards=True)
                target = _nearest_entity(text, match.end(), backwards=False)
                if subject and target:
                    found.add((subject, relation, target))
    return found


_ARROW_FORM_RE = re.compile(
    r"(?P<rel>[a-z_]+)\s*[\(:]\s*(?P<src>[a-z_]+(?:_\d+)?)\s*(?:->|=>|\u2192)\s*"
    r"(?P<tgt>[a-z_]+(?:_\d+)?)"
)


def _canonical_relation(token: str) -> str | None:
    for relation, aliases in RELATION_ALIASES.items():
        if token == relation or token in aliases:
            return relation
    return None


def _canonical_class(token: str) -> str | None:
    if OBJECT_ID_RE.fullmatch(token):
        return token
    for label, aliases in CLASS_ALIASES.items():
        if token == label or token in aliases:
            return label
    return None


def _nearest_entity(text: str, position: int, backwards: bool, window: int = 60) -> str | None:
    span = text[max(0, position - window):position] if backwards else text[position:position + window]
    ids = OBJECT_ID_RE.findall(span)
    if ids:
        return ids[-1] if backwards else ids[0]
    best: tuple[int, str] | None = None
    for label, aliases in CLASS_ALIASES.items():
        for alias in aliases:
            for match in re.finditer(rf"\b{re.escape(alias)}\b", span):
                distance = len(span) - match.end() if backwards else match.start()
                if best is None or distance < best[0]:
                    best = (distance, label)
    return best[1] if best else None


def _total_count(text: str, totals: tuple[str, ...]) -> int | None:
    """Read a scene total only when the answer actually says it is a total."""
    for marker in totals:
        for match in re.finditer(re.escape(marker), text):
            window_after = text[match.end():match.end() + 30]
            window_before = text[max(0, match.start() - 30):match.start()]
            found = re.search(r"\b(\d+)\b", window_after) or re.search(r"\b(\d+)\b(?!.*\b\d+\b)", window_before)
            if found:
                return int(found.group(1))
    match = re.search(
        r"\b(\d+)\s+(?:oggetti|objects|elementi|elements)\b(?!\s+(?:di|of|per))",
        text,
    )
    return int(match.group(1)) if match else None


def _coordinates(text: str) -> list[tuple[float, float, float]]:
    pattern = re.compile(
        r"(-?\d+(?:[.,]\d+)?)\s*[,;]\s*(-?\d+(?:[.,]\d+)?)\s*[,;]\s*(-?\d+(?:[.,]\d+)?)"
    )
    out: list[tuple[float, float, float]] = []
    for match in pattern.finditer(text):
        try:
            out.append(tuple(float(g.replace(",", ".")) for g in match.groups()))  # type: ignore[arg-type]
        except ValueError:
            continue
    return out


def _polarity(text: str) -> bool | None:
    """Yes/no reading for questions that ask whether something holds."""
    head = text[:80]
    for token in _DENY:
        if re.search(rf"^\W*{re.escape(token)}\b", head):
            return False
    for token in _AFFIRM:
        if re.search(rf"^\W*{re.escape(token)}\b", head):
            return True
    if any(marker in text for marker in ("non ci sono", "non e presente", "not present", "are no")):
        return False
    return None
