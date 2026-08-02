from __future__ import annotations

from typing import Final, TypedDict


class PromptExample(TypedDict):
    id: int
    prompt: str


PROMPT_RUBRIC: Final[dict[str, str]] = {
    "absent_or_wrong": "Risposta assente o errata",
    "partial": "Parzialmente corretta, generica o incompleta",
    "correct": "Corretta, specifica e coerente con la scena",
}


PROMPT_EXAMPLES: Final[list[PromptExample]] = [
    {
        "id": 1,
        "prompt": "Descrivi la scena in modo sintetico, indicando gli elementi principali che riconosci.",
    },
    {
        "id": 2,
        "prompt": "Qual e l'elemento piu importante o dominante nella scena?",
    },
    {
        "id": 3,
        "prompt": "Quali oggetti o elementi architettonici riesci a identificare con maggiore sicurezza?",
    },
    {
        "id": 4,
        "prompt": "La scena rappresenta uno spazio interno, esterno o misto?",
    },
    {
        "id": 5,
        "prompt": "Quali sono i confini principali dello spazio rappresentato?",
    },
    {
        "id": 6,
        "prompt": "Quali elementi sembrano delimitare, contenere o organizzare la scena?",
    },
    {
        "id": 7,
        "prompt": "Quali elementi risultano adiacenti tra loro?",
    },
    {
        "id": 8,
        "prompt": "Quali elementi sembrano essere sopra o sotto altri elementi?",
    },
    {
        "id": 9,
        "prompt": "Ci sono elementi che si intersecano o si sovrappongono?",
    },
    {
        "id": 10,
        "prompt": "Quali elementi sembrano supportare o sostenere altri elementi?",
    },
    {
        "id": 11,
        "prompt": "Quali oggetti sembrano appartenere allo stesso sistema costruttivo?",
    },
    {
        "id": 12,
        "prompt": "Riesci a distinguere tra elementi portanti ed elementi non portanti?",
    },
    {
        "id": 13,
        "prompt": "Quali elementi sembrano avere una funzione strutturale?",
    },
    {
        "id": 14,
        "prompt": "Quali elementi sembrano avere una funzione distributiva, di passaggio o di accesso?",
    },
    {
        "id": 15,
        "prompt": "La scena mostra una gerarchia chiara tra elementi principali e secondari?",
    },
    {
        "id": 16,
        "prompt": "Quali relazioni spaziali sono piu evidenti nella scena?",
    },
    {
        "id": 17,
        "prompt": "Ci sono elementi ambigui o difficili da interpretare? Quali e perche?",
    },
    {
        "id": 18,
        "prompt": "La risposta del modello distingue tra cio che vede e cio che inferisce?",
    },
    {
        "id": 19,
        "prompt": (
            "Il modello sta descrivendo correttamente le relazioni tra oggetti "
            "o sta facendo ipotesi troppo generiche?"
        ),
    },
    {
        "id": 20,
        "prompt": (
            "Se dovessi sintetizzare la scena con una sola etichetta tipologica, "
            "quale useresti e con quale livello di confidenza?"
        ),
    },
]


__all__ = ["PROMPT_EXAMPLES", "PROMPT_RUBRIC", "PromptExample"]
