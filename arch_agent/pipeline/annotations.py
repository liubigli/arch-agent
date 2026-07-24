from __future__ import annotations

from pathlib import Path
import math
import unicodedata

import numpy as np
import pandas as pd


LABEL_ALIASES = {
    "arch": "arch",
    "arco": "arch",
    "archi": "arch",
    "column": "column",
    "columns": "column",
    "colonna": "column",
    "colonne": "column",
    "door_window": "door_window",
    "door": "door_window",
    "doors": "door_window",
    "window": "door_window",
    "windows": "door_window",
    "apertura": "door_window",
    "aperture": "door_window",
    "floor": "floor",
    "floors": "floor",
    "pavimento": "floor",
    "pavimenti": "floor",
    "molding": "moldings",
    "moldings": "moldings",
    "modanatura": "moldings",
    "modanature": "moldings",
    "roof": "roof",
    "roofs": "roof",
    "tetto": "roof",
    "tetti": "roof",
    "stairs": "stairs",
    "stair": "stairs",
    "scala": "stairs",
    "scale": "stairs",
    "vault": "vault",
    "vaults": "vault",
    "volta": "vault",
    "volte": "vault",
    "wall": "wall",
    "walls": "wall",
    "muro": "wall",
    "muri": "wall",
    "parete": "wall",
    "pareti": "wall",
    "other": "other",
    "altro": "other",
}

LABEL_COLUMNS = ("semantic_label", "label", "class", "classe", "object_class", "element_class", "type", "tipo")
OBJECT_COLUMNS = ("object_name", "object_id", "id", "name", "nome")
POSITION_COLUMNS = ("position", "posizione", "spatial_position", "location", "localizzazione")
X_COLUMNS = ("x", "centroid_x", "center_x", "cx")
Y_COLUMNS = ("y", "centroid_y", "center_y", "cy")
Z_COLUMNS = ("z", "centroid_z", "center_z", "cz")


def resolve_annotation_csv(point_cloud_path: str, explicit_path: str | None = None) -> str | None:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation CSV not found: {path}")
        return str(path)

    laz_path = Path(point_cloud_path)
    candidates = [
        laz_path.with_suffix(".csv"),
        laz_path.with_name(f"{laz_path.stem}_annotations.csv"),
        laz_path.with_name(f"{laz_path.stem}_metadata.csv"),
        laz_path.with_name(f"{laz_path.stem}_descriptions.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def load_object_annotations(
    csv_path: str,
    objects: dict,
    max_distance: float = 2.0,
) -> tuple[dict[str, list[dict]], list[dict]]:
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df = df.rename(columns={column: _normalize_column(column) for column in df.columns})

    annotations: dict[str, list[dict]] = {}
    unmatched: list[dict] = []
    for row_index, row in df.iterrows():
        annotation = _annotation_from_row(row, row_index, csv_path)
        semantic_label = _semantic_label_from_row(row)
        object_name, match_info = _match_row_to_object(
            row,
            objects,
            semantic_label=semantic_label,
            max_distance=max_distance,
        )
        annotation["semantic_label"] = semantic_label
        annotation["match"] = match_info
        if object_name is None:
            unmatched.append(annotation)
            continue

        annotation["object_name"] = object_name
        annotations.setdefault(object_name, []).append(annotation)

    return annotations, unmatched


def _annotation_from_row(row: pd.Series, row_index: int, csv_path: str) -> dict:
    values = {
        column: _clean_value(value)
        for column, value in row.items()
        if _clean_value(value) is not None
    }
    values["source_csv"] = str(csv_path)
    values["source_row"] = int(row_index) + 2
    return values


def _match_row_to_object(
    row: pd.Series,
    objects: dict,
    semantic_label: str | None,
    max_distance: float,
) -> tuple[str | None, dict]:
    explicit_name = _first_value(row, OBJECT_COLUMNS)
    if explicit_name and explicit_name in objects:
        return explicit_name, {"method": "object_name"}

    candidates = [
        name for name, obj in objects.items()
        if semantic_label is None or obj.get("semantic_label") == semantic_label
    ]
    if not candidates:
        return None, {"method": "none", "reason": "no candidates for semantic label"}

    position = _first_value(row, POSITION_COLUMNS)
    if position:
        selected = _select_by_position(candidates, objects, str(position))
        if selected:
            return selected, {"method": "position", "position": str(position)}

    coordinates = _coordinates_from_row(row)
    if coordinates is not None:
        selected, distance = _nearest_object(candidates, objects, coordinates)
        if selected and distance <= max_distance:
            return selected, {
                "method": "nearest_centroid",
                "distance_m": float(distance),
                "max_distance_m": float(max_distance),
            }
        return None, {
            "method": "nearest_centroid",
            "reason": "nearest object is beyond max_distance",
            "nearest_object": selected,
            "distance_m": None if selected is None else float(distance),
            "max_distance_m": float(max_distance),
        }

    if len(candidates) == 1:
        return candidates[0], {"method": "single_candidate"}

    return None, {
        "method": "none",
        "reason": "ambiguous row: provide x/y/z, position, or a single matching class",
        "candidate_count": len(candidates),
    }


def _semantic_label_from_row(row: pd.Series) -> str | None:
    raw_label = _first_value(row, LABEL_COLUMNS)
    if raw_label is None:
        return None
    return LABEL_ALIASES.get(_normalize_text(str(raw_label)), _normalize_text(str(raw_label)))


def _coordinates_from_row(row: pd.Series) -> np.ndarray | None:
    x = _float_value(_first_value(row, X_COLUMNS))
    y = _float_value(_first_value(row, Y_COLUMNS))
    z = _float_value(_first_value(row, Z_COLUMNS))
    if x is None or y is None:
        return None
    if z is None:
        return np.array([x, y], dtype=float)
    return np.array([x, y, z], dtype=float)


def _nearest_object(candidates: list[str], objects: dict, coordinates: np.ndarray) -> tuple[str | None, float]:
    best_name = None
    best_distance = math.inf
    for name in candidates:
        centroid = np.asarray(objects[name]["centroid"], dtype=float)
        compare = centroid[: len(coordinates)]
        distance = float(np.linalg.norm(compare - coordinates))
        if distance < best_distance:
            best_name = name
            best_distance = distance
    return best_name, best_distance


def _select_by_position(candidates: list[str], objects: dict, position: str) -> str | None:
    normalized = _normalize_text(position)
    centroids = {
        name: np.asarray(objects[name]["centroid"], dtype=float)
        for name in candidates
    }
    if not centroids:
        return None

    if any(term in normalized for term in ("central", "centrale", "centro", "middle")):
        mean = np.mean(np.array(list(centroids.values())), axis=0)
        return min(candidates, key=lambda name: float(np.linalg.norm(centroids[name] - mean)))
    if any(term in normalized for term in ("sinistra", "left", "ovest", "west")):
        return min(candidates, key=lambda name: float(centroids[name][0]))
    if any(term in normalized for term in ("destra", "right", "est", "east")):
        return max(candidates, key=lambda name: float(centroids[name][0]))
    if any(term in normalized for term in ("sud", "south", "front", "davanti")):
        return min(candidates, key=lambda name: float(centroids[name][1]))
    if any(term in normalized for term in ("nord", "north", "back", "dietro")):
        return max(candidates, key=lambda name: float(centroids[name][1]))
    if any(term in normalized for term in ("bassa", "basso", "inferiore", "lower", "bottom")):
        return min(candidates, key=lambda name: float(centroids[name][2]))
    if any(term in normalized for term in ("alta", "alto", "superiore", "upper", "top")):
        return max(candidates, key=lambda name: float(centroids[name][2]))
    if any(term in normalized for term in ("unica", "singola", "only", "single")) and len(candidates) == 1:
        return candidates[0]
    return None


def _first_value(row: pd.Series, columns: tuple[str, ...]) -> object | None:
    for column in columns:
        if column in row.index:
            value = _clean_value(row[column])
            if value is not None:
                return value
    return None


def _clean_value(value: object) -> object | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _float_value(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _normalize_column(value: object) -> str:
    normalized = _normalize_text(str(value))
    return re_sub_non_word(normalized)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.strip().lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def re_sub_non_word(value: str) -> str:
    chars = [char if char.isalnum() else "_" for char in value]
    compact = "_".join(part for part in "".join(chars).split("_") if part)
    return compact
