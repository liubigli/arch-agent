from __future__ import annotations

import csv
from io import StringIO
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

LABEL_COLUMNS = (
    "semantic_label",
    "class_semantic_label",
    "semantic_class_label",
    "label",
    "class",
    "classe",
    "object_class",
    "element_class",
    "type",
    "tipo",
)
BOX_CENTER_X_COLUMNS = ("global_box_center_x", "box_center_x", "bbox_center_x", "x", "center_x", "cx")
BOX_CENTER_Y_COLUMNS = ("global_box_center_y", "box_center_y", "bbox_center_y", "y", "center_y", "cy")
BOX_CENTER_Z_COLUMNS = ("global_box_center_z", "box_center_z", "bbox_center_z", "z", "center_z", "cz")
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin1")


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
    df = _read_annotation_csv(csv_path)
    df = _repair_single_field_rows(df)
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


def _read_annotation_csv(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    raw = path.read_bytes()
    decoded_candidates: list[tuple[int, str, str]] = []

    for encoding in CSV_ENCODINGS:
        try:
            text = raw.decode(encoding)
            decoded_candidates.append((_decoded_text_score(text), encoding, text))
        except UnicodeDecodeError:
            if encoding.startswith("utf-8"):
                text = raw.decode(encoding, errors="replace")
                decoded_candidates.append(
                    (_decoded_text_score(text), f"{encoding}+replace", text)
                )

    last_error: Exception | None = None
    for _, _, text in sorted(decoded_candidates, key=lambda item: item[0]):
        try:
            return pd.read_csv(StringIO(text), sep=None, engine="python")
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(csv_path, sep=None, engine="python")


def _decoded_text_score(text: str) -> int:
    mojibake_markers = ("Ã", "Â", "â€", "â€™", "â€œ", "â€\x9d")
    return text.count("\ufffd") * 5 + sum(
        text.count(marker) * 3 for marker in mojibake_markers
    )


def _repair_single_field_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Recover CSV rows accidentally quoted as one comma-separated field."""
    columns = list(df.columns)
    if len(columns) < 2:
        return df

    first_column = columns[0]
    other_columns = columns[1:]
    repaired_rows: list[dict] = []
    changed = False

    for _, row in df.iterrows():
        raw_value = _clean_value(row[first_column])
        other_values_are_empty = all(
            _clean_value(row[column]) is None for column in other_columns
        )
        if isinstance(raw_value, str) and "," in raw_value and other_values_are_empty:
            parsed = next(csv.reader([raw_value]))
            if len(parsed) == len(columns):
                repaired_rows.append(dict(zip(columns, parsed)))
                changed = True
                continue
        repaired_rows.append(row.to_dict())

    if not changed:
        return df
    return pd.DataFrame(repaired_rows, columns=columns)


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
    candidates = [
        name for name, obj in objects.items()
        if semantic_label is None or obj.get("semantic_label") == semantic_label
    ]
    if not candidates:
        return None, {"method": "none", "reason": "no candidates for semantic label"}

    box_center = _global_box_center_from_row(row)
    if box_center is not None:
        selected, distance = _nearest_object_by_box_center(candidates, objects, box_center)
        if selected and distance <= max_distance:
            return selected, {
                "method": "global_box_center",
                "distance_m": float(distance),
                "max_distance_m": float(max_distance),
            }
        return None, {
            "method": "global_box_center",
            "reason": "nearest object box_center is beyond max_distance",
            "nearest_object": selected,
            "distance_m": None if selected is None else float(distance),
            "max_distance_m": float(max_distance),
        }

    return None, {
        "method": "global_box_center",
        "reason": "missing global_box_center_x/y/z",
        "candidate_count": len(candidates),
    }


def _semantic_label_from_row(row: pd.Series) -> str | None:
    raw_label = _first_value(row, LABEL_COLUMNS)
    if raw_label is None:
        return None
    return LABEL_ALIASES.get(_normalize_text(str(raw_label)), _normalize_text(str(raw_label)))


def _global_box_center_from_row(row: pd.Series) -> np.ndarray | None:
    x = _float_value(_first_value(row, BOX_CENTER_X_COLUMNS))
    y = _float_value(_first_value(row, BOX_CENTER_Y_COLUMNS))
    z = _float_value(_first_value(row, BOX_CENTER_Z_COLUMNS))
    if x is None or y is None:
        return None
    if z is None:
        return np.array([x, y], dtype=float)
    return np.array([x, y, z], dtype=float)


def _nearest_object_by_box_center(
    candidates: list[str],
    objects: dict,
    coordinates: np.ndarray,
) -> tuple[str | None, float]:
    best_name = None
    best_distance = math.inf
    for name in candidates:
        bounds = objects[name].get("bounds")
        if not bounds:
            continue
        bounds_min = np.asarray(bounds["min"], dtype=float)[: len(coordinates)]
        bounds_max = np.asarray(bounds["max"], dtype=float)[: len(coordinates)]
        box_center = ((bounds_min + bounds_max) / 2.0)[: len(coordinates)]
        distance = float(np.linalg.norm(box_center - coordinates))
        if distance < best_distance:
            best_name = name
            best_distance = distance
    return best_name, best_distance


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
        if not stripped:
            return None
        return _sanitize_text(stripped)
    return value


def _sanitize_text(value: str) -> str:
    return value.encode("utf-8", errors="replace").decode("utf-8")


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
    normalized = unicodedata.normalize("NFKD", _sanitize_text(value).strip().lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def re_sub_non_word(value: str) -> str:
    chars = [char if char.isalnum() else "_" for char in value]
    compact = "_".join(part for part in "".join(chars).split("_") if part)
    return compact
