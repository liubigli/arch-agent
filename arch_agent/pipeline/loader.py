from pathlib import Path

import numpy as np
import pandas as pd
import laspy

from ..settings import get_config


_LAZ_CHUNK_SIZE = 500_000
_STREAM_OVERSAMPLE_FACTOR = 2


def _build_label_map() -> dict[float, str]:
    names = get_config()["semantic_classes"]["names"]
    return {float(i): name for i, name in enumerate(names)}


def _voxel_sample_by_class(
    df: pd.DataFrame,
    sample_n: int,
    voxel_size: float = 0.05,
) -> pd.DataFrame:
    if sample_n <= 0 or len(df) <= sample_n:
        return df

    sampled_parts = []
    class_counts = df["semantic_label"].value_counts()

    for label, count in class_counts.items():
        class_df = df[df["semantic_label"] == label].copy()
        class_quota = max(1, round(sample_n * count / len(df)))
        class_quota = min(class_quota, len(class_df))

        for axis in ["x", "y", "z"]:
            class_df[f"_voxel_{axis}"] = (class_df[axis] // voxel_size).astype(int)

        voxel_cols = ["_voxel_x", "_voxel_y", "_voxel_z"]
        voxel_sample = (
            class_df
            .groupby(voxel_cols, group_keys=False)
            .sample(n=1, random_state=1)
            .drop(columns=voxel_cols)
        )

        if len(voxel_sample) > class_quota:
            voxel_sample = voxel_sample.sample(n=class_quota, random_state=1)

        sampled_parts.append(voxel_sample)

    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > sample_n:
        sampled = sampled.sample(n=sample_n, random_state=1)

    return sampled


def _sample_indices(length: int, sample_ratio: float, seed: int) -> np.ndarray:
    target = max(1, round(length * sample_ratio))
    if target >= length:
        return np.arange(length)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=target, replace=False))


def _points_to_arrays(
    points,
    dimensions: set[str],
    indices: np.ndarray | None = None,
    include_normals: bool = False,
) -> dict[str, np.ndarray]:
    if "semantic_label" in dimensions:
        raw_labels = points["semantic_label"]
    elif "classification" in dimensions:
        raw_labels = points.classification
    else:
        raise ValueError(
            "LAZ file must contain a 'semantic_label' extra dimension or "
            "the standard 'classification' dimension."
        )

    arrays = {
        "x": np.asarray(points.x, dtype=np.float32),
        "y": np.asarray(points.y, dtype=np.float32),
        "z": np.asarray(points.z, dtype=np.float32),
        "semantic_label": np.asarray(raw_labels),
    }

    if {"red", "green", "blue"} <= dimensions:
        arrays["R"] = np.asarray(points.red)
        arrays["G"] = np.asarray(points.green)
        arrays["B"] = np.asarray(points.blue)

    if include_normals:
        normal_aliases = {
            "nx": ("nx", "normal_x"),
            "ny": ("ny", "normal_y"),
            "nz": ("nz", "normal_z"),
        }
        for out_col, candidates in normal_aliases.items():
            for dim_name in candidates:
                if dim_name in dimensions:
                    arrays[out_col] = np.asarray(points[dim_name], dtype=np.float32)
                    break

    if indices is not None:
        arrays = {name: values[indices] for name, values in arrays.items()}

    return arrays


def _concat_array_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    columns = parts[0].keys()
    return {
        column: np.concatenate([part[column] for part in parts])
        for column in columns
    }


def _arrays_to_df(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(arrays, copy=False)


def _load_laz_file(
    file_path: Path,
    sample_n: int | None,
    include_normals: bool = False,
) -> pd.DataFrame:
    with laspy.open(file_path) as laz:
        dimensions = set(laz.header.point_format.dimension_names)
        point_count = laz.header.point_count

        if sample_n and point_count > sample_n:
            target = min(point_count, sample_n * _STREAM_OVERSAMPLE_FACTOR)
            sample_ratio = target / point_count
            sampled_parts: list[dict[str, np.ndarray]] = []

            for chunk_index, points in enumerate(
                laz.chunk_iterator(_LAZ_CHUNK_SIZE),
                start=1,
            ):
                indices = _sample_indices(len(points), sample_ratio, seed=chunk_index)
                sampled_parts.append(
                    _points_to_arrays(
                        points,
                        dimensions,
                        indices=indices,
                        include_normals=include_normals,
                    )
                )

            return _arrays_to_df(_concat_array_parts(sampled_parts))

        return _arrays_to_df(
            _points_to_arrays(
                laz.read(),
                dimensions,
                include_normals=include_normals,
            )
        )


def load_semantic_point_cloud(
    file_path: str,
    sample_n: int = 150_000,
    include_normals: bool = False,
) -> pd.DataFrame:
    label_map = _build_label_map()

    df = _load_laz_file(
        Path(file_path),
        sample_n=sample_n,
        include_normals=include_normals,
    )
    df["semantic_label"] = df["semantic_label"].map(label_map)

    n_before = len(df)
    df = df.dropna(subset=["semantic_label"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  [WARN] {n_dropped} rows with unknown label removed")

    if sample_n and len(df) > sample_n:
        df = _voxel_sample_by_class(df, sample_n=sample_n, voxel_size=0.05)

    print(f"  Loaded {len(df):,} points — {df['semantic_label'].nunique()} classes: "
          f"{sorted(df['semantic_label'].unique())}")
    return df
