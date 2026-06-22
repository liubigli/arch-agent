from pathlib import Path

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


def _points_to_df(points, dimensions: set[str]) -> pd.DataFrame:
    if "semantic_label" in dimensions:
        raw_labels = points["semantic_label"]
    elif "classification" in dimensions:
        raw_labels = points.classification
    else:
        raise ValueError(
            "LAZ file must contain a 'semantic_label' extra dimension or "
            "the standard 'classification' dimension."
        )

    df = pd.DataFrame(
        {
            "x": points.x,
            "y": points.y,
            "z": points.z,
            "semantic_label": raw_labels,
        }
    )

    if {"red", "green", "blue"} <= dimensions:
        df["R"] = points.red
        df["G"] = points.green
        df["B"] = points.blue

    normal_aliases = {
        "nx": ("nx", "normal_x"),
        "ny": ("ny", "normal_y"),
        "nz": ("nz", "normal_z"),
    }
    for out_col, candidates in normal_aliases.items():
        for dim_name in candidates:
            if dim_name in dimensions:
                df[out_col] = points[dim_name]
                break

    return df


def _load_laz_file(file_path: Path, sample_n: int | None) -> pd.DataFrame:
    with laspy.open(file_path) as laz:
        dimensions = set(laz.header.point_format.dimension_names)
        point_count = laz.header.point_count

        if sample_n and point_count > sample_n:
            target = min(point_count, sample_n * _STREAM_OVERSAMPLE_FACTOR)
            sample_ratio = target / point_count
            sampled_parts = []

            for chunk_index, points in enumerate(
                laz.chunk_iterator(_LAZ_CHUNK_SIZE),
                start=1,
            ):
                chunk_df = _points_to_df(points, dimensions)
                chunk_target = max(1, round(len(chunk_df) * sample_ratio))
                if len(chunk_df) > chunk_target:
                    chunk_df = chunk_df.sample(
                        n=chunk_target,
                        random_state=chunk_index,
                    )
                sampled_parts.append(chunk_df)

            return pd.concat(sampled_parts, ignore_index=True)

        return _points_to_df(laz.read(), dimensions)


def load_semantic_point_cloud(file_path: str, sample_n: int = 150_000) -> pd.DataFrame:
    label_map = _build_label_map()

    df = _load_laz_file(Path(file_path), sample_n=sample_n)
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
