import pandas as pd

from ..settings import get_config


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


def load_semantic_point_cloud(file_path: str, sample_n: int = 150_000) -> pd.DataFrame:
    label_map = _build_label_map()

    df = pd.read_csv(file_path, sep=";", decimal=".")
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
