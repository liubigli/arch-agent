import pandas as pd

from ..settings import get_config


def _build_label_map() -> dict[float, str]:
    names = get_config()["semantic_classes"]["names"]
    return {float(i): name for i, name in enumerate(names)}


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
        df = df.sample(n=sample_n, random_state=1)

    print(f"  Loaded {len(df):,} points — {df['semantic_label'].nunique()} classes: "
          f"{sorted(df['semantic_label'].unique())}")
    return df
