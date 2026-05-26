import pandas as pd

CLASS_NAMES = [
    "arch", "column", "moldings", "floor", "door_window",
    "wall", "stairs", "vault", "roof", "other",
]
LABEL_MAP = {float(i): CLASS_NAMES[i] for i in range(len(CLASS_NAMES))}


def load_semantic_point_cloud(file_path: str, sample_n: int = 150_000) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=";", decimal=".")
    df["semantic_label"] = df["semantic_label"].map(LABEL_MAP)

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
