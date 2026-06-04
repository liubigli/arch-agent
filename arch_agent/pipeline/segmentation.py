import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans

SEGMENTATION_METHODS = {
    "column": "kmeans_elbow",
    "arch": "kmeans_elbow",
    "door_window": "kmeans_elbow",
    "wall": "dbscan",
    "floor": "dbscan",
    "vault": "dbscan",
    "roof": "dbscan",
    "stairs": "dbscan",
    "moldings": "dbscan",
    "other": "dbscan",
}

KMEANS_MAX_K = {
    "column": 20,
    "arch": 20,
    "door_window": 30,
}


def extract_semantic_objects(
    df: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 15,
) -> dict:
    objects = {}

    for label in df["semantic_label"].unique():
        label_points = df[df["semantic_label"] == label]
        if len(label_points) < min_samples:
            continue

        coords = label_points[["x", "y", "z"]].values
        method = SEGMENTATION_METHODS.get(label, "dbscan")

        if method == "kmeans_elbow" and len(coords) >= min_samples * 2:
            cluster_labels = _cluster_kmeans_elbow(
                coords,
                max_k=KMEANS_MAX_K.get(label, 12),
                min_cluster_size=min_samples,
            )
        else:
            cluster_labels = _cluster_dbscan(
                coords,
                eps=eps,
                min_samples=min_samples,
            )

        label_points_copy = label_points.copy()
        label_points_copy["cluster"] = cluster_labels

        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:
                continue

            cluster_pts = label_points_copy[label_points_copy["cluster"] == cluster_id]
            if len(cluster_pts) < min_samples:
                continue

            key = f"{label}_{cluster_id}"
            objects[key] = {
                "points": cluster_pts,
                "centroid": cluster_pts[["x", "y", "z"]].mean().values,
                "bounds": {
                    "min": cluster_pts[["x", "y", "z"]].min().values,
                    "max": cluster_pts[["x", "y", "z"]].max().values,
                },
                "semantic_label": label,
                "point_count": len(cluster_pts),
                "segmentation_method": method,
            }

    return objects


def _cluster_dbscan(
    coords: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)


def _cluster_kmeans_elbow(
    coords: np.ndarray,
    max_k: int = 12,
    min_cluster_size: int = 15,
) -> np.ndarray:
    n_points = len(coords)
    if n_points < min_cluster_size * 2:
        return np.zeros(n_points, dtype=int)

    k_max = min(max_k, max(1, n_points // min_cluster_size))
    if k_max <= 1:
        return np.zeros(n_points, dtype=int)
