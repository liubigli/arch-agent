import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


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
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)

        label_points_copy = label_points.copy()
        label_points_copy["cluster"] = clustering.labels_

        for cluster_id in np.unique(clustering.labels_):
            if cluster_id == -1:
                continue
            cluster_pts = label_points_copy[label_points_copy["cluster"] == cluster_id]
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
            }

    return objects
