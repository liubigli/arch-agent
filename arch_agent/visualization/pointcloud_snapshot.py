"""Render an offline point-cloud snapshot with semantic DBSCAN boxes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ..pipeline.loader import load_semantic_point_cloud  # noqa: E402
from ..pipeline.segmentation import extract_semantic_objects  # noqa: E402


CLASS_COLORS = {
    "arch": "#e41a1c",
    "column": "#2455ff",
    "moldings": "#d81bce",
    "floor": "#22a447",
    "door_window": "#00bfd5",
    "wall": "#ff3030",
    "stairs": "#ff7f00",
    "vault": "#8a2be2",
    "roof": "#d39b00",
    "other": "#666666",
}


def _rgb_colors(df) -> np.ndarray:
    if not {"R", "G", "B"}.issubset(df.columns):
        return np.full((len(df), 3), 0.35)

    rgb = df[["R", "G", "B"]].to_numpy(dtype=float)
    maximum = float(np.nanmax(rgb)) if rgb.size else 1.0
    scale = 65535.0 if maximum > 255.0 else 255.0
    return np.clip(rgb / scale, 0.0, 1.0)


def _semantic_colors(df) -> np.ndarray:
    return np.vstack(
        [
            matplotlib.colors.to_rgb(
                CLASS_COLORS.get(str(label), CLASS_COLORS["other"])
            )
            for label in df["semantic_label"]
        ]
    )


def _box_edges(bounds_min: np.ndarray, bounds_max: np.ndarray):
    x0, y0, z0 = bounds_min
    x1, y1, z1 = bounds_max
    corners = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ]
    )
    indices = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    return ((corners[a], corners[b]) for a, b in indices)


def render_snapshot(
    scene_path: str,
    output_path: str,
    *,
    eps: float = 0.5,
    min_samples: int = 15,
    sample_n: int = 150_000,
    point_size: float = 0.35,
    elev: float = 48.0,
    azim: float = -58.0,
    show_labels: bool = False,
    color_mode: str = "rgb",
) -> Path:
    df = load_semantic_point_cloud(scene_path, sample_n=sample_n)
    objects = extract_semantic_objects(df, eps=eps, min_samples=min_samples)
    if not objects:
        raise ValueError("No DBSCAN objects were extracted from the scene.")

    xyz = df[["x", "y", "z"]].to_numpy(dtype=float)
    origin = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
    local_xyz = xyz - origin

    figure = plt.figure(figsize=(16, 10), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("white")
    axis.scatter(
        local_xyz[:, 0],
        local_xyz[:, 1],
        local_xyz[:, 2],
        c=_rgb_colors(df) if color_mode == "rgb" else _semantic_colors(df),
        s=point_size,
        alpha=0.72,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )

    present_classes = set()
    ordered_objects = sorted(
        objects.items(), key=lambda item: item[1]["semantic_label"] == "vault"
    )
    for object_name, obj in ordered_objects:
        label = str(obj["semantic_label"])
        present_classes.add(label)
        color = CLASS_COLORS.get(label, CLASS_COLORS["other"])
        bounds_min = np.asarray(obj["bounds"]["min"], dtype=float) - origin
        bounds_max = np.asarray(obj["bounds"]["max"], dtype=float) - origin
        for start, end in _box_edges(bounds_min, bounds_max):
            linewidth = 2.2 if label == "vault" else 1.15
            axis.plot(*zip(start, end), color=color, linewidth=linewidth, alpha=0.95)
        if show_labels:
            anchor = (bounds_min + bounds_max) / 2.0
            anchor[2] = bounds_max[2]
            axis.text(*anchor, object_name, color=color, fontsize=5.5)

    spans = np.ptp(local_xyz, axis=0)
    center = (local_xyz.min(axis=0) + local_xyz.max(axis=0)) / 2.0
    half_span = max(float(spans.max()) / 2.0, 0.5)
    axis.set_xlim(center[0] - half_span, center[0] + half_span)
    axis.set_ylim(center[1] - half_span, center[1] + half_span)
    axis.set_zlim(center[2] - half_span, center[2] + half_span)
    axis.set_box_aspect((1, 1, 0.7))
    axis.view_init(elev=elev, azim=azim)
    axis.set_axis_off()

    legend = [
        Line2D([0], [0], color=CLASS_COLORS.get(label, "#666666"), lw=4, label=label)
        for label in sorted(present_classes)
    ]
    axis.legend(
        handles=legend,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=11,
        handlelength=1.0,
    )
    figure.suptitle(
        f"{Path(scene_path).stem} - {color_mode.upper()} point cloud and DBSCAN objects",
        fontsize=15,
        y=0.94,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=250, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)
    print(f"Rendered {len(df):,} points and {len(objects)} objects")
    print(f"Saved: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save a point-cloud image with semantic DBSCAN bounding boxes."
    )
    parser.add_argument("point_cloud_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--sample-n", type=int, default=150_000)
    parser.add_argument("--point-size", type=float, default=0.35)
    parser.add_argument("--elev", type=float, default=48.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--labels", action="store_true")
    parser.add_argument(
        "--color-mode", choices=("rgb", "semantic"), default="rgb"
    )
    args = parser.parse_args()

    render_snapshot(
        args.point_cloud_path,
        args.output,
        eps=args.eps,
        min_samples=args.min_samples,
        sample_n=args.sample_n,
        point_size=args.point_size,
        elev=args.elev,
        azim=args.azim,
        show_labels=args.labels,
        color_mode=args.color_mode,
    )


if __name__ == "__main__":
    main()
