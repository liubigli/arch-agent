"""Open3D viewer for semantic point clouds and DBSCAN object clusters."""

from __future__ import annotations

import argparse
import colorsys
import os
from pathlib import Path, PureWindowsPath

import numpy as np


if os.environ.get("WSL_DISTRO_NAME") and os.environ.get("DISPLAY"):
    # Open3D/GLFW can fail on WSLg Wayland with "Failed to initialize GLEW".
    # Using the X11 socket exposed by WSLg is more reliable for this viewer.
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ.setdefault("XDG_SESSION_TYPE", "x11")

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    o3d = None

from arch_agent.pipeline.loader import load_semantic_point_cloud
from arch_agent.pipeline.segmentation import extract_semantic_objects


def resolve_local_path(path_value: str) -> Path:
    """Resolve Windows paths when this script is run from WSL."""
    path = Path(path_value)
    if path.exists():
        return path

    windows_path = PureWindowsPath(path_value)
    if windows_path.drive:
        drive = windows_path.drive.rstrip(":").lower()
        wsl_path = Path("/mnt") / drive / Path(*windows_path.parts[1:])
        if wsl_path.exists():
            return wsl_path

    return path


def _require_open3d() -> None:
    if o3d is None:
        raise SystemExit(
            "Open3D is not available. Run 'pixi install' from the project root."
        )


def _stable_palette(keys: list[str], saturation: float = 0.72, value: float = 0.92) -> dict[str, np.ndarray]:
    if not keys:
        return {}

    palette = {}
    for index, key in enumerate(keys):
        hue = index / max(1, len(keys))
        palette[key] = np.array(colorsys.hsv_to_rgb(hue, saturation, value), dtype=float)
    return palette


def _make_point_cloud(points: np.ndarray, colors: np.ndarray):
    _require_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(float))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(float))
    return pcd


def _scene_diagonal(points: np.ndarray) -> float:
    if points.size == 0:
        return 1.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def _show_geometries(
    geometries: list,
    window_name: str,
    point_size: float,
    background_color: tuple[float, float, float],
) -> None:
    _require_open3d()
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(window_name=window_name, width=1200, height=800):
        raise SystemExit(
            "Open3D could not create a window. If you are in WSL, check that WSLg "
            "or an X server is available."
        )

    for geometry in geometries:
        vis.add_geometry(geometry)

    render_option = vis.get_render_option()
    render_option.point_size = float(point_size)
    render_option.background_color = np.array(background_color, dtype=float)

    vis.run()
    vis.destroy_window()


def visualize_semantic_pointcloud(
    df,
    point_size: float = 3.0,
    classes: list[str] | None = None,
    add_axes: bool = False,
) -> None:
    """Visualize the point cloud with one flat color per semantic label."""
    if classes:
        df = df[df["semantic_label"].isin(classes)].copy()

    if df.empty:
        print("No points to visualize.")
        return

    points = df[["x", "y", "z"]].to_numpy(float)
    labels = df["semantic_label"].astype(str).to_numpy()
    unique_labels = sorted(df["semantic_label"].astype(str).unique())
    color_map = _stable_palette(unique_labels)
    colors = np.vstack([color_map[label] for label in labels])

    pcd = _make_point_cloud(points, colors)
    geometries = [pcd]
    if add_axes:
        axis_size = max(0.5, _scene_diagonal(points) * 0.08)
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size))

    print(f"Visualizing {len(points):,} points with {len(unique_labels)} semantic classes.")
    for label in unique_labels:
        count = int((labels == label).sum())
        print(f"  - {label}: {count:,} points")

    _show_geometries(
        geometries,
        window_name="Semantic Point Cloud",
        point_size=point_size,
        background_color=(1.0, 1.0, 1.0),
    )


def visualize_clustered_objects(
    objects: dict,
    point_size: float = 3.0,
    classes: list[str] | None = None,
    with_boxes: bool = False,
    add_axes: bool = False,
) -> None:
    """Visualize DBSCAN objects with a different color for each cluster."""
    selected_objects = {
        name: data
        for name, data in objects.items()
        if not classes or data.get("semantic_label") in classes
    }

    if not selected_objects:
        print("No DBSCAN objects to visualize.")
        return

    object_names = sorted(selected_objects)
    color_map = _stable_palette(object_names, saturation=0.62, value=0.95)
    all_points = []
    all_colors = []
    geometries = []

    for object_name in object_names:
        object_data = selected_objects[object_name]
        points_df = object_data["points"]
        xyz = points_df[["x", "y", "z"]].to_numpy(float)
        if xyz.size == 0:
            continue

        color = color_map[object_name]
        all_points.append(xyz)
        all_colors.append(np.repeat(color[None, :], xyz.shape[0], axis=0))

        if with_boxes:
            bounds = object_data["bounds"]
            box = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=np.asarray(bounds["min"], dtype=float),
                max_bound=np.asarray(bounds["max"], dtype=float),
            )
            box.color = color
            geometries.append(box)

    if not all_points:
        print("DBSCAN clusters are empty.")
        return

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    geometries.insert(0, _make_point_cloud(points, colors))

    if add_axes:
        axis_size = max(0.5, _scene_diagonal(points) * 0.08)
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size))

    print(f"Visualizing {len(object_names)} DBSCAN objects and {len(points):,} points.")
    for object_name in object_names:
        object_data = selected_objects[object_name]
        label = object_data["semantic_label"]
        count = object_data["point_count"]
        print(f"  - {object_name} ({label}): {count:,} points")

    _show_geometries(
        geometries,
        window_name="DBSCAN Clustered Point Cloud Objects",
        point_size=point_size,
        background_color=(0.1, 0.1, 0.1),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize semantic point clouds and DBSCAN clusters with Open3D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("point_cloud_path", help="Input .laz/.las point cloud.")
    parser.add_argument(
        "--mode",
        choices=("semantic", "clusters", "both"),
        default="semantic",
        help="Viewer mode.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional semantic classes to display, for example: --classes column wall",
    )
    parser.add_argument("--sample-n", type=int, default=150_000, help="Max points to load; use 0 for all points.")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN epsilon.")
    parser.add_argument("--min-samples", type=int, default=15, help="DBSCAN min_samples.")
    parser.add_argument("--point-size", type=float, default=1.5, help="Open3D point size.")
    parser.add_argument("--with-boxes", action="store_true", help="Show DBSCAN AABB boxes in cluster mode.")
    parser.add_argument("--axes", action="store_true", help="Show a coordinate frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    point_cloud_path = resolve_local_path(args.point_cloud_path)
    if not point_cloud_path.exists():
        raise FileNotFoundError(f"Point cloud path not found: {point_cloud_path}")

    print(f"Loading point cloud: {point_cloud_path}")
    df = load_semantic_point_cloud(
        str(point_cloud_path),
        sample_n=args.sample_n if args.sample_n > 0 else None,
    )

    if args.mode in {"semantic", "both"}:
        visualize_semantic_pointcloud(
            df,
            point_size=args.point_size,
            classes=args.classes,
            add_axes=args.axes,
        )

    if args.mode in {"clusters", "both"}:
        print(f"Running DBSCAN (eps={args.eps}, min_samples={args.min_samples})")
        objects = extract_semantic_objects(df, eps=args.eps, min_samples=args.min_samples)
        print(f"Found {len(objects)} DBSCAN objects.")
        visualize_clustered_objects(
            objects,
            point_size=args.point_size,
            classes=args.classes,
            with_boxes=args.with_boxes,
            add_axes=args.axes,
        )


if __name__ == "__main__":
    main()
