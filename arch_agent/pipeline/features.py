import numpy as np
from scipy.spatial import ConvexHull

from ..settings import get_config

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False


def _structural_elements() -> set[str]:
    return set(get_config()["semantic_classes"]["structural"])


def sum_two_integers(first: int, second: int) -> int:
    return first + second


def _surface_area_convex_hull(points: np.ndarray) -> float:
    try:
        return float(ConvexHull(points).area) if points.shape[0] >= 4 else 0.0
    except Exception:
        return 0.0


def _surface_area_poisson(points: np.ndarray, normals: np.ndarray) -> float:
    if not _O3D_AVAILABLE or points.shape[0] < 10:
        return _surface_area_convex_hull(points)
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(float))
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(float))
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
        area = float(mesh.get_surface_area())
        return area if area > 0 else _surface_area_convex_hull(points)
    except Exception:
        return _surface_area_convex_hull(points)


def compute_object_features(objects: dict, use_normals: bool = False) -> dict:
    structural = _structural_elements()
    features = {}

    for obj_name, obj_data in objects.items():
        pts_df = obj_data["points"]
        points = pts_df[["x", "y", "z"]].to_numpy(float)
        mins = obj_data["bounds"]["min"]
        maxs = obj_data["bounds"]["max"]
        dims = maxs - mins
        volume = float(np.prod(dims))

        has_normals = all(c in pts_df.columns for c in ["nx", "ny", "nz"])
        if use_normals and has_normals:
            normals = pts_df[["nx", "ny", "nz"]].to_numpy(float)
            surface_area = _surface_area_poisson(points, normals)
        else:
            surface_area = _surface_area_convex_hull(points)

        compactness = (surface_area ** 3) / (36 * np.pi * volume ** 2) if volume > 0 else 0.0
        label = obj_data["semantic_label"]

        features[obj_name] = {
            "volume": volume,
            "surface_area": surface_area,
            "compactness": float(compactness),
            "height": float(dims[2]),
            "semantic_label": label,
            "centroid": obj_data["centroid"],
            "point_density": float(obj_data["point_count"] / volume) if volume > 0 else 0.0,
            "element_type": "structural" if label in structural else "finishing",
        }

    return features


def compute_scene_features(objects: dict) -> dict:
    features = {}
    room_volume = _estimate_room_volume_box(objects)
    if room_volume:
        features["room_volume"] = room_volume
    return features


def _estimate_room_volume_box(objects: dict) -> dict | None:
    floors = [
        (name, obj) for name, obj in objects.items()
        if obj["semantic_label"] == "floor"
    ]
    envelope = [
        obj for obj in objects.values()
        if obj["semantic_label"] in {"wall", "column", "roof", "vault"}
    ]
    if not floors or not envelope:
        return None

    floor_name, floor = max(floors, key=lambda item: _xy_area(item[1]["bounds"]))
    floor_bounds = floor["bounds"]
    floor_dims = floor_bounds["max"][:2] - floor_bounds["min"][:2]
    floor_area = _xy_area(floor_bounds)
    z_min = float(floor_bounds["max"][2])
    z_max = max(float(obj["bounds"]["max"][2]) for obj in envelope)
    height = max(0.0, z_max - z_min)

    return {
        "method": "floor_aabb_area_x_envelope_height",
        "floor_object": floor_name,
        "floor_base_dimensions": [float(floor_dims[0]), float(floor_dims[1])],
        "floor_base_area": floor_area,
        "lower_z": z_min,
        "upper_z": z_max,
        "height": height,
        "volume": floor_area * height,
    }


def _xy_area(bounds: dict) -> float:
    dims = bounds["max"][:2] - bounds["min"][:2]
    return float(max(dims[0], 0.0) * max(dims[1], 0.0))
