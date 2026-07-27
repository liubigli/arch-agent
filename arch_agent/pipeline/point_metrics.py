from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


RGB_COLUMNS = ("R", "G", "B")
XYZ_COLUMNS = ("x", "y", "z")


def has_rgb(df) -> bool:
    return df is not None and all(column in df.columns for column in RGB_COLUMNS)


def rgb_statistics(df, bin_size: int = 32) -> dict | None:
    if not has_rgb(df) or df.empty:
        return None

    values = df[list(RGB_COLUMNS)].dropna().to_numpy(float)
    if values.size == 0:
        return None

    max_channel = max(float(np.nanmax(values)), 1.0)
    divisor = 257.0 if max_channel > 255 else 1.0
    rgb8_float = np.clip(values / divisor, 0, 255)
    rgb8 = np.rint(rgb8_float).astype(np.int16)

    bin_size = max(1, min(int(bin_size), 256))
    quantized = np.clip(
        (rgb8 // bin_size) * bin_size + bin_size // 2,
        0,
        255,
    ).astype(np.int16)
    unique, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant_index = int(np.argmax(counts))
    intensity = rgb8_float.mean(axis=1)

    return {
        "point_count": int(values.shape[0]),
        "mean_raw": tuple(float(value) for value in values.mean(axis=0)),
        "mean_rgb8": tuple(int(round(value)) for value in rgb8_float.mean(axis=0)),
        "std_rgb8": tuple(float(value) for value in rgb8_float.std(axis=0)),
        "min_rgb8": tuple(int(value) for value in rgb8.min(axis=0)),
        "max_rgb8": tuple(int(value) for value in rgb8.max(axis=0)),
        "dominant_rgb8": tuple(int(value) for value in unique[dominant_index]),
        "dominant_percent": float(counts[dominant_index] / len(rgb8) * 100.0),
        "intensity_mean": float(intensity.mean()),
        "intensity_std": float(intensity.std()),
        "bin_size": int(bin_size),
        "divisor": float(divisor),
    }


def format_rgb_summary(name: str, df, language: str = "en") -> str:
    stats = rgb_statistics(df)
    if stats is None:
        if language == "it":
            return f"I valori RGB non sono disponibili per {name}."
        return f"RGB values are not available for {name}."

    raw = stats["mean_raw"]
    mean = stats["mean_rgb8"]
    std = stats["std_rgb8"]
    min_rgb = stats["min_rgb8"]
    max_rgb = stats["max_rgb8"]
    dominant = stats["dominant_rgb8"]

    if language == "it":
        return "\n".join([
            f"Riassunto colore per {name}:",
            f"  Punti analizzati: {stats['point_count']:,}",
            f"  RGB medio raw: ({raw[0]:.1f}, {raw[1]:.1f}, {raw[2]:.1f})",
            f"  RGB medio 8-bit: ({mean[0]}, {mean[1]}, {mean[2]})",
            "  Deviazione standard RGB 8-bit: "
            f"({std[0]:.1f}, {std[1]:.1f}, {std[2]:.1f})",
            "  Range RGB 8-bit: "
            f"R {min_rgb[0]}-{max_rgb[0]}, "
            f"G {min_rgb[1]}-{max_rgb[1]}, "
            f"B {min_rgb[2]}-{max_rgb[2]}",
            "  Colore dominante quantizzato: "
            f"({dominant[0]}, {dominant[1]}, {dominant[2]}) "
            f"({stats['dominant_percent']:.1f}% dei punti, bin={stats['bin_size']})",
            "  Intensita media 8-bit: "
            f"{stats['intensity_mean']:.1f} +/- {stats['intensity_std']:.1f}",
        ])

    return "\n".join([
        f"Color summary for {name}:",
        f"  Points analyzed: {stats['point_count']:,}",
        f"  Mean RGB raw: ({raw[0]:.1f}, {raw[1]:.1f}, {raw[2]:.1f})",
        f"  Mean RGB 8-bit: ({mean[0]}, {mean[1]}, {mean[2]})",
        f"  RGB 8-bit standard deviation: ({std[0]:.1f}, {std[1]:.1f}, {std[2]:.1f})",
        "  RGB 8-bit range: "
        f"R {min_rgb[0]}-{max_rgb[0]}, "
        f"G {min_rgb[1]}-{max_rgb[1]}, "
        f"B {min_rgb[2]}-{max_rgb[2]}",
        "  Quantized dominant color: "
        f"({dominant[0]}, {dominant[1]}, {dominant[2]}) "
        f"({stats['dominant_percent']:.1f}% of points, bin={stats['bin_size']})",
        "  Mean 8-bit intensity: "
        f"{stats['intensity_mean']:.1f} +/- {stats['intensity_std']:.1f}",
    ])


def roughness_statistics(
    df,
    sample_size: int = 5_000,
    k_neighbors: int = 24,
    seed: int = 1,
) -> dict:
    if df is None or df.empty:
        return {"available": False, "reason": "no point-cloud dataframe is available"}
    if not all(column in df.columns for column in XYZ_COLUMNS):
        return {"available": False, "reason": "x/y/z columns are missing"}

    coords = df[list(XYZ_COLUMNS)].dropna().to_numpy(float)
    point_count = int(coords.shape[0])
    if point_count < 6:
        return {
            "available": False,
            "reason": "at least 6 points are required for local plane fitting",
        }

    sample_size = max(1, int(sample_size))
    sample_count = min(point_count, sample_size)
    if sample_count < point_count:
        rng = np.random.default_rng(seed)
        sample_indices = np.sort(
            rng.choice(point_count, size=sample_count, replace=False)
        )
        sampled_coords = coords[sample_indices]
    else:
        sampled_coords = coords

    k = min(max(int(k_neighbors), 6), point_count)
    tree = cKDTree(coords)
    _, neighbor_indices = tree.query(sampled_coords, k=k)
    if k == 1:
        neighbor_indices = neighbor_indices[:, np.newaxis]

    residuals = []
    variations = []
    for indices in neighbor_indices:
        local = coords[indices]
        centered = local - local.mean(axis=0)
        covariance = centered.T @ centered / max(len(local) - 1, 1)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        total = float(eigenvalues.sum())
        residuals.append(float(eigenvalues[0] ** 0.5))
        variations.append(float(eigenvalues[0] / total) if total > 0 else 0.0)

    residuals_array = np.asarray(residuals, dtype=float)
    variations_array = np.asarray(variations, dtype=float)
    mean_residual = float(residuals_array.mean())

    return {
        "available": True,
        "point_count": point_count,
        "sample_count": int(sample_count),
        "k_neighbors": int(k),
        "mean_residual_m": mean_residual,
        "median_residual_m": float(np.median(residuals_array)),
        "std_residual_m": float(residuals_array.std()),
        "p95_residual_m": float(np.percentile(residuals_array, 95)),
        "max_residual_m": float(residuals_array.max()),
        "mean_surface_variation": float(variations_array.mean()),
        "median_surface_variation": float(np.median(variations_array)),
        "roughness_level": _roughness_level(mean_residual),
    }


def format_roughness_summary(
    name: str,
    df,
    sample_size: int = 5_000,
    k_neighbors: int = 24,
    language: str = "en",
) -> str:
    stats = roughness_statistics(
        df,
        sample_size=sample_size,
        k_neighbors=k_neighbors,
    )
    if not stats.get("available"):
        if language == "it":
            return f"Rugosità non disponibile per {name}: {stats['reason']}."
        return f"Surface roughness is not available for {name}: {stats['reason']}."

    level = _roughness_level_label(stats["roughness_level"], language)
    if language == "it":
        return "\n".join([
            f"Rugosità superficiale stimata per {name}:",
            "  Metodo: PCA locale su vicini k-nearest; misura dello scarto dal piano locale.",
            f"  Punti disponibili: {stats['point_count']:,}",
            f"  Punti campionati: {stats['sample_count']:,}",
            f"  Vicini per punto: {stats['k_neighbors']}",
            f"  Scarto medio dal piano locale: {stats['mean_residual_m']:.4f} m",
            f"  Scarto mediano: {stats['median_residual_m']:.4f} m",
            f"  Deviazione standard: {stats['std_residual_m']:.4f} m",
            f"  95 percentile: {stats['p95_residual_m']:.4f} m",
            f"  Scarto massimo: {stats['max_residual_m']:.4f} m",
            f"  Surface variation media: {stats['mean_surface_variation']:.6f}",
            f"  Livello qualitativo: {level}",
            "  Nota: la metrica può includere rumore, curvatura e segmentazione, non solo rugosità materica.",
        ])

    return "\n".join([
        f"Estimated surface roughness for {name}:",
        "  Method: local PCA over k-nearest neighbors; residual from the local best-fit plane.",
        f"  Available points: {stats['point_count']:,}",
        f"  Sampled points: {stats['sample_count']:,}",
        f"  Neighbors per point: {stats['k_neighbors']}",
        f"  Mean local-plane residual: {stats['mean_residual_m']:.4f} m",
        f"  Median residual: {stats['median_residual_m']:.4f} m",
        f"  Standard deviation: {stats['std_residual_m']:.4f} m",
        f"  95th percentile: {stats['p95_residual_m']:.4f} m",
        f"  Maximum residual: {stats['max_residual_m']:.4f} m",
        f"  Mean surface variation: {stats['mean_surface_variation']:.6f}",
        f"  Qualitative level: {level}",
        "  Note: the metric can include noise, curvature, and segmentation effects; it is not material identification.",
    ])


def _roughness_level(mean_residual_m: float) -> str:
    if mean_residual_m < 0.005:
        return "low"
    if mean_residual_m < 0.02:
        return "moderate"
    if mean_residual_m < 0.05:
        return "high"
    return "very_high"


def _roughness_level_label(level: str, language: str) -> str:
    if language == "it":
        return {
            "low": "bassa",
            "moderate": "moderata",
            "high": "alta",
            "very_high": "molto alta",
        }.get(level, level)
    return level.replace("_", " ")
