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
            return f"Rugosita non disponibile per {name}: {stats['reason']}."
        return f"Surface roughness is not available for {name}: {stats['reason']}."

    level = _roughness_level_label(stats["roughness_level"], language)
    if language == "it":
        return "\n".join([
            f"Rugosita superficiale stimata per {name}:",
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
            "  Nota: la metrica puo includere rumore, curvatura e segmentazione, non solo rugosita materica.",
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
        "  Note: the metric can include noise, curvature, and segmentation effects, not only material roughness.",
    ])


def material_hypotheses(
    df,
    semantic_label: str | None = None,
    sample_size: int = 3_000,
    k_neighbors: int = 24,
) -> dict:
    color = rgb_statistics(df)
    roughness = roughness_statistics(
        df,
        sample_size=sample_size,
        k_neighbors=k_neighbors,
    )
    priors = _semantic_material_priors(semantic_label)
    candidates: dict[str, dict] = {}

    if color is None and not roughness.get("available") and not priors:
        return {
            "available": False,
            "reason": "RGB, roughness, and semantic material priors are not available",
        }

    evidence = []
    if semantic_label:
        evidence.append(f"semantic_label={semantic_label}")
    if color is not None:
        family = _color_family(color["mean_rgb8"])
        color_strength = _color_strength(color["mean_rgb8"])
        evidence.append(
            "mean_rgb8="
            f"({color['mean_rgb8'][0]}, {color['mean_rgb8'][1]}, {color['mean_rgb8'][2]})"
        )
        evidence.append(f"color_family={family}")
        evidence.append(f"color_strength={color_strength}")
        _add_color_material_candidates(
            candidates,
            family,
            semantic_label,
            color_strength=color_strength,
        )
    else:
        family = "unknown"
        color_strength = "none"
        evidence.append("RGB unavailable")

    if roughness.get("available"):
        roughness_level = roughness["roughness_level"]
        evidence.append(f"roughness={roughness_level}")
        _adjust_candidates_by_roughness(candidates, roughness_level)
    else:
        roughness_level = "unknown"
        evidence.append("roughness unavailable")

    for material in priors:
        _add_candidate(
            candidates,
            material,
            0.25,
            "semantic prior for architectural class",
        )

    if not candidates:
        _add_candidate(
            candidates,
            "unknown material",
            0.1,
            "insufficient color/material evidence",
        )

    sorted_candidates = sorted(
        candidates.values(),
        key=lambda item: item["score"],
        reverse=True,
    )
    top_score = sorted_candidates[0]["score"]
    confidence = _material_confidence(
        top_score,
        color is not None,
        roughness_level,
        color_strength,
    )

    return {
        "available": True,
        "semantic_label": semantic_label,
        "color": color,
        "color_family": family,
        "color_strength": color_strength,
        "roughness": roughness,
        "candidates": sorted_candidates[:5],
        "confidence": confidence,
        "evidence": evidence,
    }


def format_material_summary(
    name: str,
    df,
    semantic_label: str | None = None,
    sample_size: int = 3_000,
    k_neighbors: int = 24,
    language: str = "en",
) -> str:
    result = material_hypotheses(
        df,
        semantic_label=semantic_label,
        sample_size=sample_size,
        k_neighbors=k_neighbors,
    )
    if not result.get("available"):
        if language == "it":
            return f"Materiale non inferibile per {name}: {result['reason']}."
        return f"Material cannot be inferred for {name}: {result['reason']}."

    color = result["color"]
    roughness = result["roughness"]
    candidates = result["candidates"]

    if color is None:
        color_line_it = "Colore: RGB non disponibile"
        color_line_en = "Color: RGB unavailable"
    else:
        rgb = color["mean_rgb8"]
        dominant = color["dominant_rgb8"]
        color_line_it = (
            f"Colore: RGB medio 8-bit ({rgb[0]}, {rgb[1]}, {rgb[2]}), "
            f"dominante quantizzato ({dominant[0]}, {dominant[1]}, {dominant[2]})"
        )
        color_line_en = (
            f"Color: mean 8-bit RGB ({rgb[0]}, {rgb[1]}, {rgb[2]}), "
            f"quantized dominant ({dominant[0]}, {dominant[1]}, {dominant[2]})"
        )

    if roughness.get("available"):
        roughness_line_it = (
            "Rugosita: "
            f"{_roughness_level_label(roughness['roughness_level'], 'it')} "
            f"(scarto medio {roughness['mean_residual_m']:.4f} m)"
        )
        roughness_line_en = (
            "Roughness: "
            f"{_roughness_level_label(roughness['roughness_level'], 'en')} "
            f"(mean residual {roughness['mean_residual_m']:.4f} m)"
        )
    else:
        roughness_line_it = "Rugosita: non disponibile"
        roughness_line_en = "Roughness: unavailable"

    if language == "it":
        lines = [
            f"Inferenza materiale per {name}:",
            f"  Classe semantica: {semantic_label or 'non specificata'}",
            f"  {color_line_it}",
            f"  Famiglia colore: {result['color_family']} ({_color_strength_label(result['color_strength'], language)})",
            f"  {roughness_line_it}",
            "  Materiali candidati:",
        ]
        lines.extend(
            f"    - {item['material']}: score={item['score']:.2f}; "
            f"indizi={'; '.join(item['reasons'][:3])}"
            for item in candidates
        )
        lines.extend([
            f"  Confidenza: {_confidence_label(result['confidence'], language)}",
            "  Nota: e una classificazione euristica basata su colore, classe e rugosita; "
            "non sostituisce analisi materiche o radiometriche calibrate.",
        ])
        return "\n".join(lines)

    lines = [
        f"Material inference for {name}:",
        f"  Semantic class: {semantic_label or 'not specified'}",
        f"  {color_line_en}",
        f"  Color family: {result['color_family']} ({_color_strength_label(result['color_strength'], language)})",
        f"  {roughness_line_en}",
        "  Candidate materials:",
    ]
    lines.extend(
        f"    - {item['material']}: score={item['score']:.2f}; "
        f"evidence={'; '.join(item['reasons'][:3])}"
        for item in candidates
    )
    lines.extend([
        f"  Confidence: {_confidence_label(result['confidence'], language)}",
        "  Note: this is a heuristic classification based on color, class, and roughness; "
        "it does not replace calibrated material or radiometric analysis.",
    ])
    return "\n".join(lines)


def _roughness_level(mean_residual_m: float) -> str:
    if mean_residual_m < 0.005:
        return "low"
    if mean_residual_m < 0.02:
        return "moderate"
    if mean_residual_m < 0.05:
        return "high"
    return "very_high"


def _semantic_material_priors(semantic_label: str | None) -> list[str]:
    priors = {
        "arch": ["stone", "brick masonry", "concrete"],
        "column": ["stone", "marble/limestone", "brick masonry", "concrete"],
        "moldings": ["plaster/stucco", "stone", "terracotta"],
        "floor": ["stone paving", "brick/tile", "concrete", "wood"],
        "door_window": ["glass", "wood", "metal"],
        "wall": ["stone", "brick masonry", "plaster", "concrete"],
        "stairs": ["stone", "brick/tile", "concrete", "wood"],
        "vault": ["stone", "brick masonry", "plaster"],
        "roof": ["tile/terracotta", "stone", "wood", "metal"],
    }
    return priors.get(semantic_label or "", [])


def _add_color_material_candidates(
    candidates: dict[str, dict],
    family: str,
    semantic_label: str | None,
    color_strength: str = "medium",
) -> None:
    structural_labels = {"arch", "column", "wall", "vault", "roof", "stairs", "floor"}
    is_structural = semantic_label in structural_labels
    strength_weight = {
        "none": 0.0,
        "weak": 0.35,
        "medium": 0.7,
        "strong": 1.0,
    }.get(color_strength, 0.7)

    if family in {"red_orange", "terracotta", "brown_orange", "ochre"}:
        _add_candidate(candidates, "brick masonry", 0.55 * strength_weight, "reddish/orange RGB")
        _add_candidate(candidates, "terracotta/tile", 0.45 * strength_weight, "reddish/orange RGB")
        if semantic_label == "roof":
            _add_candidate(candidates, "tile/terracotta", 0.65 * strength_weight, "roof class with warm red/orange color")
    elif family == "brown":
        _add_candidate(candidates, "wood", 0.45 * strength_weight, "brown RGB")
        _add_candidate(candidates, "weathered stone", 0.35 * strength_weight, "brown/earthy RGB")
        if semantic_label in {"floor", "stairs", "door_window", "roof"}:
            _add_candidate(candidates, "wood", 0.25 * strength_weight, "class often compatible with timber")
    elif family in {"white", "light_gray"}:
        _add_candidate(candidates, "plaster/stucco", 0.45, "light neutral RGB")
        _add_candidate(candidates, "marble/limestone", 0.45, "light neutral RGB")
        if is_structural:
            _add_candidate(candidates, "stone", 0.25, "structural class with light neutral color")
    elif family in {"gray", "dark_gray", "muted_dark", "muted_gray"}:
        neutral_weight = 0.6 if family in {"muted_dark", "muted_gray"} else 1.0
        _add_candidate(candidates, "stone", 0.5 * neutral_weight, "gray neutral RGB")
        _add_candidate(candidates, "concrete", 0.4 * neutral_weight, "gray neutral RGB")
        if family in {"muted_dark", "muted_gray"}:
            _add_candidate(candidates, "weathered stone", 0.25, "muted low-saturation RGB")
        if semantic_label == "door_window":
            _add_candidate(candidates, "metal", 0.25, "gray RGB on opening class")
    elif family in {"blue", "cyan"}:
        if semantic_label == "door_window":
            _add_candidate(candidates, "glass", 0.6, "blue/cyan RGB on opening class")
        _add_candidate(candidates, "painted surface", 0.25, "blue/cyan RGB")
    elif family == "green":
        _add_candidate(candidates, "biological patina/vegetation trace", 0.4, "green RGB")
        _add_candidate(candidates, "painted surface", 0.25, "green RGB")
    elif family == "black":
        _add_candidate(candidates, "dark stone", 0.35, "dark neutral RGB")
        _add_candidate(candidates, "metal", 0.3, "dark neutral RGB")


def _adjust_candidates_by_roughness(candidates: dict[str, dict], level: str) -> None:
    if level == "low":
        for material in ("glass", "metal", "marble/limestone", "plaster/stucco"):
            _add_candidate(candidates, material, 0.15, "low local roughness")
    elif level == "moderate":
        for material in ("stone", "plaster/stucco", "concrete", "wood"):
            _add_candidate(candidates, material, 0.1, "moderate local roughness")
    elif level == "high":
        for material in ("stone", "brick masonry", "weathered stone"):
            _add_candidate(candidates, material, 0.15, "high local roughness")
    elif level == "very_high":
        for material in ("rough stone", "brick masonry", "degraded plaster"):
            _add_candidate(candidates, material, 0.2, "very high local roughness")


def _add_candidate(
    candidates: dict[str, dict],
    material: str,
    score: float,
    reason: str,
) -> None:
    entry = candidates.setdefault(
        material,
        {"material": material, "score": 0.0, "reasons": []},
    )
    entry["score"] = min(1.0, float(entry["score"]) + float(score))
    if reason not in entry["reasons"]:
        entry["reasons"].append(reason)


def _material_confidence(
    score: float,
    has_color_data: bool,
    roughness_level: str,
    color_strength: str,
) -> str:
    if not has_color_data:
        return "low"
    if color_strength == "weak":
        return "medium" if score >= 0.65 and roughness_level != "unknown" else "low"
    if score >= 0.8 and roughness_level != "unknown":
        return "medium-high"
    if score >= 0.55:
        return "medium"
    return "low"


def _confidence_label(level: str, language: str) -> str:
    if language == "it":
        return {
            "low": "bassa",
            "medium": "media",
            "medium-high": "media-alta",
        }.get(level, level)
    return level


def _color_family(rgb: tuple[int, int, int]) -> str:
    r, g, b = [float(value) for value in rgb]
    h, s, v = _rgb_to_hsv(r, g, b)
    chroma = max(r, g, b) - min(r, g, b)

    if v < 35:
        return "black"
    if s < 0.25 or chroma < 24:
        if v > 205:
            return "white"
        if v > 145:
            return "light_gray"
        if v > 95:
            return "muted_gray"
        if v > 70:
            return "muted_dark"
        return "dark_gray"
    if 0 <= h < 18 or h >= 345:
        return "red_orange"
    if 18 <= h < 38:
        return "terracotta"
    if 38 <= h < 65:
        return "brown_orange" if v < 165 else "ochre"
    if 65 <= h < 165:
        return "green"
    if 165 <= h < 205:
        return "cyan"
    if 205 <= h < 260:
        return "blue"
    if 260 <= h < 345:
        return "violet"
    if v < 120:
        return "brown"
    return "mixed"


def _color_strength(rgb: tuple[int, int, int]) -> str:
    r, g, b = [float(value) for value in rgb]
    _, saturation, value = _rgb_to_hsv(r, g, b)
    chroma = max(r, g, b) - min(r, g, b)
    if saturation < 0.18 or chroma < 18:
        return "weak"
    if saturation < 0.35 or chroma < 45 or value < 90:
        return "medium"
    return "strong"


def _color_strength_label(strength: str, language: str) -> str:
    if language == "it":
        return {
            "none": "colore non disponibile",
            "weak": "indizio cromatico debole",
            "medium": "indizio cromatico medio",
            "strong": "indizio cromatico forte",
        }.get(strength, strength)
    return {
        "none": "color unavailable",
        "weak": "weak color cue",
        "medium": "medium color cue",
        "strong": "strong color cue",
    }.get(strength, strength)


def _rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    max_value = max(r_norm, g_norm, b_norm)
    min_value = min(r_norm, g_norm, b_norm)
    delta = max_value - min_value

    if delta == 0:
        hue = 0.0
    elif max_value == r_norm:
        hue = (60.0 * ((g_norm - b_norm) / delta) + 360.0) % 360.0
    elif max_value == g_norm:
        hue = 60.0 * ((b_norm - r_norm) / delta + 2.0)
    else:
        hue = 60.0 * ((r_norm - g_norm) / delta + 4.0)

    saturation = 0.0 if max_value == 0 else delta / max_value
    value = max_value * 255.0
    return hue, saturation, value


def _roughness_level_label(level: str, language: str) -> str:
    if language == "it":
        return {
            "low": "bassa",
            "moderate": "moderata",
            "high": "alta",
            "very_high": "molto alta",
        }.get(level, level)
    return level.replace("_", " ")
