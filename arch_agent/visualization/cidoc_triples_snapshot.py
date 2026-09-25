"""Render CIDOC architectural triples in the spatial-triple diagram style."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PREDICATES = {
    "crm:P2_has_type": ("has type", "#2563eb"),
    "crm:P45_consists_of": ("consists of", "#e67e22"),
    "crm:P103_was_intended_for": ("intended for", "#159447"),
    "crm:P56_bears_feature": ("bears feature", "#dc2626"),
    "crm:P46i_forms_part_of": ("forms part of", "#8a2be2"),
}

CIDOC_CLASS_LABELS = {
    "crm:E22_Human-Made_Object": "crm:E22\nHuman-Made Object",
    "crm:E26_Physical_Feature": "crm:E26\nPhysical Feature",
    "crm:E55_Type": "crm:E55\nType",
    "crm:E57_Material": "crm:E57\nMaterial",
}

NODE_COLORS = {
    "crm:E22_Human-Made_Object": "#2455ff",
    "crm:E26_Physical_Feature": "#d81bce",
    "crm:E55_Type": "#159447",
    "crm:E57_Material": "#e67e22",
}

COLUMN_X = {
    "crm:E22_Human-Made_Object": -1.8,
    "crm:E26_Physical_Feature": -0.6,
    "crm:E55_Type": 0.7,
    "crm:E57_Material": 1.8,
}


def _scene_name_from_path(path: Path) -> str:
    match = re.search(r"(scena\d+_[A-Za-z0-9_]+?)(?:_cidoc|_threshold|$)", path.stem)
    return match.group(1) if match else path.stem


def _display_name(value: str) -> str:
    value = re.sub(r"^Scena\d+[A-Za-z0-9]*_", "", str(value))
    value = re.sub(r"^(Funzione|Materiale|Tipo)_", r"\1: ", value)
    value = value.replace("_", " ")
    value = re.sub(r"(?<=[a-zà-ù])(?=[A-Z])", " ", value)
    return "\n".join(textwrap.wrap(value, width=24))


def render_cidoc_triples(input_path: str, output_path: str) -> Path:
    scene_name = _scene_name_from_path(Path(input_path))
    columns = ["source", "predicate", "target", "source_class", "target_class"]
    frame = pd.read_csv(input_path, sep="\t", comment="#", names=columns)

    graph = nx.MultiDiGraph()
    node_classes: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        node_classes[str(row.source)] = str(row.source_class)
        node_classes[str(row.target)] = str(row.target_class)
        graph.add_edge(str(row.source), str(row.target), predicate=str(row.predicate))

    position = {}
    for cidoc_class, x_value in COLUMN_X.items():
        nodes = sorted(
            node for node, node_class in node_classes.items()
            if node_class == cidoc_class
        )
        y_values = np.linspace(1.0, -1.0, max(1, len(nodes)))
        position.update(
            {node: np.array([x_value, y]) for node, y in zip(nodes, y_values)}
        )
    figure, axis = plt.subplots(figsize=(28, 19), facecolor="white")
    axis.set_facecolor("white")

    nx.draw_networkx_nodes(
        graph,
        position,
        node_size=3000,
        node_color="#f8fafc",
        edgecolors=[NODE_COLORS.get(node_classes[node], "#64748b") for node in graph],
        linewidths=2.2,
        node_shape="s",
        ax=axis,
    )
    nx.draw_networkx_labels(
        graph,
        position,
        labels={node: _display_name(node) for node in graph},
        font_size=6.3,
        font_weight="bold",
        ax=axis,
    )

    for predicate, (label, color) in PREDICATES.items():
        edges = [
            (source, target, key)
            for source, target, key, data in graph.edges(keys=True, data=True)
            if data["predicate"] == predicate
        ]
        nx.draw_networkx_edges(
            graph,
            position,
            edgelist=edges,
            edge_color=color,
            width=1.2,
            alpha=0.65,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=13,
            connectionstyle="arc3,rad=0.06",
            node_size=3000,
            ax=axis,
        )

    edge_labels = {
        (source, target, key): PREDICATES.get(data["predicate"], (data["predicate"], ""))[0]
        for source, target, key, data in graph.edges(keys=True, data=True)
    }
    nx.draw_networkx_edge_labels(
        graph,
        position,
        edge_labels=edge_labels,
        font_size=4.8,
        rotate=False,
        bbox={"alpha": 0.78, "color": "white", "pad": 0.1, "linewidth": 0},
        ax=axis,
    )

    predicate_legend = [
        Line2D([0], [0], color=color, lw=2.5, label=label)
        for label, color in PREDICATES.values()
    ]
    class_legend = [
        Line2D(
            [0], [0], marker="s", color="white", markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=2, markersize=10,
            label=cidoc_class.replace("crm:", ""),
        )
        for cidoc_class, color in NODE_COLORS.items()
        if cidoc_class in set(node_classes.values())
    ]
    axis.legend(
        handles=predicate_legend + class_legend,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        frameon=False,
        title="CIDOC relations and classes",
        fontsize=9,
    )
    axis.set_title(
        f"{scene_name} - architectural relationships (CIDOC knowledge graph)",
        fontsize=18,
        pad=18,
    )
    axis.axis("off")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=250, bbox_inches="tight", pad_inches=0.2)
    plt.close(figure)
    print(
        f"Saved {output} ({graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges)"
    )
    return output


def render_cidoc_schema(input_path: str, output_path: str) -> Path:
    scene_name = _scene_name_from_path(Path(input_path))
    columns = ["source", "predicate", "target", "source_class", "target_class"]
    frame = pd.read_csv(input_path, sep="\t", comment="#", names=columns)
    combinations = (
        frame[["source_class", "predicate", "target_class"]]
        .drop_duplicates()
        .sort_values(["source_class", "predicate", "target_class"])
    )

    relation_patterns = len(combinations)
    figure, axis = plt.subplots(figsize=(16, 9), facecolor="white")
    axis.set_facecolor("white")
    centers = {
        "crm:E22_Human-Made_Object": (-3.2, 1.35),
        "crm:E26_Physical_Feature": (-3.2, -1.35),
        "crm:E55_Type": (3.2, 1.35),
        "crm:E57_Material": (3.2, -1.35),
    }
    for cidoc_class, (x, y) in centers.items():
        width, height = 2.15, 0.95
        node = FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.03",
            facecolor="#f8fafc",
            edgecolor=NODE_COLORS[cidoc_class],
            linewidth=3,
            zorder=5,
        )
        axis.add_patch(node)
        axis.text(
            x, y, CIDOC_CLASS_LABELS[cidoc_class], ha="center", va="center",
            fontsize=12, fontweight="bold", zorder=6,
        )

    arrows = [
        ((-2.1, 1.58), (2.1, 1.58), "crm:P2 has type", "#2563eb", 0.18, (0.0, 2.18)),
        ((-2.1, 1.18), (2.1, 1.18), "crm:P103 was intended for", "#159447", -0.14, (0.0, 0.72)),
        ((-2.25, -1.02), (2.3, 1.02), "crm:P2 has type", "#2563eb", 0.08, (-0.65, 0.05)),
        ((-2.15, -1.52), (2.15, 1.05), "crm:P103 was intended for", "#159447", -0.10, (0.75, -0.25)),
        ((-2.15, 1.05), (2.15, -1.05), "crm:P45 consists of", "#e67e22", 0.04, (0.65, 0.28)),
        ((-2.1, -1.35), (2.1, -1.35), "crm:P45 consists of", "#e67e22", 0.0, (0.0, -1.62)),
        ((-3.48, 0.86), (-3.48, -0.86), "crm:P56 bears feature", "#dc2626", 0.0, (-4.45, 0.18)),
        ((-2.92, -0.86), (-2.92, 0.86), "crm:P46i forms part of", "#8a2be2", 0.0, (-1.85, -0.18)),
    ]
    for start, end, label, color, curvature, label_position in arrows:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=2.8,
            color=color,
            connectionstyle=f"arc3,rad={curvature}",
            zorder=2,
        )
        axis.add_patch(arrow)
        axis.text(
            *label_position,
            label,
            color=color,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.96, "pad": 2},
            zorder=7,
        )

    axis.set_title(
        f"{scene_name} - CIDOC CRM architectural relationship schema",
        fontsize=18,
        pad=20,
    )
    axis.text(
        0.5,
        0.02,
        f"Aggregated from the {len(frame)} CIDOC triples generated from scene annotations",
        transform=axis.transAxes,
        ha="center",
        fontsize=10,
        color="#475569",
    )
    axis.set_xlim(-5.2, 5.2)
    axis.set_ylim(-2.65, 2.65)
    axis.axis("off")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=250, bbox_inches="tight", pad_inches=0.25)
    plt.close(figure)
    print(
        f"Saved {output} (4 CIDOC classes, {relation_patterns} relation patterns)"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("triples_file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--schematic", action="store_true")
    args = parser.parse_args()
    if args.schematic:
        render_cidoc_schema(args.triples_file, args.output)
    else:
        render_cidoc_triples(args.triples_file, args.output)


if __name__ == "__main__":
    main()
