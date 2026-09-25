"""Render manually validated spatial triples as a publication-ready PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402


RELATION_COLORS = {
    "above": "#2563eb",
    "near": "#dc2626",
    "adjacent_to": "#159447",
}

CLASS_COLORS = {
    "column": "#2455ff",
    "door_window": "#00a9bd",
    "floor": "#22a447",
    "moldings": "#d81bce",
    "vault": "#8a2be2",
    "wall": "#ef3038",
}


def _read_triples(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _validated_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "manual_label" not in frame.columns:
        return frame.copy()
    labels = frame["manual_label"].astype(str).str.strip().str.lower()
    return frame[labels.isin({"correct", "valid", "true", "1", "yes", "si", "sì"})].copy()


def _canonical_edges(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    seen: set[tuple] = set()
    for row in frame.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)
        relation = str(row.relationship)
        if relation == "below":
            source, target, relation = target, source, "above"
        if relation in {"near", "adjacent_to"}:
            key = (tuple(sorted((source, target))), relation)
        else:
            key = (source, target, relation)
        if key in seen:
            continue
        seen.add(key)
        edges.append((source, target, relation))
    return edges


def render_spatial_triples(input_path: str, output_path: str) -> Path:
    source_path = Path(input_path)
    frame = _validated_rows(_read_triples(source_path))
    edges = _canonical_edges(frame)

    graph = nx.MultiDiGraph()
    node_classes: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        node_classes[str(row.source)] = str(row.source_label)
        node_classes[str(row.target)] = str(row.target_label)
    graph.add_nodes_from(node_classes)
    for source, target, relation in edges:
        graph.add_edge(source, target, relation=relation)

    position = nx.spring_layout(graph, seed=24, k=1.45, iterations=500)
    figure, axis = plt.subplots(figsize=(24, 16), facecolor="white")
    axis.set_facecolor("white")

    nx.draw_networkx_nodes(
        graph,
        position,
        node_size=2300,
        node_color="#f8fafc",
        edgecolors=[CLASS_COLORS.get(node_classes[node], "#475569") for node in graph],
        linewidths=2.2,
        node_shape="s",
        ax=axis,
    )
    nx.draw_networkx_labels(graph, position, font_size=8.5, font_weight="bold", ax=axis)

    for relation, color in RELATION_COLORS.items():
        relation_edges = [
            (source, target, key)
            for source, target, key, data in graph.edges(keys=True, data=True)
            if data["relation"] == relation
        ]
        nx.draw_networkx_edges(
            graph,
            position,
            edgelist=relation_edges,
            edge_color=color,
            width=1.15,
            alpha=0.62,
            arrows=relation == "above",
            arrowstyle="-|>",
            arrowsize=13,
            connectionstyle="arc3,rad=0.08",
            node_size=2300,
            ax=axis,
        )

    edge_labels = {
        (source, target, key): data["relation"]
        for source, target, key, data in graph.edges(keys=True, data=True)
    }
    nx.draw_networkx_edge_labels(
        graph,
        position,
        edge_labels=edge_labels,
        font_size=5.7,
        font_color="#111827",
        rotate=False,
        label_pos=0.5,
        bbox={"alpha": 0.78, "color": "white", "pad": 0.12, "linewidth": 0},
        ax=axis,
    )

    relation_legend = [
        Line2D([0], [0], color=color, lw=2.5, label=relation)
        for relation, color in RELATION_COLORS.items()
    ]
    class_legend = [
        Line2D(
            [0], [0], marker="s", color="white", markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=2, markersize=10, label=label,
        )
        for label, color in CLASS_COLORS.items()
        if label in set(node_classes.values())
    ]
    axis.legend(
        handles=relation_legend + class_legend,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        frameon=False,
        title="Relations and classes",
        fontsize=9,
    )
    axis.set_title(
        "scena4_VAL - manually validated spatial triples\n"
        "below is represented by the inverse above edge; symmetric relations are deduplicated",
        fontsize=17,
        pad=18,
    )
    axis.axis("off")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=250, bbox_inches="tight", pad_inches=0.18)
    plt.close(figure)
    print(
        f"Saved {output} ({graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} normalized edges)"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("triples_file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render_spatial_triples(args.triples_file, args.output)


if __name__ == "__main__":
    main()
