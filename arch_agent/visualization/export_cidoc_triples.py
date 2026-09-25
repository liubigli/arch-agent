"""Build and export CIDOC triples for visualization."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from arch_agent.pipeline.pipeline import PipelineParams, run_pipeline
from arch_agent.visualization.graph_viewer import build_l3_graph_from_context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("point_cloud_path")
    parser.add_argument("--annotation-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--distance-threshold", type=float, default=2.0)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--sample-n", type=int, default=150_000)
    args = parser.parse_args()

    context = run_pipeline(
        PipelineParams(
            point_cloud_path=args.point_cloud_path,
            annotation_csv_path=args.annotation_csv,
            distance_threshold=args.distance_threshold,
            eps=args.eps,
            min_samples=args.min_samples,
            sample_n=args.sample_n,
        )
    )
    graph = build_l3_graph_from_context(context)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# CIDOC/KG triples - {Path(args.point_cloud_path).stem}\n")
        handle.write("# source\tpredicate\ttarget\tsource_class\ttarget_class\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for source, target, data in graph.edges(data=True):
            writer.writerow(
                [
                    source,
                    data.get("predicate", data.get("relationship", "")),
                    target,
                    graph.nodes[source].get("cidoc_class", ""),
                    graph.nodes[target].get("cidoc_class", ""),
                ]
            )
    print(f"Saved {graph.number_of_edges()} CIDOC triples: {output}")


if __name__ == "__main__":
    main()
