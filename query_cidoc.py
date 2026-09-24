"""Standalone CIDOC knowledge-graph query utility."""

import argparse

from arch_agent.pipeline.pipeline import PipelineParams, run_pipeline
from arch_agent.tools.cidoc_tools import create_cidoc_tools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and query the CIDOC graph without starting the LLM agent."
    )
    parser.add_argument("point_cloud_path")
    parser.add_argument("--annotation-csv", required=True)
    parser.add_argument("--distance-threshold", type=float, default=2.0)
    parser.add_argument("--object-name")
    parser.add_argument("--semantic-label")
    parser.add_argument("--predicate")
    parser.add_argument(
        "--direction",
        choices=("outgoing", "incoming", "both"),
        default="both",
    )
    parser.add_argument("--limit", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = run_pipeline(
        PipelineParams(
            point_cloud_path=args.point_cloud_path,
            annotation_csv_path=args.annotation_csv,
            distance_threshold=args.distance_threshold,
        )
    )
    query_tool = create_cidoc_tools(ctx)[0]
    result = query_tool.invoke(
        {
            "object_name": args.object_name,
            "semantic_label": args.semantic_label,
            "predicate": args.predicate,
            "direction": args.direction,
            "limit": args.limit,
        }
    )
    print(result)


if __name__ == "__main__":
    main()
