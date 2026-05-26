"""
Usage:
    python main.py path/to/scene.csv
    python main.py path/to/scene.csv --eps 0.3 --min-samples 10
    python main.py path/to/scene.csv --distance-threshold 5.0 --use-normals
    python main.py path/to/scene.csv --model llama3.1
"""

import argparse
import sys

from arch_agent.pipeline.pipeline import PipelineParams, run_pipeline
from arch_agent.agent import run_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Architectural Scene Agent — LLM-powered analysis of semantic point clouds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv_path",
        help="Input CSV file (columns: x;y;z;R;G;B;nx;ny;nz;semantic_label)",
    )

    # Pipeline params
    group = parser.add_argument_group("pipeline parameters")
    group.add_argument(
        "--eps", type=float, default=0.5,
        help="DBSCAN epsilon for object segmentation (smaller = tighter clusters)",
    )
    group.add_argument(
        "--min-samples", type=int, default=15,
        help="DBSCAN min_samples (reduce to 10 for sparse clouds)",
    )
    group.add_argument(
        "--distance-threshold", type=float, default=3.0,
        help="Max centroid distance (m) to consider two objects spatially related",
    )
    group.add_argument(
        "--sample-n", type=int, default=150_000,
        help="Max points to load (0 = no limit)",
    )
    group.add_argument(
        "--use-normals", action="store_true",
        help="Use Poisson reconstruction for surface area (slower, more accurate)",
    )

    # Agent params
    group2 = parser.add_argument_group("agent parameters")
    group2.add_argument(
        "--model", default="llama3",
        help="Ollama model name (must be pulled via 'ollama pull <model>')",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    params = PipelineParams(
        csv_path=args.csv_path,
        sample_n=args.sample_n if args.sample_n > 0 else None,
        eps=args.eps,
        min_samples=args.min_samples,
        distance_threshold=args.distance_threshold,
        use_normals=args.use_normals,
    )

    ctx = run_pipeline(params)
    run_agent(ctx, model=args.model)


if __name__ == "__main__":
    main()
