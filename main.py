"""
Usage:
    python main.py
    python main.py path/to/scene.laz
    python main.py --eps 0.3 --min-samples 10
    python main.py --distance-threshold 5.0 --use-normals
    python main.py --model llama3.1
"""

import argparse
from pathlib import Path

from arch_agent.pipeline.pipeline import PipelineParams, run_pipeline
from arch_agent.agent import run_agent


DEFAULT_POINT_CLOUD_PATH = (
    r"C:\Users\Utente\Desktop\Lucrezia\Lu_test_project"
    r"\laz_archdataset_palette_originale"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Architectural Scene Agent — LLM-powered analysis of semantic point clouds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "point_cloud_path",
        nargs="?",
        default=DEFAULT_POINT_CLOUD_PATH,
        help=(
            "Input LAZ file or directory. Labels are read from a "
            "'semantic_label' extra dimension or from the standard "
            "'classification' dimension."
        ),
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


def select_point_cloud(path_value: str) -> str:
    path = Path(path_value)
    if path.is_file():
        return str(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Point cloud path not found: {path}")

    laz_files = sorted(
        file for file in path.iterdir()
        if file.is_file() and file.suffix.lower() == ".laz"
    )
    if not laz_files:
        raise FileNotFoundError(f"No .laz files found in directory: {path}")

    print("\nSelect a LAZ point cloud:")
    for index, file_path in enumerate(laz_files, start=1):
        print(f"  {index}. {file_path.name}")

    while True:
        choice = input(f"File number [1-{len(laz_files)}]: ").strip()
        if choice.isdigit():
            selected_index = int(choice)
            if 1 <= selected_index <= len(laz_files):
                return str(laz_files[selected_index - 1])

        print("Invalid selection. Enter one of the listed numbers.")


def main() -> None:
    args = parse_args()
    point_cloud_path = select_point_cloud(args.point_cloud_path)

    params = PipelineParams(
        point_cloud_path=point_cloud_path,
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
