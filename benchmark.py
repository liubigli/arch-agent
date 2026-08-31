"""
Usage:
    python benchmark.py path/to/scene.laz --questions-file tmp/domande_per_scene.txt
    python benchmark.py path/to/scene.laz --model llama3.1 --limit 20
    python benchmark.py path/to/scene.laz --models llama3.1 qwen3:30b mistral-small3.2
    python benchmark.py path/to/scene.laz --model llama3.1 --output-dir results

For each model, the runner writes raw, evaluation, and manual-review JSON/CSV
files named as benchmark_<kind>_<scene>_<model>_<YYYYMMDD>_test_<n>.
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

from arch_agent.pipeline.pipeline import PipelineParams, run_pipeline
from arch_agent.benchmark.harness import (
    evaluate_benchmark,
    load_questions,
    manual_review_records,
    run_benchmark,
    write_evaluation_report,
    write_manual_review_report,
    write_raw_report,
)
from main import DEFAULT_POINT_CLOUD_PATH, parse_think_override, resolve_local_path, select_point_cloud

EXPECTED_QUESTION_COUNT = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark tool-calling reliability of the Architectural Scene Agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "point_cloud_path",
        nargs="?",
        default=DEFAULT_POINT_CLOUD_PATH,
        help="Input LAZ file or directory, same as main.py.",
    )

    group = parser.add_argument_group("pipeline parameters")
    group.add_argument("--eps", type=float, default=0.5)
    group.add_argument("--min-samples", type=int, default=15)
    group.add_argument("--distance-threshold", type=float, default=3.0)
    group.add_argument("--sample-n", type=int, default=150_000)
    group.add_argument("--use-normals", action="store_true")
    group.add_argument("--annotation-csv", default=None)
    group.add_argument("--annotation-match-threshold", type=float, default=2.0)

    group3 = parser.add_argument_group("visualization parameters")
    group3.add_argument(
        "--plot-segmentation", action="store_true",
        help=(
            "Open an Open3D window showing the segmented (DBSCAN) point "
            "cloud objects once the scene graph has been built."
        ),
    )

    group2 = parser.add_argument_group("benchmark parameters")
    group2.add_argument(
        "--model", default="llama3",
        help="Ollama model name (must be pulled via 'ollama pull <model>')",
    )
    group2.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Optional list of Ollama models to benchmark sequentially. "
            "Accepts space-separated values or comma-separated groups. "
            "When set, this overrides --model."
        ),
    )
    group2.add_argument(
        "--questions-file", default="benchmark/domande_per_scene.txt",
        help="Plain-text file with one question per line (lines must end with '?').",
    )
    group2.add_argument(
        "--limit", type=int, default=0,
        help="Max number of questions to run (0 = no limit).",
    )
    group2.add_argument(
        "--capture-reasoning", action="store_true",
        help=(
            "Add a preliminary, tool-unbound reasoning step before each tool "
            "call so the model's chain-of-thought can be logged. Roughly "
            "doubles LLM calls per question."
        ),
    )
    group2.add_argument(
        "--think",
        choices=("auto", "true", "false"),
        default="auto",
        help=(
            "Override Ollama thinking mode. Use 'auto' for the model profile, "
            "'true' to force thinking, or 'false' to disable it."
        ),
    )
    group2.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for benchmark reports. Default: benchmark_results. "
            "Files are named automatically with kind, scene, model, date, and test number."
        ),
    )
    group2.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Deprecated compatibility option. When provided without "
            "--output-dir, its parent directory is used as output directory."
        ),
    )

    return parser.parse_args()


def _sanitize_for_path(text: str) -> str:
    return "".join(char if char.isalnum() or char in ".-_" else "_" for char in text)


def _sanitize_model_for_path(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return sanitized or "model"


def _date_for_path() -> str:
    return datetime.now().strftime("%Y%m%d")


def _models_from_args(args: argparse.Namespace) -> list[str]:
    raw_models = args.models if args.models else [args.model]
    models: list[str] = []
    for item in raw_models:
        for model in str(item).split(","):
            model = model.strip()
            if model:
                models.append(model)
    return models


def _model_name_for_report(model: str, think_mode: str) -> str:
    model_name = _sanitize_model_for_path(model)
    if think_mode != "auto":
        model_name = f"{model_name}_think_{think_mode}"
    return model_name


def _output_dir_from_args(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    if args.output_prefix:
        return Path(args.output_prefix).parent
    return Path("benchmark_results")


def _next_test_number(
    output_dir: Path,
    scene_name: str,
    model_name: str,
    date: str,
) -> int:
    pattern = f"benchmark_raw_{scene_name}_{model_name}_{date}_test_*.json"
    test_numbers: list[int] = []
    for path in output_dir.glob(pattern):
        match = re.search(r"_test_(\d+)\.json$", path.name)
        if match:
            test_numbers.append(int(match.group(1)))
    return (max(test_numbers) + 1) if test_numbers else 1


def _report_paths(
    output_dir: Path,
    scene_name: str,
    model_name: str,
    date: str,
    test_n: int,
) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    for kind in ("raw", "evaluation", "manual_review"):
        stem = f"benchmark_{kind}_{scene_name}_{model_name}_{date}_test_{test_n}"
        paths[kind] = {
            "json": output_dir / f"{stem}.json",
            "csv": output_dir / f"{stem}.csv",
        }
    return paths


def main() -> None:
    args = parse_args()
    point_cloud_path = select_point_cloud(args.point_cloud_path)
    annotation_csv_path = (
        str(resolve_local_path(args.annotation_csv))
        if args.annotation_csv
        else None
    )

    params = PipelineParams(
        point_cloud_path=point_cloud_path,
        sample_n=args.sample_n if args.sample_n > 0 else None,
        eps=args.eps,
        min_samples=args.min_samples,
        distance_threshold=args.distance_threshold,
        use_normals=args.use_normals,
        annotation_csv_path=annotation_csv_path,
        annotation_match_threshold=args.annotation_match_threshold,
    )
    ctx = run_pipeline(params)

    if args.plot_segmentation:
        from arch_agent.visualization.pointcloud_viewer import visualize_clustered_objects
        visualize_clustered_objects(ctx.objects)

    questions = load_questions(args.questions_file)
    print(f"Loaded questions: {len(questions)}")
    if len(questions) != EXPECTED_QUESTION_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_QUESTION_COUNT} questions, loaded {len(questions)}"
        )
    if args.limit > 0:
        questions = questions[: args.limit]
    print(f"Running questions: {len(questions)} from {args.questions_file}")
    models = _models_from_args(args)
    think_override = parse_think_override(args.think)
    print(f"Models: {', '.join(models)}")
    print(f"Thinking mode: {args.think}")

    def on_result(result):
        status = "ERROR" if result.error else f"{result.num_tool_calls} tool call(s)"
        print(f"  [{result.model} | {status}] {result.question}")

    output_dir = _output_dir_from_args(args)
    scene_name = _sanitize_for_path(Path(point_cloud_path).stem)
    date = _date_for_path()
    for model in models:
        print(f"\nBenchmarking model: {model}")
        model_name = _model_name_for_report(model, args.think)
        test_n = _next_test_number(output_dir, scene_name, model_name, date)
        metadata = {
            "scene": scene_name,
            "model": model,
            "think": args.think,
            "date": date,
            "test_n": test_n,
            "questions_loaded": EXPECTED_QUESTION_COUNT,
        }

        raw_records = run_benchmark(
            ctx,
            model,
            questions,
            capture_reasoning=args.capture_reasoning,
            think_override=think_override,
            on_result=on_result,
        )
        evaluation_records, summary = evaluate_benchmark(raw_records, ctx)
        manual_records = manual_review_records(evaluation_records)

        paths = _report_paths(output_dir, scene_name, model_name, date, test_n)
        write_raw_report(
            metadata,
            raw_records,
            paths["raw"]["json"],
            paths["raw"]["csv"],
        )
        write_evaluation_report(
            metadata,
            evaluation_records,
            summary,
            paths["evaluation"]["json"],
            paths["evaluation"]["csv"],
        )
        write_manual_review_report(
            metadata,
            manual_records,
            paths["manual_review"]["json"],
            paths["manual_review"]["csv"],
        )

        print("\nSummary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print("\nReports written:")
        for kind, kind_paths in paths.items():
            print(f"  {kind}: {kind_paths['json']} | {kind_paths['csv']}")


if __name__ == "__main__":
    main()
