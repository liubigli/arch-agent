"""
Usage:
    python benchmark.py path/to/scene.laz --questions-file tmp/domande_per_scene.txt
    python benchmark.py path/to/scene.laz --model llama3.1 --limit 20
    python benchmark.py path/to/scene.laz --model llama3.1 --output-prefix results/llama3_1

By default, reports are written to benchmark_results/<scene_name>/<model>.csv/.json
"""

import argparse
from pathlib import Path

from arch_agent.pipeline.pipeline import PipelineParams, run_pipeline
from arch_agent.benchmark.harness import (
    load_questions,
    run_benchmark,
    summarize,
    write_csv_report,
    write_json_report,
)
from main import DEFAULT_POINT_CLOUD_PATH, resolve_local_path, select_point_cloud


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

    group2 = parser.add_argument_group("benchmark parameters")
    group2.add_argument(
        "--model", default="llama3",
        help="Ollama model name (must be pulled via 'ollama pull <model>')",
    )
    group2.add_argument(
        "--questions-file", default="tmp/domande_per_scene.txt",
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
        "--output-prefix", default=None,
        help=(
            "Prefix for the .csv/.json report files. Default: "
            "benchmark_results/<model>/<scene_name>."
        ),
    )

    return parser.parse_args()


def _sanitize_for_path(text: str) -> str:
    return "".join(char if char.isalnum() or char in ".-_" else "_" for char in text)


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

    questions = load_questions(args.questions_file)
    if args.limit > 0:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} unique questions from {args.questions_file}")

    def on_result(result):
        status = "ERROR" if result.error else f"{result.num_tool_calls} tool call(s)"
        print(f"  [{status}] {result.question}")

    results = run_benchmark(
        ctx,
        args.model,
        questions,
        capture_reasoning=args.capture_reasoning,
        on_result=on_result,
    )

    if args.output_prefix:
        prefix = args.output_prefix
    else:
        scene_name = _sanitize_for_path(Path(point_cloud_path).stem)
        model_name = _sanitize_for_path(args.model)
        prefix = f"benchmark_results/{model_name}/{scene_name}"
    csv_path = Path(f"{prefix}.csv")
    json_path = Path(f"{prefix}.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_report(results, csv_path)
    write_json_report(results, json_path)

    print("\n" + summarize(results))
    print(f"\nReports written to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()