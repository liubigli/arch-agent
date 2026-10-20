"""
Usage:
    python main.py
    python main.py path/to/scene.laz
    python main.py --eps 0.3 --min-samples 10
    python main.py --distance-threshold 5.0 --use-normals
    python main.py --model llama3.1
"""

import argparse
from pathlib import Path, PureWindowsPath

from arch_agent.pipeline.pipeline import PipelineParams, run_pipeline


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
        "--export-json", metavar="PATH",
        help="Export loaded points and all scene data as JSON (never overwrites an existing file)",
    )
    group.add_argument(
        "--export-usd", metavar="PATH",
        help="Export the scene as OpenUSD (.usd/.usda/.usdc; requires usd-core)",
    )
    group.add_argument(
        "--export-ifc", metavar="PATH",
        help="Export approximate BIM objects as IFC4 (.ifc; requires ifcopenshell)",
    )
    group.add_argument(
        "--usd-color-max", type=int, choices=(255, 65535), default=65535,
        help="RGB encoding maximum for USD display colors (LAZ defaults to 16-bit)",
    )
    group.add_argument(
        "--export-only", action="store_true",
        help="Exit after export without starting the agent (requires an --export-* option)",
    )
    group.add_argument(
        "--eps", type=float, default=0.5,
        help="DBSCAN epsilon for object segmentation (smaller = tighter clusters)",
    )
    group.add_argument(
        "--min-samples", type=int, default=15,
        help="DBSCAN min_samples (reduce to 10 for sparse clouds)",
    )
    group.add_argument(
        "--distance-threshold", type=float, default=2.0,
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
    group.add_argument(
        "--annotation-csv",
        default=None,
        help=(
            "Optional CSV with user-provided descriptions. If omitted, the "
            "pipeline looks for a CSV next to the LAZ with the same basename "
            "or suffixes _annotations/_metadata/_descriptions."
        ),
    )
    group.add_argument(
        "--annotation-match-threshold",
        type=float,
        default=2.0,
        help="Max distance in meters for matching CSV global_box_center to object AABB box_center",
    )

    # Visualization params
    group3 = parser.add_argument_group("visualization parameters")
    group3.add_argument(
        "--plot-segmentation", action="store_true",
        help=(
            "Open an Open3D window showing the segmented (DBSCAN) point "
            "cloud objects once the scene graph has been built."
        ),
    )

    # Agent params
    group2 = parser.add_argument_group("agent parameters")
    group2.add_argument(
        "--model", default="llama3",
        help="Ollama model name (must be pulled via 'ollama pull <model>')",
    )
    group2.add_argument(
        "--capture-reasoning", action="store_true",
        help=(
            "Add a preliminary, tool-unbound reasoning step before each tool "
            "call and print it as '[thinking] ...'. Roughly doubles LLM "
            "calls per turn."
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
        "--deterministic-router",
        action="store_true",
        help=(
            "Answer supported simple queries directly with Python before "
            "calling the LLM. Leave disabled when testing whether the LLM "
            "chooses the right tools."
        ),
    )

    args = parser.parse_args()
    if args.export_only and not (args.export_json or args.export_usd or args.export_ifc):
        parser.error("--export-only requires --export-json, --export-usd or --export-ifc")
    return args


def resolve_local_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.exists():
        return path

    windows_path = PureWindowsPath(path_value)
    if windows_path.drive:
        drive = windows_path.drive.rstrip(":").lower()
        wsl_path = Path("/mnt") / drive / Path(*windows_path.parts[1:])
        if wsl_path.exists():
            return wsl_path

    return path


def parse_think_override(value: str) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def select_point_cloud(path_value: str) -> str:
    path = resolve_local_path(path_value)
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

    if args.export_json:
        from arch_agent.scene_export import export_scene_json
        output = export_scene_json(ctx, resolve_local_path(args.export_json))
        print(f"Scene JSON saved to: {output}")
    if args.export_usd:
        from arch_agent.usd_export import export_scene_usd
        output = export_scene_usd(ctx, resolve_local_path(args.export_usd), color_max=args.usd_color_max)
        print(f"OpenUSD scene saved to: {output}")
    if args.export_ifc:
        from arch_agent.ifc_export import export_scene_ifc
        output = export_scene_ifc(ctx, resolve_local_path(args.export_ifc))
        print(f"IFC scene saved to: {output}")
    if args.export_only:
        return

    if args.plot_segmentation:
        from arch_agent.visualization.pointcloud_viewer import visualize_clustered_objects
        visualize_clustered_objects(ctx.objects)

    from arch_agent.agent import run_agent

    run_agent(
        ctx,
        model=args.model,
        capture_reasoning=args.capture_reasoning,
        deterministic_router=args.deterministic_router,
        think_override=parse_think_override(args.think),
    )


if __name__ == "__main__":
    main()
