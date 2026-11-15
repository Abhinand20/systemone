"""Unified `bench` CLI entrypoint for dm-bench."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
from pathlib import Path
import sys
from typing import Sequence

from rich.console import Console
from rich.table import Table
import yaml

from dm_bench.config import find_repo_root, load_core_matrix, load_env, load_game_config
from dm_bench.games.tic_tac_toe.corpus import DATASET_RELATIVE_PATHS, build_all_corpora
from dm_bench.logging_io import RunLogger

console = Console()


def _parse_list_arg(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    out: list[str] = []
    for item in values:
        for part in item.split(","):
            stripped = part.strip()
            if stripped:
                out.append(stripped)
    return out or None


def _cmd_check_env(repo_root: Path) -> int:
    env_path = load_env(repo_root)
    table = Table(title="dm-bench Environment Check")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    table.add_row(
        ".env file",
        "[green]FOUND[/green]" if env_path is not None else "[yellow]MISSING[/yellow]",
        str(env_path) if env_path is not None else str(repo_root / ".env"),
    )

    for var in ("TYPESAFE_API_KEY", "GEMINI_API_KEY"):
        present = bool(os.environ.get(var, "").strip())
        table.add_row(
            f"Env: {var}",
            "[green]SET[/green]" if present else "[yellow]UNSET[/yellow]",
            "Non-empty" if present else "Required for live API calls (--mock works without it)",
        )

    imports_ok = True
    for mod_name in (
        "typesafe_sdk",
        "google.genai",
        "pydantic",
        "yaml",
        "PIL",
        "rich",
        "numpy",
        "matplotlib",
        "fastapi",
        "uvicorn",
        "httpx",
        "dm_bench",
    ):
        try:
            importlib.import_module(mod_name)
            table.add_row(f"Import: {mod_name}", "[green]OK[/green]", "")
        except Exception as exc:
            imports_ok = False
            table.add_row(f"Import: {mod_name}", "[red]FAIL[/red]", str(exc))

    datasets_ok = True
    for split_name, rel_path in DATASET_RELATIVE_PATHS.items():
        full_path = repo_root / rel_path
        exists = full_path.is_file()
        if not exists:
            datasets_ok = False
        table.add_row(
            f"Dataset: {split_name}",
            "[green]OK[/green]" if exists else "[red]MISSING[/red]",
            str(rel_path),
        )

    console.print(table)
    return 0 if (imports_ok and datasets_ok) else 1


def _cmd_build_corpus(repo_root: Path) -> int:
    written = build_all_corpora(repo_root)
    table = Table(title="Built Tic-Tac-Toe Corpora")
    table.add_column("Split", style="cyan")
    table.add_column("Path")
    table.add_column("Records", justify="right")
    for split_name, path in written.items():
        with path.open("r", encoding="utf-8") as fh:
            count = sum(1 for line in fh if line.strip())
        table.add_row(split_name, str(path.relative_to(repo_root)), str(count))
    console.print(table)
    return 0


def _cmd_smoke(args: argparse.Namespace, repo_root: Path) -> int:
    from dm_bench.smoke import run_smoke

    rows = _parse_list_arg(args.rows)
    models = _parse_list_arg(args.models)
    summary = asyncio.run(
        run_smoke(
            repo_root=repo_root,
            mock=args.mock,
            rows=rows,
            models=models,
            latency_trials=args.latency_trials,
            max_cost_usd=args.max_cost_usd,
            run_name=args.name,
        )
    )
    console.print(f"[bold green]Smoke run finished:[/bold green] {summary.get('run_dir')}")
    return 0


def _print_run_summary(run_dir: Path) -> None:
    logger = RunLogger(run_dir)
    manifest = logger.load_manifest()
    if manifest is None:
        console.print(f"Run finished: {run_dir}")
        return
    name_part = f" name={manifest.run_name!r}" if manifest.run_name else ""
    console.print(
        f"[bold]{manifest.experiment_id}[/bold] run_id={manifest.run_id}{name_part} "
        f"status={manifest.status} decisions={manifest.total_decisions} "
        f"episodes={manifest.total_episodes} spent=${manifest.total_spent_usd:.6f} "
        f"dir={run_dir}"
    )


def _print_h2h_summary_table(run_dir: Path) -> None:
    from dm_bench.analysis.stats import compute_h2h_metrics

    logger = RunLogger(run_dir)
    matches = logger.load_h2h_matches()
    if not matches:
        console.print("[yellow]No H2H matches recorded in this run.[/yellow]")
        return

    metrics = compute_h2h_metrics(matches)
    table = Table(title="H2H Matchup Scoreboard")
    table.add_column("Matchup (A vs B)", style="cyan")
    table.add_column("Mode")
    table.add_column("Games", justify="right")
    table.add_column("W-D-L (A)", justify="right")
    table.add_column("Score A", justify="right")
    table.add_column("p (decisive)", justify="right")
    table.add_column("Forfeits A/B", justify="right")
    table.add_column("Opt% A / B", justify="right")
    table.add_column("Cost", justify="right")

    for row in metrics.values():
        p_str = f"{row.p_value_decisive:.4f}" if row.p_value_decisive is not None else "n/a"
        table.add_row(
            f"{row.row_a} vs {row.row_b}",
            row.action_mode,
            str(row.games),
            f"{row.wins_a}-{row.draws}-{row.wins_b}",
            f"{row.score_a:.3f} [{row.score_a_ci_low:.2f}, {row.score_a_ci_high:.2f}]",
            p_str,
            f"{row.forfeit_wins_b}/{row.forfeit_wins_a}",
            f"{row.optimal_rate_a:.1%} / {row.optimal_rate_b:.1%}",
            f"${row.total_cost_usd:.4f}",
        )
    console.print(table)


def _run_h2h_from_args(args: argparse.Namespace, repo_root: Path) -> int:
    from dm_bench.runners.h2h import run_h2h_experiment

    rows = _parse_list_arg(args.rows)
    models = _parse_list_arg(args.models)
    matchups = _parse_list_arg(args.matchups)
    modalities = _parse_list_arg(args.modalities)
    action_modes = _parse_list_arg(args.action_modes)
    opening_mode = args.opening_mode or "standard"

    run_dir = asyncio.run(
        run_h2h_experiment(
            repo_root=repo_root,
            pilot=args.pilot,
            mock=args.mock,
            concurrency=args.concurrency,
            max_cost_usd=args.max_cost_usd,
            resume_run_id=args.resume_run_id,
            models=models,
            row_ids=rows,
            matchups=matchups,
            modalities=modalities,
            action_modes=action_modes,
            games=args.games,
            opening_mode=opening_mode,
            run_name=args.name,
            console=console,
        )
    )
    _print_run_summary(run_dir)
    _print_h2h_summary_table(run_dir)
    return 0


def _cmd_h2h(args: argparse.Namespace, repo_root: Path) -> int:
    return _run_h2h_from_args(args, repo_root)


def _cmd_eval(args: argparse.Namespace, repo_root: Path) -> int:
    from dm_bench.runners.matches import run_matches_experiment
    from dm_bench.runners.offline import run_offline_experiment

    rows = _parse_list_arg(args.rows)
    action_modes = _parse_list_arg(args.action_modes)
    opponents = _parse_list_arg(args.opponents)
    models = _parse_list_arg(args.models)
    matchups = _parse_list_arg(args.matchups)
    modalities = _parse_list_arg(args.modalities)
    exp = args.exp

    if opponents is not None and exp not in ("T6", "all"):
        raise ValueError("--opponents is only supported with --exp T6 or --exp all")
    if action_modes is not None and exp not in ("T6", "H2H", "all"):
        raise ValueError("--action-modes is only supported with --exp T6, --exp H2H, or --exp all")

    h2h_only_used = any(
        val is not None
        for val in (models, matchups, modalities, args.games, args.opening_mode)
    )
    if h2h_only_used and exp != "H2H":
        raise ValueError(
            "--models, --matchups, --modalities, --games, and --opening-mode are only supported with --exp H2H (or `bench h2h`)"
        )

    if exp == "H2H":
        if args.full_240:
            raise ValueError("--full-240 is not used with --exp H2H")
        return _run_h2h_from_args(args, repo_root)

    if exp == "all":
        if args.resume_run_id is not None:
            raise ValueError("--resume-run-id cannot be used with --exp all")
        from dm_bench.runners.offline import T3_DEFAULT_ROWS, T5_DEFAULT_ROWS

        for offline_exp in ("T1_T2", "T3", "T4", "T5"):
            exp_rows = rows
            if rows is not None and offline_exp == "T3":
                exp_rows = [r for r in rows if r in T3_DEFAULT_ROWS]
                if not exp_rows:
                    continue
            elif rows is not None and offline_exp == "T5":
                exp_rows = [r for r in rows if r in T5_DEFAULT_ROWS]
                if not exp_rows:
                    continue
            run_dir = asyncio.run(
                run_offline_experiment(
                    offline_exp,
                    repo_root=repo_root,
                    pilot=args.pilot,
                    full_240=args.full_240,
                    mock=args.mock,
                    concurrency=args.concurrency,
                    max_cost_usd=args.max_cost_usd,
                    row_ids=exp_rows,
                    run_name=args.name,
                    console=console,
                )
            )
            _print_run_summary(run_dir)
        run_dir = asyncio.run(
            run_matches_experiment(
                repo_root=repo_root,
                pilot=args.pilot,
                mock=args.mock,
                concurrency=args.concurrency,
                max_cost_usd=args.max_cost_usd,
                row_ids=rows,
                action_modes=action_modes,
                opponents=opponents,
                run_name=args.name,
                console=console,
            )
        )
        _print_run_summary(run_dir)
        return 0

    if exp == "T6":
        run_dir = asyncio.run(
            run_matches_experiment(
                repo_root=repo_root,
                pilot=args.pilot,
                mock=args.mock,
                concurrency=args.concurrency,
                max_cost_usd=args.max_cost_usd,
                resume_run_id=args.resume_run_id,
                row_ids=rows,
                action_modes=action_modes,
                opponents=opponents,
                run_name=args.name,
                console=console,
            )
        )
        _print_run_summary(run_dir)
        return 0

    run_dir = asyncio.run(
        run_offline_experiment(
            exp,
            repo_root=repo_root,
            pilot=args.pilot,
            full_240=args.full_240,
            mock=args.mock,
            concurrency=args.concurrency,
            max_cost_usd=args.max_cost_usd,
            resume_run_id=args.resume_run_id,
            row_ids=rows,
            run_name=args.name,
            console=console,
        )
    )
    _print_run_summary(run_dir)
    return 0


def _discover_latest_runs(repo_root: Path) -> list[Path]:
    results_root = repo_root / "experiments" / "results"
    if not results_root.is_dir():
        return []
    discovered: list[Path] = []
    for exp_dir in sorted(results_root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == "M0_smoke":
            continue
        candidates: list[tuple[str, str, Path]] = []
        for run_dir in exp_dir.iterdir():
            manifest_path = run_dir / "manifest.yaml"
            if not (run_dir.is_dir() and manifest_path.is_file()):
                continue
            try:
                data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            except Exception:
                data = {}
            created = str(data.get("created_at_utc", ""))
            candidates.append((created, run_dir.name, run_dir))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]))
            discovered.append(candidates[-1][2])
    return discovered


def _cmd_analyze(args: argparse.Namespace, repo_root: Path) -> int:
    from dm_bench.analysis.report import analyze_runs

    if args.runs:
        run_dirs = [
            Path(p) if Path(p).is_absolute() else (repo_root / p)
            for p in args.runs
        ]
    else:
        run_dirs = _discover_latest_runs(repo_root)

    if not run_dirs:
        console.print("[red]No experiment runs found to analyze.[/red]")
        return 1

    result = analyze_runs(run_dirs, repo_root=repo_root, tag=args.tag)
    console.print(f"[bold green]Analysis report:[/bold green] {result.get('report_path')}")
    for plot_path in result.get("plots", []):
        console.print(f"  plot: {plot_path}")
    return 0


def _cmd_view(args: argparse.Namespace, repo_root: Path) -> int:
    from dm_bench.viewer.app import serve_viewer

    serve_viewer(repo_root=repo_root, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the `bench` argparse CLI parser."""
    parser = argparse.ArgumentParser(
        prog="bench",
        description="dm-bench: Multimodal Decision Model vs Generative VLM/LLM benchmark harness.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repository root override.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check-env", help="Verify .env, API keys, imports, and datasets.")
    subparsers.add_parser("build-corpus", help="Build deterministic Tic-Tac-Toe JSONL splits.")

    smoke_p = subparsers.add_parser("smoke", help="Run Milestone 0 contract & latency smoke test.")
    smoke_p.add_argument("--mock", action="store_true", help="Use deterministic mock adapters.")
    smoke_p.add_argument("--rows", nargs="+", default=None, help="Filter specific row_ids.")
    smoke_p.add_argument("--models", nargs="+", default=None, help="Filter specific model_keys.")
    smoke_p.add_argument("--latency-trials", type=int, default=10, help="Sequential latency trials.")
    smoke_p.add_argument("--max-cost-usd", type=float, default=1.0, help="Hard USD spend cap.")
    smoke_p.add_argument(
        "--name",
        default=None,
        help="Optional label for this run (added to run_id, manifest, and trace viewer).",
    )

    eval_p = subparsers.add_parser("eval", help="Run Track 1 (T1-T5), Track 2 (T6), or H2H evaluation.")
    eval_p.add_argument(
        "--exp",
        required=True,
        choices=["T1", "T2", "T1_T2", "T3", "T4", "T5", "T6", "H2H", "all"],
        help="Experiment ID to execute.",
    )
    eval_p.add_argument("--pilot", action="store_true", help="Run pilot subset (30 states / 20 games).")
    eval_p.add_argument("--full-240", action="store_true", help="Use 240-state test split for T3/T5.")
    eval_p.add_argument("--mock", action="store_true", help="Use deterministic mock adapters.")
    eval_p.add_argument("--concurrency", type=int, default=4, help="Max concurrent async tasks.")
    eval_p.add_argument("--max-cost-usd", type=float, default=10.0, help="Hard USD spend cap.")
    eval_p.add_argument("--resume-run-id", default=None, help="Resume an existing run_id.")
    eval_p.add_argument("--rows", nargs="+", default=None, help="Optional row_id filter.")
    eval_p.add_argument(
        "--action-modes",
        nargs="+",
        default=None,
        help="Optional T6/H2H action_modes filter (all_9, legal_only).",
    )
    eval_p.add_argument(
        "--opponents",
        nargs="+",
        default=None,
        help="Optional T6 opponent filter (random, minimax).",
    )
    eval_p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional H2H model filter/pair (e.g. jev gdec).",
    )
    eval_p.add_argument(
        "--matchups",
        nargs="+",
        default=None,
        help="Optional H2H matchup specs (e.g. jev:gdec or A_jev_V:A_gdec_V).",
    )
    eval_p.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help="Optional H2H modality filter (V, T, VT).",
    )
    eval_p.add_argument(
        "--games",
        type=int,
        default=None,
        help="Optional H2H number of games per matchup/action_mode (default 40, or 20 with --pilot).",
    )
    eval_p.add_argument(
        "--opening-mode",
        choices=["standard", "empty"],
        default=None,
        help="Optional H2H opening schedule ('standard' or 'empty').",
    )
    eval_p.add_argument(
        "--name",
        default=None,
        help="Optional label for this run (added to run_id, manifest, and trace viewer).",
    )

    h2h_p = subparsers.add_parser("h2h", help="Run Head-to-Head (H2H) closed-loop model vs model evaluation.")
    h2h_p.add_argument("--pilot", action="store_true", help="Run pilot subset (20 games per matchup/mode).")
    h2h_p.add_argument("--mock", action="store_true", help="Use deterministic mock adapters.")
    h2h_p.add_argument("--concurrency", type=int, default=4, help="Max concurrent async tasks.")
    h2h_p.add_argument("--max-cost-usd", type=float, default=10.0, help="Hard USD spend cap.")
    h2h_p.add_argument("--resume-run-id", default=None, help="Resume an existing H2H run_id.")
    h2h_p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model keys or aliases to pair (e.g. jev gdec). Matched across selected modalities.",
    )
    h2h_p.add_argument("--rows", nargs="+", default=None, help="Optional condition row_id pool to pair.")
    h2h_p.add_argument(
        "--matchups",
        nargs="+",
        default=None,
        help="Explicit matchup specs (e.g. jev:gdec or A_jev_V:A_gdec_V).",
    )
    h2h_p.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help="Modalities to include when matching models/rows (V, T, VT).",
    )
    h2h_p.add_argument(
        "--action-modes",
        nargs="+",
        default=None,
        help="Action modes to run (all_9, legal_only). Default: all_9.",
    )
    h2h_p.add_argument(
        "--games",
        type=int,
        default=None,
        help="Number of games per matchup/action_mode (default: 40, or 20 with --pilot).",
    )
    h2h_p.add_argument(
        "--opening-mode",
        choices=["standard", "empty"],
        default="standard",
        help="Opening schedule: 'standard' (balanced 0/1/2-ply fair openings) or 'empty'.",
    )
    h2h_p.add_argument(
        "--name",
        default=None,
        help="Optional label for this run (added to run_id, manifest, and trace viewer).",
    )

    analyze_p = subparsers.add_parser("analyze", help="Compute stats, plots, and Markdown report.")
    analyze_p.add_argument("--runs", nargs="+", default=None, help="Explicit run directories to analyze.")
    analyze_p.add_argument("--tag", default="latest", help="Output tag for plots and summary report.")

    view_p = subparsers.add_parser("view", help="Launch FastAPI trace viewer.")
    view_p.add_argument("--host", default="127.0.0.1", help="Bind host.")
    view_p.add_argument("--port", type=int, default=8000, help="Bind port.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `bench`."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    repo_root = find_repo_root(args.repo_root)

    if args.command == "check-env":
        return _cmd_check_env(repo_root)
    if args.command == "build-corpus":
        return _cmd_build_corpus(repo_root)
    if args.command == "smoke":
        return _cmd_smoke(args, repo_root)
    if args.command == "eval":
        return _cmd_eval(args, repo_root)
    if args.command == "h2h":
        return _cmd_h2h(args, repo_root)
    if args.command == "analyze":
        return _cmd_analyze(args, repo_root)
    if args.command == "view":
        return _cmd_view(args, repo_root)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
