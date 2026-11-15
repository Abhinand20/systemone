"""FastAPI backend and static SPA server for inspecting dm-bench traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from dm_bench.config import find_repo_root
from dm_bench.games.tic_tac_toe.engine import CELL_KEYS, is_terminal
from dm_bench.games.tic_tac_toe.oracle import analyze_position
from dm_bench.logging_io import read_jsonl

STATIC_DIR = Path(__file__).resolve().parent / "static"

CORE_ROW_ORDER: tuple[str, ...] = (
    "A_jev_V",
    "B_jev_T",
    "D_jev_VT",
    "A_gdec_V",
    "B_gdec_T",
    "D_gdec_VT",
    "C_ghyd_V",
    "E_ghyd_VT",
    "F_ghyd_T",
)
_ROW_ORDER_INDEX: dict[str, int] = {row_id: idx for idx, row_id in enumerate(CORE_ROW_ORDER)}

FailureFilter = Literal["all", "failures", "illegal", "suboptimal", "invalid"]


def _resolve_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    return find_repo_root()


def _validate_single_segment(value: str, label: str) -> str:
    if (
        not value
        or "/" in value
        or "\\" in value
        or ".." in value
        or "\x00" in value
        or value in (".", "..")
    ):
        raise HTTPException(status_code=400, detail=f"Invalid {label}: {value!r}")
    return value


def _resolve_run_dir(root: Path, experiment_id: str, run_id: str) -> Path:
    exp = _validate_single_segment(experiment_id, "experiment_id")
    run = _validate_single_segment(run_id, "run_id")
    results_root = (root / "experiments" / "results").resolve()
    run_dir = (results_root / exp / run).resolve()
    if not run_dir.is_relative_to(results_root) or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run {exp}/{run} not found")
    return run_dir


def _row_sort_key(row_id: str) -> tuple[int, str]:
    return (_ROW_ORDER_INDEX.get(row_id, len(CORE_ROW_ORDER)), row_id)


def _enrich_decision(row: dict[str, Any], frames_dir: Path) -> dict[str, Any]:
    out = dict(row)
    if out.get("effective_regret") is None and out.get("minimax_regret") is not None:
        out["effective_regret"] = out["minimax_regret"]
    elif out.get("minimax_regret") is None and out.get("effective_regret") is not None:
        out["minimax_regret"] = out["effective_regret"]

    frame_urls: list[str] = []
    for raw_path in out.get("frame_paths") or []:
        name = Path(str(raw_path)).name
        if name:
            frame_urls.append(f"/media/frames/{name}")
    if not frame_urls:
        for digest in out.get("frame_hashes") or []:
            candidate = f"{digest}.png"
            if (frames_dir / candidate).is_file():
                frame_urls.append(f"/media/frames/{candidate}")
    out["frame_urls"] = frame_urls
    return out


def _matches_failure_filter(row: dict[str, Any], failure_filter: FailureFilter) -> bool:
    if failure_filter == "all":
        return True
    response_valid = bool(row.get("response_valid", True))
    action_is_legal = bool(row.get("action_is_legal", False))
    action_is_optimal = bool(row.get("action_is_optimal", False))
    if failure_filter == "invalid":
        return not response_valid
    if failure_filter == "illegal":
        return response_valid and not action_is_legal
    if failure_filter == "suboptimal":
        return response_valid and action_is_legal and not action_is_optimal
    if failure_filter == "failures":
        return (not response_valid) or (not action_is_legal) or (not action_is_optimal)
    return True


def _extract_oracle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    board = list(first.get("board") or [])
    player_to_move = first.get("player_to_move")
    state_value = first.get("state_value")
    optimal_actions = list(first.get("optimal_actions") or [])
    action_values = dict(first.get("action_values") or {})
    tactical_category = first.get("tactical_category")

    if (
        len(board) == 9
        and player_to_move in ("X", "O")
        and not is_terminal(board)
        and (not action_values or not optimal_actions or state_value is None)
    ):
        analysis = analyze_position(board, player_to_move)
        if state_value is None:
            state_value = analysis.state_value
        if not action_values:
            action_values = dict(analysis.action_values)
        if not optimal_actions:
            optimal_actions = list(analysis.optimal_actions)
        if tactical_category is None:
            tactical_category = analysis.tactical_category

    ordered_action_values = {
        key: action_values[key] for key in CELL_KEYS if key in action_values
    }
    return {
        "board": board,
        "player_to_move": player_to_move,
        "starting_player": first.get("starting_player"),
        "ply": first.get("ply", 0),
        "move_history": list(first.get("move_history") or []),
        "legal_actions": list(first.get("legal_actions") or []),
        "state_value": state_value,
        "optimal_actions": optimal_actions,
        "action_values": ordered_action_values,
        "tactical_category": tactical_category,
    }


def _discover_runs(root: Path) -> list[dict[str, Any]]:
    results_dir = root / "experiments" / "results"
    if not results_dir.is_dir():
        return []

    runs: list[dict[str, Any]] = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            manifest_path = run_dir / "manifest.yaml"
            if not manifest_path.is_file():
                continue
            try:
                raw_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw_manifest, dict):
                continue

            entry = dict(raw_manifest)
            entry.setdefault("experiment_id", exp_dir.name)
            entry.setdefault("run_id", run_dir.name)
            entry["has_decisions"] = (run_dir / "decisions.jsonl").is_file()
            entry["has_episodes"] = (run_dir / "episodes.jsonl").is_file()
            runs.append(entry)

    runs.sort(
        key=lambda item: (str(item.get("created_at_utc") or ""), str(item.get("run_id") or "")),
        reverse=True,
    )
    return runs


def create_app(repo_root: Path | None = None) -> FastAPI:
    """Build the FastAPI application serving the dm-bench trace viewer."""
    root = _resolve_root(repo_root)
    frames_dir = (root / "experiments" / "media" / "frames").resolve()

    app = FastAPI(title="dm-bench Trace Viewer", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=404, detail="Viewer UI not found")
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

    @app.get("/media/frames/{filename:path}")
    def get_frame(filename: str) -> FileResponse:
        if (
            not filename
            or "/" in filename
            or "\\" in filename
            or ".." in filename
            or "\x00" in filename
            or not filename.lower().endswith(".png")
        ):
            raise HTTPException(status_code=400, detail="Invalid frame filename")

        candidate = (frames_dir / filename).resolve()
        if candidate.parent != frames_dir or not candidate.is_relative_to(frames_dir):
            raise HTTPException(status_code=400, detail="Invalid frame path")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="Frame not found")
        return FileResponse(candidate, media_type="image/png")

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        return {"runs": _discover_runs(root)}

    @app.get("/api/runs/{experiment_id}/{run_id}/decisions")
    def list_decisions(
        experiment_id: str,
        run_id: str,
        row_id: str | None = None,
        condition: str | None = None,
        tactical_category: str | None = None,
        render_variant: str | None = None,
        text_format: str | None = None,
        action_mode: str | None = None,
        failure_filter: FailureFilter = "all",
        only_failures: bool = False,
        only_illegal: bool = False,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=1000),
    ) -> dict[str, Any]:
        run_dir = _resolve_run_dir(root, experiment_id, run_id)
        all_rows = [
            _enrich_decision(row, frames_dir)
            for row in read_jsonl(run_dir / "decisions.jsonl")
        ]

        effective_failure_filter: FailureFilter = failure_filter
        if effective_failure_filter == "all":
            if only_illegal:
                effective_failure_filter = "illegal"
            elif only_failures:
                effective_failure_filter = "failures"

        facets = {
            "row_ids": sorted(
                {str(r["row_id"]) for r in all_rows if r.get("row_id")},
                key=_row_sort_key,
            ),
            "conditions": sorted(
                {str(r["condition_code"]) for r in all_rows if r.get("condition_code")}
            ),
            "tactical_categories": sorted(
                {str(r["tactical_category"]) for r in all_rows if r.get("tactical_category")}
            ),
            "render_variants": sorted(
                {str(r["render_variant"]) for r in all_rows if r.get("render_variant")}
            ),
            "text_formats": sorted(
                {str(r["text_format"]) for r in all_rows if r.get("text_format")}
            ),
            "action_modes": sorted(
                {str(r["action_mode"]) for r in all_rows if r.get("action_mode")}
            ),
        }

        filtered = [
            r
            for r in all_rows
            if (row_id is None or r.get("row_id") == row_id)
            and (condition is None or r.get("condition_code") == condition)
            and (tactical_category is None or r.get("tactical_category") == tactical_category)
            and (render_variant is None or r.get("render_variant") == render_variant)
            and (text_format is None or r.get("text_format") == text_format)
            and (action_mode is None or r.get("action_mode") == action_mode)
            and _matches_failure_filter(r, effective_failure_filter)
        ]

        page = filtered[offset : offset + limit]
        return {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "total": len(filtered),
            "total_unfiltered": len(all_rows),
            "offset": offset,
            "limit": limit,
            "decisions": page,
            "facets": facets,
        }

    @app.get("/api/runs/{experiment_id}/{run_id}/matrix/{episode_id}")
    def get_matrix(
        experiment_id: str,
        run_id: str,
        episode_id: str,
        render_variant: str | None = None,
        text_format: str | None = None,
        action_mode: str | None = None,
    ) -> dict[str, Any]:
        run_dir = _resolve_run_dir(root, experiment_id, run_id)
        all_rows = [
            _enrich_decision(row, frames_dir)
            for row in read_jsonl(run_dir / "decisions.jsonl")
            if row.get("episode_id") == episode_id
        ]
        if not all_rows:
            raise HTTPException(status_code=404, detail=f"Position {episode_id!r} not found")

        available_variants = {
            "render_variants": sorted(
                {str(r["render_variant"]) for r in all_rows if r.get("render_variant")}
            ),
            "text_formats": sorted(
                {str(r["text_format"]) for r in all_rows if r.get("text_format")}
            ),
            "action_modes": sorted(
                {str(r["action_mode"]) for r in all_rows if r.get("action_mode")}
            ),
        }

        matched = [
            r
            for r in all_rows
            if (render_variant is None or r.get("render_variant") == render_variant)
            and (text_format is None or r.get("text_format") == text_format)
            and (action_mode is None or r.get("action_mode") == action_mode)
        ]
        if not matched:
            raise HTTPException(
                status_code=404,
                detail=f"No rows for {episode_id!r} match the selected variant filters",
            )

        matched.sort(
            key=lambda r: (
                _row_sort_key(str(r.get("row_id") or "")),
                int(r.get("step_idx") or 0),
                str(r.get("timestamp_utc") or ""),
            )
        )

        oracle_summary = _extract_oracle_summary(matched)
        frame_urls: list[str] = []
        for row in matched:
            if row.get("frame_urls"):
                frame_urls = list(row["frame_urls"])
                break
        if not frame_urls:
            for row in all_rows:
                if row.get("frame_urls"):
                    frame_urls = list(row["frame_urls"])
                    break

        state_text = next(
            (row.get("state_text") for row in matched if row.get("state_text")),
            None,
        )

        return {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "episode_id": episode_id,
            **oracle_summary,
            "state_text": state_text,
            "frame_urls": frame_urls,
            "available_variants": available_variants,
            "rows": matched,
        }

    @app.get("/api/runs/{experiment_id}/{run_id}/episodes")
    def list_episodes(
        experiment_id: str,
        run_id: str,
        row_id: str | None = None,
        opponent_type: str | None = None,
        outcome: str | None = None,
        action_mode: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ) -> dict[str, Any]:
        run_dir = _resolve_run_dir(root, experiment_id, run_id)
        all_episodes = read_jsonl(run_dir / "episodes.jsonl")

        facets = {
            "row_ids": sorted(
                {str(e["row_id"]) for e in all_episodes if e.get("row_id")},
                key=_row_sort_key,
            ),
            "opponent_types": sorted(
                {str(e["opponent_type"]) for e in all_episodes if e.get("opponent_type")}
            ),
            "outcomes": sorted(
                {str(e["outcome"]) for e in all_episodes if e.get("outcome")}
            ),
            "action_modes": sorted(
                {str(e["action_mode"]) for e in all_episodes if e.get("action_mode")}
            ),
        }

        filtered = [
            e
            for e in all_episodes
            if (row_id is None or e.get("row_id") == row_id)
            and (opponent_type is None or e.get("opponent_type") == opponent_type)
            and (outcome is None or e.get("outcome") == outcome)
            and (action_mode is None or e.get("action_mode") == action_mode)
        ]

        page = filtered[offset : offset + limit]
        return {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "total": len(filtered),
            "total_unfiltered": len(all_episodes),
            "offset": offset,
            "limit": limit,
            "episodes": page,
            "facets": facets,
        }

    @app.get("/api/runs/{experiment_id}/{run_id}/episodes/{episode_id}")
    def get_episode(
        experiment_id: str,
        run_id: str,
        episode_id: str,
        row_id: str | None = None,
        action_mode: str | None = None,
        opponent_type: str | None = None,
    ) -> dict[str, Any]:
        run_dir = _resolve_run_dir(root, experiment_id, run_id)
        all_episodes = [
            e for e in read_jsonl(run_dir / "episodes.jsonl") if e.get("episode_id") == episode_id
        ]
        all_decisions = [
            _enrich_decision(d, frames_dir)
            for d in read_jsonl(run_dir / "decisions.jsonl")
            if d.get("episode_id") == episode_id
        ]

        if not all_episodes and not all_decisions:
            raise HTTPException(status_code=404, detail=f"Episode {episode_id!r} not found")

        matched_episodes = [
            e
            for e in all_episodes
            if (row_id is None or e.get("row_id") == row_id)
            and (action_mode is None or e.get("action_mode") == action_mode)
            and (opponent_type is None or e.get("opponent_type") == opponent_type)
        ]

        if all_episodes and not matched_episodes:
            raise HTTPException(
                status_code=404,
                detail=f"Episode {episode_id!r} not found for specified filters",
            )

        selected_episode = matched_episodes[0] if matched_episodes else None
        effective_row_id = row_id or (selected_episode.get("row_id") if selected_episode else None)
        effective_action_mode = action_mode or (
            selected_episode.get("action_mode") if selected_episode else None
        )
        effective_opponent = opponent_type or (
            selected_episode.get("opponent_type") if selected_episode else None
        )

        trace = [
            d
            for d in all_decisions
            if (effective_row_id is None or d.get("row_id") == effective_row_id)
            and (
                effective_action_mode is None
                or d.get("action_mode") is None
                or d.get("action_mode") == effective_action_mode
            )
            and (
                effective_opponent is None
                or d.get("opponent_type") is None
                or d.get("opponent_type") == effective_opponent
            )
        ]
        trace.sort(
            key=lambda d: (
                int(d.get("step_idx") or 0),
                int(d.get("ply") or 0),
                str(d.get("timestamp_utc") or ""),
            )
        )

        available_rows = sorted(
            {str(e["row_id"]) for e in all_episodes if e.get("row_id")}
            | {str(d["row_id"]) for d in all_decisions if d.get("row_id")},
            key=_row_sort_key,
        )

        return {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "episode_id": episode_id,
            "episode": selected_episode,
            "available_rows": available_rows,
            "decisions": trace,
        }

    return app


def serve_viewer(
    repo_root: Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Launch the FastAPI trace viewer with uvicorn."""
    app = create_app(repo_root=repo_root)
    uvicorn.run(app, host=host, port=port)
