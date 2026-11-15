"""FastAPI backend and static SPA server for inspecting dm-bench traces."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from dm_bench.analysis.stats import (
    DEFAULT_BOOTSTRAP_SEED,
    RENDER_VARIANT_ORDER,
    TACTICAL_CATEGORY_ORDER,
    TEXT_FORMAT_ORDER,
    bootstrap_ci,
    compare_paired_decisions,
    compute_dm_calibration,
    compute_h2h_metrics,
    compute_h2h_standings,
    compute_tactical_metrics,
    compute_track1_metrics,
    compute_track2_metrics,
    filter_evaluated_decisions,
    load_runs_data,
    select_core_t2_decisions,
    select_dm_decisions,
    select_t3_decisions,
    select_t4_decisions,
    select_t5_decisions,
    sort_row_ids,
)
from dm_bench.config import find_repo_root
from dm_bench.contracts import DecisionLogRow, EpisodeLogRow
from dm_bench.games.tic_tac_toe.engine import CELL_KEYS, is_terminal
from dm_bench.games.tic_tac_toe.oracle import analyze_position
from dm_bench.logging_io import (
    RunLogger,
    read_jsonl,
    reconcile_decisions,
    reconcile_episodes_with_decisions,
)

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
            has_h2h = (run_dir / "h2h_matches.jsonl").is_file()
            entry["has_decisions"] = (run_dir / "decisions.jsonl").is_file()
            entry["has_h2h_matches"] = has_h2h
            entry["has_episodes"] = (run_dir / "episodes.jsonl").is_file() or has_h2h
            runs.append(entry)

    runs.sort(
        key=lambda item: (str(item.get("created_at_utc") or ""), str(item.get("run_id") or "")),
        reverse=True,
    )
    return runs


def _summarize_rows_with_ci(
    decisions: Sequence[DecisionLogRow],
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Summarize decisions per `row_id` with bootstrap 95% confidence intervals."""
    eval_decisions = filter_evaluated_decisions(decisions)
    if not eval_decisions:
        return []
    by_row: dict[str, list[DecisionLogRow]] = defaultdict(list)
    for row in eval_decisions:
        by_row[row.row_id].append(row)

    results: list[dict[str, Any]] = []
    for rid in sort_row_ids(list(by_row.keys())):
        grp = by_row[rid]
        if not grp:
            continue
        base = grp[0]
        normalized = [
            r.model_copy(
                update={
                    "experiment_id": base.experiment_id,
                    "render_variant": None,
                    "text_format": None,
                    "action_mode": base.action_mode,
                }
            )
            for r in grp
        ]
        metrics = next(iter(compute_track1_metrics(normalized).values())).to_dict()
        opt_ci = bootstrap_ci(
            [1.0 if r.action_is_optimal else 0.0 for r in grp],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        ill_ci = bootstrap_ci(
            [1.0 if (r.response_valid and not r.action_is_legal) else 0.0 for r in grp],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        metrics["optimal_ci_low"] = opt_ci.ci_low
        metrics["optimal_ci_high"] = opt_ci.ci_high
        metrics["illegal_ci_low"] = ill_ci.ci_low
        metrics["illegal_ci_high"] = ill_ci.ci_high
        metrics["source_experiment_id"] = base.experiment_id
        metrics["source_run_id"] = base.run_id
        results.append(metrics)
    return results


def _compute_paired_highlights(
    pool: Sequence[DecisionLogRow],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Compute paired statistical comparisons across modalities and model families."""
    single_step = [
        r
        for r in pool
        if r.step_idx == 0 and r.experiment_id not in ("T6", "H2H")
    ]
    if not single_step:
        return {}

    by_row: dict[str, dict[str, DecisionLogRow]] = defaultdict(dict)
    for r in single_step:
        by_row[r.row_id].setdefault(r.episode_id, r)

    specs: dict[str, list[tuple[str, str, str]]] = {
        "modality_text_vs_image": [
            ("TypeSafe Jev: Text Only vs Image Only", "B_jev_T", "A_jev_V"),
            ("Gemini Decision 1: Text Only vs Image Only", "B_gdec_T", "A_gdec_V"),
            ("Gemini 4 Hydrogen: Text Only vs Image Only", "F_ghyd_T", "C_ghyd_V"),
        ],
        "modality_hybrid_vs_single": [
            ("TypeSafe Jev: Image+Text vs Image Only", "D_jev_VT", "A_jev_V"),
            ("TypeSafe Jev: Image+Text vs Text Only", "D_jev_VT", "B_jev_T"),
            ("Gemini Decision 1: Image+Text vs Image Only", "D_gdec_VT", "A_gdec_V"),
            ("Gemini Decision 1: Image+Text vs Text Only", "D_gdec_VT", "B_gdec_T"),
            ("Gemini 4 Hydrogen: Image+Text vs Image Only", "E_ghyd_VT", "C_ghyd_V"),
            ("Gemini 4 Hydrogen: Image+Text vs Text Only", "E_ghyd_VT", "F_ghyd_T"),
        ],
        "decision_model_vs_generative": [
            ("Image Only: TypeSafe Jev vs Gemini 4 Hydrogen", "A_jev_V", "C_ghyd_V"),
            ("Text Only: TypeSafe Jev vs Gemini 4 Hydrogen", "B_jev_T", "F_ghyd_T"),
            ("Image + Text: TypeSafe Jev vs Gemini 4 Hydrogen", "D_jev_VT", "E_ghyd_VT"),
            ("Image Only: Gemini Decision 1 vs Gemini 4 Hydrogen", "A_gdec_V", "C_ghyd_V"),
            ("Text Only: Gemini Decision 1 vs Gemini 4 Hydrogen", "B_gdec_T", "F_ghyd_T"),
            ("Image + Text: Gemini Decision 1 vs Gemini 4 Hydrogen", "D_gdec_VT", "E_ghyd_VT"),
        ],
        "decision_model_vs_decision_model": [
            ("Image Only: TypeSafe Jev vs Gemini Decision 1", "A_jev_V", "A_gdec_V"),
            ("Text Only: TypeSafe Jev vs Gemini Decision 1", "B_jev_T", "B_gdec_T"),
            ("Image + Text: TypeSafe Jev vs Gemini Decision 1", "D_jev_VT", "D_gdec_VT"),
        ],
    }

    output: dict[str, list[dict[str, Any]]] = {}
    for group_name, pairs in specs.items():
        items: list[dict[str, Any]] = []
        for label, rx, ry in pairs:
            if rx not in by_row or ry not in by_row:
                continue
            comp = compare_paired_decisions(
                list(by_row[rx].values()),
                list(by_row[ry].values()),
                label=label,
                row_x_label=rx,
                row_y_label=ry,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            if comp is not None:
                items.append(comp.to_dict())
        output[group_name] = items
    return output


def _compute_tactical_matrix(pool: Sequence[DecisionLogRow]) -> dict[str, Any]:
    tactical_pool = [d for d in pool if d.tactical_category]
    if not tactical_pool:
        return {"source_experiment_id": None, "categories": [], "rows": []}
    tactical_metrics = compute_tactical_metrics(tactical_pool)
    rows = sort_row_ids(list({k[0] for k in tactical_metrics}))
    categories = [c for c in TACTICAL_CATEGORY_ORDER if any((r, c) in tactical_metrics for r in rows)]
    categories.extend(sorted({k[1] for k in tactical_metrics} - set(categories)))
    return {
        "source_experiment_id": tactical_pool[0].experiment_id,
        "categories": categories,
        "rows": [
            {
                "row_id": rid,
                "cells": {
                    cat: tactical_metrics[(rid, cat)].to_dict()
                    for cat in categories
                    if (rid, cat) in tactical_metrics
                },
            }
            for rid in rows
        ],
    }


def _compute_variant_matrix(
    pool: Sequence[DecisionLogRow],
    *,
    field: Literal["render_variant", "text_format"],
    order: tuple[str, ...],
    default_val: str,
    preferred_exp: str,
    fallback_exp: str,
) -> dict[str, Any]:
    if not pool:
        return {"source_experiment_id": None, "variants": [], "rows": []}
    grouped: dict[tuple[str, str], list[DecisionLogRow]] = defaultdict(list)
    for row in pool:
        val = getattr(row, field) or default_val
        grouped[(row.row_id, str(val))].append(row)

    rows = sort_row_ids(list({k[0] for k in grouped}))
    present_vars = {k[1] for k in grouped}
    variants = [v for v in order if v in present_vars]
    variants.extend(sorted(present_vars - set(variants)))

    out_rows: list[dict[str, Any]] = []
    for rid in rows:
        cells: dict[str, dict[str, Any]] = {}
        for var in variants:
            grp = grouped.get((rid, var), [])
            if not grp:
                continue
            base = grp[0]
            normalized = [
                r.model_copy(
                    update={
                        "experiment_id": base.experiment_id,
                        "action_mode": base.action_mode,
                        "render_variant": var if field == "render_variant" else None,
                        "text_format": var if field == "text_format" else None,
                    }
                )
                for r in grp
            ]
            cells[var] = next(iter(compute_track1_metrics(normalized).values())).to_dict()
        out_rows.append({"row_id": rid, "cells": cells})

    src_exp = preferred_exp if any(d.experiment_id == preferred_exp for d in pool) else fallback_exp
    return {"source_experiment_id": src_exp, "variants": variants, "rows": out_rows}


def _compute_legality_ablation(
    decisions: Sequence[DecisionLogRow],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    all9_t4, legal_only_t4 = select_t4_decisions(decisions)
    if not legal_only_t4:
        return {"paired": [], "standalone_legal_only": [], "needs_suite_baseline": False}

    lo_by_row: dict[str, list[DecisionLogRow]] = defaultdict(list)
    for d in legal_only_t4:
        lo_by_row[d.row_id].append(d)

    if not all9_t4:
        return {
            "paired": [],
            "standalone_legal_only": _summarize_rows_with_ci(
                legal_only_t4, n_bootstrap=n_bootstrap, seed=seed
            ),
            "needs_suite_baseline": True,
        }

    a9_by_row: dict[str, list[DecisionLogRow]] = defaultdict(list)
    for d in all9_t4:
        a9_by_row[d.row_id].append(d)

    paired_items: list[dict[str, Any]] = []
    for rid in sort_row_ids(list(set(a9_by_row) & set(lo_by_row))):
        comp = compare_paired_decisions(
            lo_by_row[rid],
            a9_by_row[rid],
            label=f"{rid}: legal_only vs all_9",
            row_x_label=f"{rid} (legal_only)",
            row_y_label=f"{rid} (all_9)",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        if comp is None:
            continue
        paired_items.append(
            {
                "row_id": rid,
                "shared_n": comp.shared_n,
                "all_9_acc": comp.optimal_ci.mean_y,
                "legal_only_acc": comp.optimal_ci.mean_x,
                "delta_acc": comp.optimal_ci.delta,
                "ci_low": comp.optimal_ci.ci_low,
                "ci_high": comp.optimal_ci.ci_high,
                "mcnemar_pvalue": comp.mcnemar_pvalue,
                "all_9_illegal_rate": comp.illegal_ci.mean_y,
                "legal_only_illegal_rate": comp.illegal_ci.mean_x,
            }
        )
    return {
        "paired": paired_items,
        "standalone_legal_only": [],
        "needs_suite_baseline": False,
    }


def _compute_calibration_payload(
    decisions: Sequence[DecisionLogRow],
    *,
    scope: str,
) -> dict[str, Any]:
    dm_pool = (
        select_dm_decisions(decisions)
        if scope == "suite"
        else [
            d
            for d in decisions
            if d.family == "decision_model" or d.action_probabilities is not None
        ]
    )
    if not dm_pool:
        return {"by_model": {}, "by_row": {}}

    by_model: dict[str, list[DecisionLogRow]] = defaultdict(list)
    by_row: dict[str, list[DecisionLogRow]] = defaultdict(list)
    for d in dm_pool:
        by_model[d.model_key].append(d)
        by_row[d.row_id].append(d)

    return {
        "by_model": {
            mkey: compute_dm_calibration(by_model[mkey]).to_dict()
            for mkey in sorted(by_model.keys())
        },
        "by_row": {
            rid: compute_dm_calibration(by_row[rid]).to_dict()
            for rid in sort_row_ids(list(by_row.keys()))
        },
    }


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
        matchup_id: str | None = None,
        outcome: str | None = None,
        action_mode: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ) -> dict[str, Any]:
        run_dir = _resolve_run_dir(root, experiment_id, run_id)
        run_logger = RunLogger(run_dir)
        dedup_h2h_map: dict[tuple[str, str, str, str], Any] = {}
        for m in run_logger.load_h2h_matches():
            dedup_h2h_map[m.resume_key()] = m
        all_h2h = [m.model_dump(mode="json") for m in dedup_h2h_map.values()]

        dedup_episodes_map: dict[tuple[str, str, str, str, str], EpisodeLogRow] = {}
        for ep in run_logger.load_episodes():
            dedup_episodes_map[ep.resume_key()] = ep
        dedup_decisions_map: dict[
            tuple[str, str, str, int, str | None, str | None, str, str | None, str | None],
            DecisionLogRow,
        ] = {}
        for row in run_logger.load_decisions():
            dkey = (
                row.experiment_id,
                row.row_id,
                row.episode_id,
                row.step_idx,
                row.render_variant,
                row.text_format,
                row.action_mode,
                row.matchup_id,
                row.opponent_type,
            )
            dedup_decisions_map[dkey] = row
        reconciled_decs = reconcile_decisions(dedup_decisions_map.values())
        reconciled_eps = reconcile_episodes_with_decisions(
            dedup_episodes_map.values(), reconciled_decs
        )
        all_episodes = [ep.model_dump(mode="json") for ep in reconciled_eps]

        is_h2h = bool(all_h2h) and (not all_episodes or experiment_id == "H2H")
        if is_h2h:
            matchup_ids = sorted(
                {str(m["matchup_id"]) for m in all_h2h if m.get("matchup_id")}
            )
            facets = {
                "row_ids": sorted(
                    {
                        str(rid)
                        for m in all_h2h
                        for rid in (m.get("row_a"), m.get("row_b"))
                        if rid
                    },
                    key=_row_sort_key,
                ),
                "matchup_ids": matchup_ids,
                "opponent_types": matchup_ids,
                "outcomes": sorted(
                    {str(m["outcome"]) for m in all_h2h if m.get("outcome")}
                ),
                "action_modes": sorted(
                    {str(m["action_mode"]) for m in all_h2h if m.get("action_mode")}
                ),
            }
            eff_matchup = matchup_id or opponent_type
            filtered = [
                m
                for m in all_h2h
                if (
                    row_id is None
                    or m.get("row_a") == row_id
                    or m.get("row_b") == row_id
                )
                and (eff_matchup is None or m.get("matchup_id") == eff_matchup)
                and (outcome is None or m.get("outcome") == outcome)
                and (action_mode is None or m.get("action_mode") == action_mode)
            ]
            page = filtered[offset : offset + limit]
            return {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "is_h2h": True,
                "total": len(filtered),
                "total_unfiltered": len(all_h2h),
                "offset": offset,
                "limit": limit,
                "episodes": page,
                "facets": facets,
            }

        facets = {
            "row_ids": sorted(
                {str(e["row_id"]) for e in all_episodes if e.get("row_id")},
                key=_row_sort_key,
            ),
            "matchup_ids": [],
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
            "is_h2h": False,
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
        matchup_id: str | None = None,
    ) -> dict[str, Any]:
        run_dir = _resolve_run_dir(root, experiment_id, run_id)
        run_logger = RunLogger(run_dir)
        h2h_models_map: dict[tuple[str, str, str, str], Any] = {}
        for m in run_logger.load_h2h_matches():
            if m.episode_id == episode_id:
                h2h_models_map[m.resume_key()] = m
        all_h2h = [m.model_dump(mode="json") for m in h2h_models_map.values()]

        ep_models_map: dict[tuple[str, str, str, str, str], EpisodeLogRow] = {}
        for ep in run_logger.load_episodes():
            if ep.episode_id == episode_id:
                ep_models_map[ep.resume_key()] = ep
        dec_models_map: dict[
            tuple[str, str, str, int, str | None, str | None, str, str | None, str | None],
            DecisionLogRow,
        ] = {}
        for row in run_logger.load_decisions():
            if row.episode_id == episode_id:
                dkey = (
                    row.experiment_id,
                    row.row_id,
                    row.episode_id,
                    row.step_idx,
                    row.render_variant,
                    row.text_format,
                    row.action_mode,
                    row.matchup_id,
                    row.opponent_type,
                )
                dec_models_map[dkey] = row
        reconciled_dec_models = reconcile_decisions(dec_models_map.values())
        reconciled_ep_models = reconcile_episodes_with_decisions(
            ep_models_map.values(), reconciled_dec_models
        )
        all_episodes = [ep.model_dump(mode="json") for ep in reconciled_ep_models]
        all_decisions = [
            _enrich_decision(row.model_dump(mode="json"), frames_dir)
            for row in reconciled_dec_models
        ]

        is_h2h = bool(all_h2h) and (not all_episodes or experiment_id == "H2H")

        if is_h2h:
            if not all_h2h and not all_decisions:
                raise HTTPException(status_code=404, detail=f"Episode {episode_id!r} not found")
            eff_matchup = matchup_id or opponent_type
            matched_h2h = [
                m
                for m in all_h2h
                if (
                    row_id is None
                    or m.get("row_a") == row_id
                    or m.get("row_b") == row_id
                )
                and (action_mode is None or m.get("action_mode") == action_mode)
                and (eff_matchup is None or m.get("matchup_id") == eff_matchup)
            ]
            if all_h2h and not matched_h2h:
                raise HTTPException(
                    status_code=404,
                    detail=f"Episode {episode_id!r} not found for specified filters",
                )

            selected_episode = matched_h2h[0] if matched_h2h else None
            effective_action_mode = action_mode or (
                selected_episode.get("action_mode") if selected_episode else None
            )
            sel_matchup_id = (
                selected_episode.get("matchup_id") if selected_episode else eff_matchup
            )
            row_a = selected_episode.get("row_a") if selected_episode else None
            row_b = selected_episode.get("row_b") if selected_episode else None

            trace_raw = []
            for d in all_decisions:
                if (
                    effective_action_mode is not None
                    and d.get("action_mode") is not None
                    and d.get("action_mode") != effective_action_mode
                ):
                    continue
                if sel_matchup_id and d.get("matchup_id"):
                    if d.get("matchup_id") != sel_matchup_id:
                        continue
                elif row_a and row_b:
                    if d.get("row_id") not in {row_a, row_b}:
                        continue
                trace_raw.append(d)

            trace_raw.sort(
                key=lambda d: (
                    int(d.get("ply") or 0),
                    int(d.get("step_idx") or 0),
                    str(d.get("timestamp_utc") or ""),
                )
            )
            trace = [
                {
                    **d,
                    "player_side": (
                        "A"
                        if (row_a and d.get("row_id") == row_a)
                        else ("B" if (row_b and d.get("row_id") == row_b) else None)
                    ),
                }
                for d in trace_raw
            ]
            available_rows = sorted(
                {
                    str(rid)
                    for m in all_h2h
                    for rid in (m.get("row_a"), m.get("row_b"))
                    if rid
                }
                | {str(d["row_id"]) for d in all_decisions if d.get("row_id")},
                key=_row_sort_key,
            )
            return {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "episode_id": episode_id,
                "is_h2h": True,
                "episode": selected_episode,
                "available_rows": available_rows,
                "decisions": trace,
            }

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
            "is_h2h": False,
            "episode": selected_episode,
            "available_rows": available_rows,
            "decisions": trace,
        }

    @app.get("/api/runs/{experiment_id}/{run_id}/summary")
    def get_run_summary(
        experiment_id: str,
        run_id: str,
        scope: Literal["run", "suite"] = "suite",
        n_bootstrap: int = Query(1000, ge=100, le=10000),
    ) -> dict[str, Any]:
        selected_dir = _resolve_run_dir(root, experiment_id, run_id)
        discovered = _discover_runs(root)
        selected_entry = next(
            (
                entry
                for entry in discovered
                if entry.get("experiment_id") == experiment_id and entry.get("run_id") == run_id
            ),
            None,
        )
        if selected_entry is None:
            selected_entry = {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "mock_mode": False,
                "pilot_mode": False,
                "total_decisions": 0,
                "total_episodes": 0,
            }

        target_mock = bool(selected_entry.get("mock_mode", False))
        if scope == "run":
            included_entries = [selected_entry]
            run_dirs = [selected_dir]
        else:
            included_entries = [selected_entry]
            run_dirs = [selected_dir]
            seen_experiments = {experiment_id}
            for entry in discovered:
                exp = str(entry.get("experiment_id") or "")
                rid = str(entry.get("run_id") or "")
                if not exp or not rid or exp in seen_experiments or exp == "M0_smoke":
                    continue
                if bool(entry.get("mock_mode", False)) != target_mock:
                    continue
                seen_experiments.add(exp)
                included_entries.append(entry)
                run_dirs.append(_resolve_run_dir(root, exp, rid))

        loaded = load_runs_data(run_dirs)
        decisions = loaded.decisions
        episodes = loaded.episodes
        h2h_matches = loaded.h2h_matches

        decision_ep_keys = {(d.experiment_id, d.row_id, d.episode_id) for d in decisions}
        h2h_decision_ep_keys = {
            (d.experiment_id, d.matchup_id or "", d.episode_id)
            for d in decisions
            if d.experiment_id == "H2H"
        }
        total_spend_usd = (
            sum(d.api_cost_usd for d in decisions)
            + sum(
                e.total_cost_usd
                for e in episodes
                if (e.experiment_id, e.row_id, e.episode_id) not in decision_ep_keys
            )
            + sum(
                m.total_cost_usd
                for m in h2h_matches
                if (m.experiment_id, m.matchup_id, m.episode_id) not in h2h_decision_ep_keys
            )
        )

        eval_decisions = filter_evaluated_decisions(decisions)
        if scope == "suite":
            core_t2 = select_core_t2_decisions(eval_decisions)
            if core_t2:
                lb_pool = core_t2
                source_label = "Core Board Puzzles (Clean Standard · All 9 Squares)"
            else:
                non_t6 = [
                    d for d in eval_decisions if d.experiment_id not in ("T6", "H2H")
                ]
                lb_pool = non_t6 if non_t6 else list(eval_decisions)
                source_label = (
                    "All Puzzle Moves Across Loaded Runs"
                    if non_t6
                    else "All Moves Across Loaded Runs"
                )
        else:
            lb_pool = list(eval_decisions)
            source_label = (
                "Core Board Puzzles (Clean Standard · All 9 Squares)"
                if experiment_id in {"T1", "T2", "T1_T2", "T1+T2"}
                else "Moves in Selected Run"
            )

        leaderboard_rows = _summarize_rows_with_ci(
            lb_pool, n_bootstrap=n_bootstrap, seed=DEFAULT_BOOTSTRAP_SEED
        )
        tactical_breakdown = _compute_tactical_matrix(lb_pool)

        if scope == "suite":
            t3_pool = select_t3_decisions(eval_decisions)
            t5_pool = select_t5_decisions(eval_decisions)
        else:
            t3_pool = (
                list(eval_decisions)
                if len({d.render_variant for d in eval_decisions if d.render_variant}) > 1
                else []
            )
            t5_pool = (
                list(eval_decisions)
                if len({d.text_format for d in eval_decisions if d.text_format}) > 1
                else []
            )

        rendering_robustness = _compute_variant_matrix(
            t3_pool,
            field="render_variant",
            order=RENDER_VARIANT_ORDER,
            default_val="clean_default",
            preferred_exp="T3",
            fallback_exp=experiment_id,
        )
        t5_mat = _compute_variant_matrix(
            t5_pool,
            field="text_format",
            order=TEXT_FORMAT_ORDER,
            default_val="json_canonical",
            preferred_exp="T5",
            fallback_exp=experiment_id,
        )
        prompt_formats = {
            "source_experiment_id": t5_mat["source_experiment_id"],
            "formats": t5_mat["variants"],
            "rows": t5_mat["rows"],
        }

        legality_ablation = _compute_legality_ablation(
            eval_decisions, n_bootstrap=n_bootstrap, seed=DEFAULT_BOOTSTRAP_SEED
        )
        calibration = _compute_calibration_payload(eval_decisions, scope=scope)
        paired_comparisons = _compute_paired_highlights(
            lb_pool, n_bootstrap=n_bootstrap, seed=DEFAULT_BOOTSTRAP_SEED
        )
        full_games_rows = [m.to_dict() for m in compute_track2_metrics(episodes).values()]
        full_games = {
            "source_experiment_id": episodes[0].experiment_id if episodes else None,
            "rows": full_games_rows,
        }
        h2h_matchups = [
            m.to_dict()
            for m in compute_h2h_metrics(
                h2h_matches,
                n_bootstrap=n_bootstrap,
                seed=DEFAULT_BOOTSTRAP_SEED,
            ).values()
        ]
        h2h_standings = [s.to_dict() for s in compute_h2h_standings(h2h_matches)]
        h2h_payload = {
            "source_experiment_id": h2h_matches[0].experiment_id if h2h_matches else None,
            "total_matches": len(h2h_matches),
            "matchups": h2h_matchups,
            "standings": h2h_standings,
        }

        return {
            "scope": scope,
            "mock_mode": target_mock,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "included_runs": [
                {
                    "experiment_id": str(e.get("experiment_id") or ""),
                    "run_id": str(e.get("run_id") or ""),
                    "run_name": e.get("run_name"),
                    "mock_mode": bool(e.get("mock_mode", False)),
                    "pilot_mode": bool(e.get("pilot_mode", False)),
                    "total_decisions": int(e.get("total_decisions") or 0),
                    "total_episodes": int(e.get("total_episodes") or 0),
                }
                for e in included_entries
            ],
            "totals": {
                "total_decisions": len(decisions),
                "total_episodes": len(episodes) + len(h2h_matches),
                "total_spend_usd": total_spend_usd,
            },
            "leaderboard": {
                "source_label": source_label,
                "rows": leaderboard_rows,
            },
            "tactical_breakdown": tactical_breakdown,
            "rendering_robustness": rendering_robustness,
            "legality_ablation": legality_ablation,
            "prompt_formats": prompt_formats,
            "calibration": calibration,
            "paired_comparisons": paired_comparisons,
            "full_games": full_games,
            "h2h": h2h_payload,
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
