"""Milestone 0 smoke test: 9-row contract verification and cold/warm latency benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from dm_bench.adapters.factory import build_row_adapters, close_adapters
from dm_bench.config import (
    BudgetExceededError,
    BudgetTracker,
    ConditionRowConfig,
    ModelConfig,
    find_repo_root,
    load_all_model_configs,
    load_core_matrix,
    load_game_config,
    load_prompt_config,
)
from dm_bench.contracts import DecisionLogRow, ImageFrame, PolicyRequest, PolicyResult, RunManifest
from dm_bench.games.tic_tac_toe.corpus import load_corpus
from dm_bench.games.tic_tac_toe.engine import (
    CELL_KEYS,
    INDEX_TO_KEY,
    Mark,
    action_criteria_for_state,
    legal_action_keys,
)
from dm_bench.games.tic_tac_toe.formats import format_ttt_state
from dm_bench.games.tic_tac_toe.oracle import analyze_position, evaluate_move
from dm_bench.games.tic_tac_toe.renderer import render_ttt_png
from dm_bench.logging_io import MediaStore, RunLogger, generate_run_id, utc_now_iso

_REPRESENTATIVE_VT_ROW: dict[str, str] = {
    "jev_1_14_0": "D_jev_VT",
    "gemini_decision_1": "D_gdec_VT",
    "gemini_4_hydrogen": "E_ghyd_VT",
}


def _select_smoke_position(root: Path, dataset_rel_paths: dict[str, str]) -> dict[str, Any]:
    """Select one canonical `forced_block` midgame position from dev/pilot/test corpus."""
    for key in ("dev", "pilot", "test", "ablation"):
        rel = dataset_rel_paths.get(key)
        if not rel:
            continue
        path = root / rel
        if path.is_file():
            for rec in load_corpus(path):
                if rec.get("tactical_category") == "forced_block":
                    return rec

    board = ["X", "X", ".", ".", "O", ".", ".", ".", "."]
    player: Mark = "O"
    analysis = analyze_position(board, player)
    return {
        "position_id": "smoke_forced_block_001",
        "canonical_hash": "smoke_fallback",
        "board": board,
        "player_to_move": player,
        "starting_player": "X",
        "move_history": [0, 4, 1],
        "ply": 3,
        "d4_sym_id": 0,
        "state_value": analysis.state_value,
        "optimal_actions": list(analysis.optimal_actions),
        "action_values": dict(analysis.action_values),
        "tactical_category": analysis.tactical_category,
    }


def _history_to_keys(history: list[int | str]) -> list[str]:
    out: list[str] = []
    for item in history:
        if isinstance(item, int) and not isinstance(item, bool):
            out.append(INDEX_TO_KEY[item])
        else:
            out.append(str(item))
    return out


def _verify_contract(row_cfg: ConditionRowConfig, res: PolicyResult) -> list[str]:
    """Validate Milestone 0 contract invariants for one PolicyResult."""
    errors: list[str] = []
    if not res.response_valid:
        errors.append(f"response_valid=False ({res.error_type}: {res.error_message})")
    if res.requested_action not in CELL_KEYS:
        errors.append(f"requested_action={res.requested_action!r} not in CELL_KEYS")
    if res.input_tokens <= 0:
        errors.append(f"input_tokens={res.input_tokens} must be > 0")
    if res.api_cost_usd is None:
        errors.append("api_cost_usd is None")

    if row_cfg.family == "decision_model":
        probs = res.action_probabilities
        if not isinstance(probs, dict) or set(probs.keys()) != set(CELL_KEYS):
            errors.append("DM action_probabilities must contain all 9 CELL_KEYS")
        else:
            prob_sum = sum(probs.values())
            if abs(prob_sum - 1.0) >= 0.02:
                errors.append(f"DM probabilities sum={prob_sum:.6f} deviates from 1.0")
        if res.confidence is None or not (0.0 <= res.confidence <= 1.0):
            errors.append(f"DM confidence={res.confidence!r} must be in [0, 1]")

    return errors


def _build_decision_log_row(
    *,
    run_id: str,
    row_cfg: ConditionRowConfig,
    model_cfg: ModelConfig,
    request: PolicyRequest,
    result: PolicyResult,
    pos: dict[str, Any],
    legal_keys: list[str],
    move_history_keys: list[str],
    step_idx: int,
) -> DecisionLogRow:
    board = pos["board"]
    player_to_move = pos["player_to_move"]
    eval_res = evaluate_move(board, player_to_move, result.requested_action)
    optimal_actions = list(pos["optimal_actions"])
    legal_set = set(legal_keys)

    optimal_mass: float | None = None
    illegal_mass: float | None = None
    if result.action_probabilities is not None:
        optimal_mass = round(
            sum(result.action_probabilities.get(k, 0.0) for k in optimal_actions), 8
        )
        illegal_mass = round(
            sum(v for k, v in result.action_probabilities.items() if k not in legal_set), 8
        )

    executed_action = (
        result.requested_action
        if (result.response_valid and eval_res.is_legal)
        else None
    )

    return DecisionLogRow(
        experiment_id="M0_smoke",
        run_id=run_id,
        row_id=row_cfg.row_id,
        condition_code=row_cfg.condition_code,
        model_key=model_cfg.model_key,
        model_id=model_cfg.model_id,
        family=row_cfg.family,
        episode_id=request.episode_id,
        step_idx=step_idx,
        timestamp_utc=utc_now_iso(),
        game="tic_tac_toe",
        render_variant=request.render_variant,
        text_format=request.text_format,
        action_mode=request.action_mode,
        frame_paths=[f.path for f in request.frames if f.path],
        frame_hashes=[f.sha256 for f in request.frames if f.sha256],
        state_text=request.state_text,
        request_payload_hash=result.request_payload_hash,
        action_criteria_keys=list(request.action_criteria.keys()),
        board=list(board),
        player_to_move=player_to_move,
        starting_player=pos.get("starting_player"),
        ply=int(pos.get("ply", len(move_history_keys))),
        move_history=move_history_keys,
        legal_actions=legal_keys,
        state_value=int(pos["state_value"]),
        optimal_actions=optimal_actions,
        action_values=dict(pos["action_values"]),
        tactical_category=pos.get("tactical_category"),
        canonical_hash=pos.get("canonical_hash"),
        d4_sym_id=pos.get("d4_sym_id"),
        requested_action=result.requested_action,
        executed_action=executed_action,
        response_valid=result.response_valid,
        action_is_legal=eval_res.is_legal,
        action_is_optimal=eval_res.is_optimal,
        legal_regret=eval_res.legal_regret,
        effective_regret=eval_res.effective_regret,
        action_probabilities=result.action_probabilities,
        confidence=result.confidence,
        optimal_mass=optimal_mass,
        illegal_mass=illegal_mass,
        retries=result.retries,
        error_type=result.error_type,
        error_message=result.error_message,
        model_latency_ms=result.model_latency_ms,
        end_to_end_latency_ms=result.end_to_end_latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        thought_tokens=result.thought_tokens,
        total_tokens=result.total_tokens,
        api_cost_usd=result.api_cost_usd,
        raw_response=result.raw_response,
    )


def _summarize_latencies(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "cold_ms": None,
            "warm_count": 0,
            "warm_p50_ms": None,
            "warm_p95_ms": None,
            "warm_mean_ms": None,
            "all_p50_ms": None,
            "all_p95_ms": None,
        }
    cold = float(values[0])
    warm = values[1:]
    warm_arr = np.asarray(warm, dtype=float) if warm else None
    all_arr = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "cold_ms": round(cold, 3),
        "warm_count": len(warm),
        "warm_p50_ms": round(float(np.percentile(warm_arr, 50)), 3) if warm_arr is not None else None,
        "warm_p95_ms": round(float(np.percentile(warm_arr, 95)), 3) if warm_arr is not None else None,
        "warm_mean_ms": round(float(np.mean(warm_arr)), 3) if warm_arr is not None else None,
        "all_p50_ms": round(float(np.percentile(all_arr, 50)), 3),
        "all_p95_ms": round(float(np.percentile(all_arr, 95)), 3),
    }


def _render_rich_summary(
    console: Console,
    run_dir: Path,
    contract_results: list[dict[str, Any]],
    latency_by_model: dict[str, dict[str, Any]],
    total_spent_usd: float,
    status: str,
) -> None:
    contract_table = Table(title=f"Milestone 0 Contract Checks ({status})")
    contract_table.add_column("Row ID", style="cyan")
    contract_table.add_column("Model")
    contract_table.add_column("Action")
    contract_table.add_column("Legal")
    contract_table.add_column("Optimal")
    contract_table.add_column("In Tok", justify="right")
    contract_table.add_column("Cost ($)", justify="right")
    contract_table.add_column("Contract")
    for item in contract_results:
        contract_table.add_row(
            item["row_id"],
            item["model_key"],
            str(item["requested_action"]),
            str(item["action_is_legal"]),
            str(item["action_is_optimal"]),
            str(item["input_tokens"]),
            f"{item['api_cost_usd']:.6f}",
            "PASS" if item["passed"] else f"FAIL: {'; '.join(item['errors'])}",
        )
    console.print(contract_table)

    if latency_by_model:
        lat_table = Table(title="Cold vs Warm Latency Benchmark")
        lat_table.add_column("Model", style="cyan")
        lat_table.add_column("Row")
        lat_table.add_column("Trials", justify="right")
        lat_table.add_column("Cold E2E (ms)", justify="right")
        lat_table.add_column("Warm p50 (ms)", justify="right")
        lat_table.add_column("Warm p95 (ms)", justify="right")
        lat_table.add_column("Cost ($)", justify="right")
        for model_key, stats in latency_by_model.items():
            e2e = stats["end_to_end_latency_ms"]
            lat_table.add_row(
                model_key,
                stats["benchmark_row_id"],
                str(stats["trials_completed"]),
                f"{e2e['cold_ms']:.1f}" if e2e["cold_ms"] is not None else "-",
                f"{e2e['warm_p50_ms']:.1f}" if e2e["warm_p50_ms"] is not None else "-",
                f"{e2e['warm_p95_ms']:.1f}" if e2e["warm_p95_ms"] is not None else "-",
                f"{stats['total_cost_usd']:.6f}",
            )
        console.print(lat_table)

    console.print(f"[bold]Run directory:[/bold] {run_dir} | [bold]Total spent:[/bold] ${total_spent_usd:.6f}")


async def run_smoke(
    repo_root: Path | None = None,
    *,
    mock: bool = False,
    rows: list[str] | None = None,
    models: list[str] | None = None,
    latency_trials: int = 10,
    max_cost_usd: float = 1.0,
) -> dict[str, Any]:
    """Run Milestone 0 contract checks and cold/warm latency benchmarks."""
    if latency_trials < 0:
        raise ValueError(f"latency_trials must be >= 0, got {latency_trials}")

    root = find_repo_root(repo_root)
    game_cfg = load_game_config("tic_tac_toe", root)
    matrix = load_core_matrix(root)
    all_model_cfgs = load_all_model_configs(root)
    prompt_cfg = load_prompt_config(game_cfg.prompt_template, root)

    selected_rows = [matrix.get_row(rid) for rid in dict.fromkeys(rows)] if rows is not None else list(matrix.rows)
    if models is not None:
        unknown = [m for m in models if m not in all_model_cfgs]
        if unknown:
            raise KeyError(f"Unknown model_key(s): {unknown}")
        model_set = set(models)
        selected_rows = [r for r in selected_rows if r.model_key in model_set]
    if not selected_rows:
        raise ValueError("No condition rows match the requested --rows/--models filters")

    selected_row_ids = [r.row_id for r in selected_rows]
    pos = _select_smoke_position(root, game_cfg.datasets)
    position_id = str(pos["position_id"])
    board: list[str] = list(pos["board"])
    player_to_move: Mark = pos["player_to_move"]
    move_history_keys = _history_to_keys(list(pos.get("move_history", [])))
    legal_keys = list(legal_action_keys(board))
    criteria = action_criteria_for_state(board, "all_9")

    png_bytes = render_ttt_png(
        board=board,
        player_to_move=player_to_move,
        variant="clean_default",
        position_id=position_id,
    )
    media_store = MediaStore(repo_root=root)
    saved_frame = media_store.save_frame(
        ImageFrame(frame_id="board_image", png_bytes=png_bytes, width=512, height=512)
    )
    state_text = format_ttt_state(
        board=board,
        player_to_move=player_to_move,
        text_format="json_canonical",
        move_history=move_history_keys,
    )

    run_id = generate_run_id("m0_smoke")
    run_dir = root / "experiments" / "results" / "M0_smoke" / run_id
    now = utc_now_iso()
    manifest = RunManifest(
        experiment_id="M0_smoke",
        run_id=run_id,
        game="tic_tac_toe",
        status="running",
        created_at_utc=now,
        updated_at_utc=now,
        mock_mode=mock,
        pilot_mode=False,
        rows=selected_row_ids,
        render_variants=["clean_default"],
        text_formats=["json_canonical"],
        action_modes=["all_9"],
        max_cost_usd=max_cost_usd,
        model_configs={
            mk: all_model_cfgs[mk].model_dump(mode="json")
            for mk in dict.fromkeys(r.model_key for r in selected_rows)
        },
        extra={"latency_trials": latency_trials, "position_id": position_id},
    )
    run_logger = RunLogger(run_dir, manifest=manifest)
    budget_tracker = BudgetTracker(max_cost_usd=max_cost_usd)
    adapters = build_row_adapters(
        repo_root=root,
        mock=mock,
        budget_tracker=budget_tracker,
        row_ids=selected_row_ids,
    )

    contract_results: list[dict[str, Any]] = []
    latency_by_model: dict[str, dict[str, Any]] = {}
    halted_by_budget = False

    def make_request(
        row_cfg: ConditionRowConfig,
        model_cfg: ModelConfig,
        episode_id: str,
        step_idx: int,
        phase: str,
    ) -> PolicyRequest:
        return PolicyRequest(
            request_id=f"{run_id}_{row_cfg.row_id}_{episode_id}_{step_idx}",
            game="tic_tac_toe",
            episode_id=episode_id,
            step_idx=step_idx,
            row_id=row_cfg.row_id,
            condition_code=row_cfg.condition_code,
            model_key=model_cfg.model_key,
            objective_instructions=prompt_cfg.objective_instructions,
            action_criteria=criteria,
            frames=[saved_frame] if row_cfg.use_vision else [],
            state_text=state_text if row_cfg.use_text else None,
            render_variant="clean_default" if row_cfg.use_vision else None,
            text_format="json_canonical" if row_cfg.use_text else None,
            action_mode="all_9",
            player_to_move=player_to_move,
            legal_actions=legal_keys,
            metadata={
                "smoke_test": True,
                "phase": phase,
                "board": board,
                "optimal_actions": list(pos["optimal_actions"]),
            },
        )

    try:
        # Phase 1: 1 contract check per selected condition row.
        for row_cfg, model_cfg, adapter in adapters.values():
            req = make_request(
                row_cfg,
                model_cfg,
                episode_id=f"{position_id}_contract",
                step_idx=0,
                phase="contract_check",
            )
            res = await adapter.decide(req)
            log_row = _build_decision_log_row(
                run_id=run_id,
                row_cfg=row_cfg,
                model_cfg=model_cfg,
                request=req,
                result=res,
                pos=pos,
                legal_keys=legal_keys,
                move_history_keys=move_history_keys,
                step_idx=0,
            )
            run_logger.log_decision(log_row)

            contract_errors = _verify_contract(row_cfg, res)
            contract_results.append(
                {
                    "row_id": row_cfg.row_id,
                    "condition_code": row_cfg.condition_code,
                    "model_key": model_cfg.model_key,
                    "family": row_cfg.family,
                    "requested_action": res.requested_action,
                    "response_valid": res.response_valid,
                    "action_is_legal": log_row.action_is_legal,
                    "action_is_optimal": log_row.action_is_optimal,
                    "input_tokens": res.input_tokens,
                    "output_tokens": res.output_tokens,
                    "thought_tokens": res.thought_tokens,
                    "api_cost_usd": res.api_cost_usd,
                    "model_latency_ms": res.model_latency_ms,
                    "end_to_end_latency_ms": res.end_to_end_latency_ms,
                    "passed": len(contract_errors) == 0,
                    "errors": contract_errors,
                }
            )

        # Phase 2: Sequential cold/warm latency trials per active model backend.
        if latency_trials > 0:
            active_models = list(dict.fromkeys(r.model_key for r in selected_rows))
            for model_key in active_models:
                preferred_vt = _REPRESENTATIVE_VT_ROW.get(model_key)
                if preferred_vt and preferred_vt in adapters:
                    bench_row_id = preferred_vt
                else:
                    bench_row_id = next(
                        r.row_id for r in selected_rows if r.model_key == model_key
                    )
                row_cfg, model_cfg, adapter = adapters[bench_row_id]

                model_latencies: list[float] = []
                e2e_latencies: list[float] = []
                trial_costs: list[float] = []
                trial_retries: list[int] = []

                for trial_idx in range(1, latency_trials + 1):
                    req = make_request(
                        row_cfg,
                        model_cfg,
                        episode_id=f"{position_id}_latency_{trial_idx:02d}",
                        step_idx=trial_idx,
                        phase="latency_benchmark",
                    )
                    res = await adapter.decide(req)
                    log_row = _build_decision_log_row(
                        run_id=run_id,
                        row_cfg=row_cfg,
                        model_cfg=model_cfg,
                        request=req,
                        result=res,
                        pos=pos,
                        legal_keys=legal_keys,
                        move_history_keys=move_history_keys,
                        step_idx=trial_idx,
                    )
                    run_logger.log_decision(log_row)

                    if res.model_latency_ms is not None:
                        model_latencies.append(float(res.model_latency_ms))
                    if res.end_to_end_latency_ms is not None:
                        e2e_latencies.append(float(res.end_to_end_latency_ms))
                    trial_costs.append(float(res.api_cost_usd))
                    trial_retries.append(int(res.retries))

                latency_by_model[model_key] = {
                    "benchmark_row_id": bench_row_id,
                    "model_id": model_cfg.model_id,
                    "trials_requested": latency_trials,
                    "trials_completed": len(trial_costs),
                    "total_cost_usd": round(sum(trial_costs), 8),
                    "total_retries": sum(trial_retries),
                    "model_latency_ms": _summarize_latencies(model_latencies),
                    "end_to_end_latency_ms": _summarize_latencies(e2e_latencies),
                }
    except BudgetExceededError:
        halted_by_budget = True
    finally:
        await close_adapters(adapters)

    all_contracts_passed = (
        len(contract_results) == len(selected_rows)
        and all(item["passed"] for item in contract_results)
    )
    if halted_by_budget:
        final_status = "halted_budget_cap"
    elif all_contracts_passed:
        final_status = "completed"
    else:
        final_status = "failed"

    final_manifest = run_logger.finalize(
        status=final_status,  # type: ignore[arg-type]
        total_spent_usd=budget_tracker.spent_usd,
    )

    summary: dict[str, Any] = {
        "experiment_id": "M0_smoke",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": final_manifest.status,
        "mock_mode": mock,
        "position_id": position_id,
        "all_contracts_passed": all_contracts_passed,
        "total_spent_usd": final_manifest.total_spent_usd,
        "total_decisions": final_manifest.total_decisions,
        "contract_checks": contract_results,
        "latency_by_model": latency_by_model,
    }

    latency_summary_path = run_dir / "latency_summary.json"
    latency_summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    _render_rich_summary(
        Console(),
        run_dir,
        contract_results,
        latency_by_model,
        final_manifest.total_spent_usd,
        final_manifest.status,
    )
    return summary
