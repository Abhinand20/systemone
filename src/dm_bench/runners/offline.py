"""Track 1 offline evaluation runner for T1, T2, T3, T4, and T5."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Literal

from rich.console import Console

from dm_bench.adapters.factory import build_row_adapters, close_adapters
from dm_bench.config import (
    BudgetExceededError,
    BudgetTracker,
    ConditionRowConfig,
    ModelConfig,
    find_repo_root,
    load_core_matrix,
    load_env,
    load_game_config,
    load_prompt_config,
)
from dm_bench.contracts import (
    DecisionLogRow,
    DecisionResumeKey,
    ImageFrame,
    PolicyRequest,
    PolicyResult,
    RunManifest,
)
from dm_bench.games.tic_tac_toe.corpus import load_corpus
from dm_bench.games.tic_tac_toe.engine import INDEX_TO_KEY, Mark, legal_action_keys
from dm_bench.games.tic_tac_toe.formats import format_ttt_state
from dm_bench.games.tic_tac_toe.oracle import analyze_position, evaluate_move
from dm_bench.games.tic_tac_toe.renderer import render_ttt_png
from dm_bench.logging_io import (
    MediaStore,
    RunLogger,
    generate_run_id,
    normalize_run_name,
    sha256_json,
    utc_now_iso,
)
from dm_bench.runners.progress import EvalProgressTracker

T3_DEFAULT_ROWS: tuple[str, ...] = ("A_jev_V", "A_gdec_V", "C_ghyd_V")
T3_VARIANTS: tuple[str, ...] = (
    "labeled_margins",
    "dark_low_contrast",
    "thin_alt_font",
    "tilt_blur",
)
T5_DEFAULT_ROWS: tuple[str, ...] = ("B_jev_T", "B_gdec_T", "F_ghyd_T")
T5_FORMATS: tuple[str, ...] = (
    "ascii_grid",
    "coord_list",
    "json_with_history",
)


@dataclass(frozen=True, slots=True)
class _OfflineSpec:
    experiment_id: str
    dataset_key: str
    default_rows: tuple[str, ...] | None
    render_variants: tuple[str, ...]
    text_formats: tuple[str, ...]
    action_mode: Literal["all_9", "legal_only"]


@dataclass(frozen=True, slots=True)
class _OfflineWorkItem:
    record: dict[str, Any]
    row_id: str
    render_variant: str | None
    text_format: str | None
    action_mode: Literal["all_9", "legal_only"]


def _normalize_experiment_id(experiment_id: str) -> str:
    cleaned = experiment_id.strip().upper()
    if cleaned in {"T1", "T2", "T1_T2", "T3", "T4", "T5"}:
        return cleaned
    if cleaned == "T1+T2":
        return "T1_T2"
    raise ValueError(f"Unsupported offline experiment_id: {experiment_id!r}")


def _resolve_offline_spec(
    experiment_id: str,
    *,
    pilot: bool,
    full_240: bool,
) -> _OfflineSpec:
    exp_id = _normalize_experiment_id(experiment_id)
    if exp_id in {"T1", "T2", "T1_T2"}:
        return _OfflineSpec(
            experiment_id=exp_id,
            dataset_key="pilot" if pilot else "test",
            default_rows=None,
            render_variants=("clean_default",),
            text_formats=("json_canonical",),
            action_mode="all_9",
        )
    if exp_id == "T3":
        dataset_key = "pilot" if pilot else ("test" if full_240 else "ablation")
        return _OfflineSpec(
            experiment_id=exp_id,
            dataset_key=dataset_key,
            default_rows=T3_DEFAULT_ROWS,
            render_variants=T3_VARIANTS,
            text_formats=(),
            action_mode="all_9",
        )
    if exp_id == "T4":
        return _OfflineSpec(
            experiment_id=exp_id,
            dataset_key="pilot" if pilot else "test",
            default_rows=None,
            render_variants=("clean_default",),
            text_formats=("json_canonical",),
            action_mode="legal_only",
        )
    # T5
    dataset_key = "pilot" if pilot else ("test" if full_240 else "ablation")
    return _OfflineSpec(
        experiment_id=exp_id,
        dataset_key=dataset_key,
        default_rows=T5_DEFAULT_ROWS,
        render_variants=(),
        text_formats=T5_FORMATS,
        action_mode="all_9",
    )


def _format_move_history(raw_history: list[Any]) -> list[str]:
    out: list[str] = []
    for item in raw_history:
        if isinstance(item, int) and not isinstance(item, bool):
            out.append(INDEX_TO_KEY[item])
        else:
            out.append(str(item))
    return out


def _compute_probability_masses(
    action_probabilities: dict[str, float] | None,
    optimal_actions: tuple[str, ...],
    legal_actions: tuple[str, ...],
) -> tuple[float | None, float | None]:
    if action_probabilities is None:
        return None, None
    opt_set = set(optimal_actions)
    legal_set = set(legal_actions)
    opt_mass = sum(float(p) for k, p in action_probabilities.items() if k in opt_set)
    ill_mass = sum(float(p) for k, p in action_probabilities.items() if k not in legal_set)
    return round(opt_mass, 8), round(ill_mass, 8)


def _build_offline_work_items(
    records: list[dict[str, Any]],
    row_configs: dict[str, ConditionRowConfig],
    spec: _OfflineSpec,
) -> list[_OfflineWorkItem]:
    items: list[_OfflineWorkItem] = []
    for rec in records:
        for row_id, row_cfg in row_configs.items():
            if spec.experiment_id == "T3" and not row_cfg.use_vision:
                continue
            if spec.experiment_id == "T5" and not row_cfg.use_text:
                continue
            render_choices: tuple[str | None, ...] = (
                spec.render_variants if row_cfg.use_vision else (None,)
            )
            text_choices: tuple[str | None, ...] = (
                spec.text_formats if row_cfg.use_text else (None,)
            )
            for rv in render_choices:
                for tf in text_choices:
                    items.append(
                        _OfflineWorkItem(
                            record=rec,
                            row_id=row_id,
                            render_variant=rv,
                            text_format=tf,
                            action_mode=spec.action_mode,
                        )
                    )
    return items


async def run_offline_experiment(
    experiment_id: str,
    repo_root: Path | None = None,
    *,
    pilot: bool = False,
    full_240: bool = False,
    mock: bool = False,
    concurrency: int = 4,
    max_cost_usd: float = 10.0,
    resume_run_id: str | None = None,
    row_ids: list[str] | None = None,
    run_name: str | None = None,
    show_progress: bool = True,
    console: Console | None = None,
) -> Path:
    """Run Track 1 offline evaluation (T1..T5) with bounded concurrency and checkpoint resume."""
    root = find_repo_root(repo_root)
    load_env(root)
    clean_run_name = normalize_run_name(run_name)

    spec = _resolve_offline_spec(experiment_id, pilot=pilot, full_240=full_240)
    game_cfg = load_game_config("tic_tac_toe", root)
    prompt_cfg = load_prompt_config(game_cfg.prompt_template, root)
    matrix_cfg = load_core_matrix(root)

    if row_ids is not None:
        target_row_ids = list(row_ids)
    elif spec.default_rows is not None:
        target_row_ids = list(spec.default_rows)
    else:
        target_row_ids = matrix_cfg.row_ids()

    row_cfg_map = {rid: matrix_cfg.get_row(rid) for rid in target_row_ids}

    rel_dataset_path = game_cfg.datasets[spec.dataset_key]
    dataset_path = root / rel_dataset_path
    records = load_corpus(dataset_path)

    results_base = root / "experiments" / "results" / spec.experiment_id
    if resume_run_id is not None:
        run_id = resume_run_id
        run_dir = results_base / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
        logger = RunLogger(run_dir)
        existing_decisions = logger.load_decisions()
        existing_keys: set[DecisionResumeKey] = {d.resume_key() for d in existing_decisions}
        initial_spent_usd = sum(d.api_cost_usd for d in existing_decisions)
        existing_manifest = logger.load_manifest()
    else:
        run_id = generate_run_id(spec.experiment_id, name=clean_run_name)
        run_dir = results_base / run_id
        logger = RunLogger(run_dir)
        existing_keys = set()
        initial_spent_usd = 0.0
        existing_manifest = None

    budget_tracker = BudgetTracker(
        max_cost_usd=max_cost_usd,
        initial_spent_usd=initial_spent_usd,
    )
    adapters = build_row_adapters(
        root,
        mock=mock,
        budget_tracker=budget_tracker,
        row_ids=target_row_ids,
    )

    model_configs_dump: dict[str, Any] = {}
    for _, model_cfg, _ in adapters.values():
        model_configs_dump[model_cfg.model_key] = model_cfg.model_dump(mode="json")

    now = utc_now_iso()
    if existing_manifest is not None:
        manifest = existing_manifest.model_copy(
            update={
                "status": "running",
                "run_name": (
                    clean_run_name if clean_run_name is not None else existing_manifest.run_name
                ),
                "updated_at_utc": now,
                "completed_at_utc": None,
                "max_cost_usd": max_cost_usd,
                "model_configs": model_configs_dump or existing_manifest.model_configs,
            }
        )
    else:
        manifest = RunManifest(
            experiment_id=spec.experiment_id,
            run_id=run_id,
            run_name=clean_run_name,
            game=game_cfg.game,
            status="running",
            created_at_utc=now,
            updated_at_utc=now,
            mock_mode=mock,
            pilot_mode=pilot,
            dataset_path=rel_dataset_path,
            rows=target_row_ids,
            render_variants=list(spec.render_variants),
            text_formats=list(spec.text_formats),
            action_modes=[spec.action_mode],
            max_cost_usd=max_cost_usd,
            total_spent_usd=initial_spent_usd,
            model_configs=model_configs_dump,
        )
    logger.write_manifest(manifest)

    media_store = MediaStore(repo_root=root)
    all_items = _build_offline_work_items(records, row_cfg_map, spec)
    pending_items = [
        item
        for item in all_items
        if (
            item.row_id,
            item.record["position_id"],
            item.render_variant,
            item.text_format,
            item.action_mode,
        )
        not in existing_keys
    ]

    frame_cache: dict[tuple[str, str], ImageFrame] = {}
    for item in pending_items:
        if item.render_variant is None:
            continue
        pos_id = item.record["position_id"]
        cache_key = (pos_id, item.render_variant)
        if cache_key not in frame_cache:
            png_bytes = render_ttt_png(
                board=item.record["board"],
                player_to_move=item.record["player_to_move"],
                variant=item.render_variant,
                position_id=pos_id,
            )
            saved_frame = media_store.save_frame(ImageFrame(png_bytes=png_bytes))
            frame_cache[cache_key] = saved_frame

    sem_limit = max(1, concurrency)
    sems_by_model: dict[tuple[str, str], asyncio.Semaphore] = {
        (model_cfg.provider, model_cfg.model_key): asyncio.Semaphore(sem_limit)
        for _, model_cfg, _ in adapters.values()
    }
    budget_halted = False
    run_error: Exception | None = None

    progress = EvalProgressTracker(
        experiment_id=spec.experiment_id,
        run_id=run_id,
        unit_label="decisions",
        total_units=len(all_items),
        already_completed=len(all_items) - len(pending_items),
        rows=target_row_ids,
        concurrency=concurrency,
        max_cost_usd=max_cost_usd,
        initial_spent_usd=initial_spent_usd,
        run_dir=run_dir,
        console=console,
        enabled=show_progress,
    )

    async def _process_item(item: _OfflineWorkItem) -> None:
        nonlocal budget_halted
        rec = item.record
        pos_id: str = rec["position_id"]
        board: tuple[str, ...] = tuple(rec["board"])
        player_to_move: Mark = rec["player_to_move"]
        legal_keys = legal_action_keys(board)
        action_criteria = prompt_cfg.get_action_criteria(
            legal_actions=list(legal_keys),
            action_mode=item.action_mode,
        )

        row_cfg, model_cfg, adapter = adapters[item.row_id]
        frames: list[ImageFrame] = []
        if row_cfg.use_vision and item.render_variant is not None:
            frames = [frame_cache[(pos_id, item.render_variant)]]

        state_text: str | None = None
        if row_cfg.use_text and item.text_format is not None:
            state_text = format_ttt_state(
                board=board,
                player_to_move=player_to_move,
                text_format=item.text_format,
                move_history=rec.get("move_history", []),
            )

        req_id = (
            f"{spec.experiment_id}:{run_id}:{item.row_id}:{pos_id}:"
            f"{item.render_variant or 'none'}:{item.text_format or 'none'}:{item.action_mode}"
        )
        req = PolicyRequest(
            request_id=req_id,
            game=game_cfg.game,
            episode_id=pos_id,
            step_idx=0,
            row_id=row_cfg.row_id,
            condition_code=row_cfg.condition_code,
            model_key=model_cfg.model_key,
            objective_instructions=prompt_cfg.objective_instructions,
            action_criteria=action_criteria,
            frames=frames,
            state_text=state_text,
            render_variant=item.render_variant,
            text_format=item.text_format,
            action_mode=item.action_mode,
            player_to_move=player_to_move,
            legal_actions=list(legal_keys),
            metadata={
                "tactical_category": rec.get("tactical_category"),
                "canonical_hash": rec.get("canonical_hash"),
                "ply": rec.get("ply"),
            },
        )

        sem = sems_by_model[(model_cfg.provider, model_cfg.model_key)]
        async with sem:
            if budget_halted:
                return
            try:
                budget_tracker.check_can_proceed()
                t0 = time.perf_counter()
                result = await adapter.decide(req)
                measured_e2e_ms = (time.perf_counter() - t0) * 1000.0
            except BudgetExceededError:
                budget_halted = True
                progress.note_budget_halt()
                return
            except Exception as exc:
                measured_e2e_ms = 0.0
                result = PolicyResult(
                    requested_action=None,
                    response_valid=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

        analysis = analyze_position(board, player_to_move)
        candidate_action = result.requested_action if result.response_valid else None
        move_eval = evaluate_move(board, player_to_move, candidate_action)
        executed_action = result.requested_action if (result.response_valid and move_eval.is_legal) else None

        optimal_mass, illegal_mass = _compute_probability_masses(
            result.action_probabilities,
            analysis.optimal_actions,
            legal_keys,
        )

        frame_paths = [f.path for f in frames if f.path]
        frame_hashes = [f.sha256 for f in frames if f.sha256]
        payload_hash = result.request_payload_hash or sha256_json(
            {
                "row_id": row_cfg.row_id,
                "episode_id": pos_id,
                "action_criteria": action_criteria,
                "state_text": state_text,
                "frame_hashes": frame_hashes,
            }
        )

        e2e_latency = (
            result.end_to_end_latency_ms
            if result.end_to_end_latency_ms is not None
            else measured_e2e_ms
        )
        model_latency = (
            result.model_latency_ms
            if result.model_latency_ms is not None
            else e2e_latency
        )

        log_row = DecisionLogRow(
            experiment_id=spec.experiment_id,
            run_id=run_id,
            row_id=row_cfg.row_id,
            condition_code=row_cfg.condition_code,
            model_key=model_cfg.model_key,
            model_id=result.model_id or model_cfg.model_id,
            family=row_cfg.family,
            episode_id=pos_id,
            step_idx=0,
            timestamp_utc=utc_now_iso(),
            game=game_cfg.game,
            render_variant=item.render_variant,
            text_format=item.text_format,
            action_mode=item.action_mode,
            frame_paths=frame_paths,
            frame_hashes=frame_hashes,
            state_text=state_text,
            request_payload_hash=payload_hash,
            action_criteria_keys=list(action_criteria.keys()),
            board=list(board),
            player_to_move=player_to_move,
            starting_player=rec.get("starting_player"),
            ply=int(rec.get("ply", len(rec.get("move_history", [])))),
            move_history=_format_move_history(rec.get("move_history", [])),
            legal_actions=list(legal_keys),
            state_value=analysis.state_value,
            optimal_actions=list(analysis.optimal_actions),
            action_values=dict(analysis.action_values),
            tactical_category=analysis.tactical_category,
            canonical_hash=rec.get("canonical_hash"),
            d4_sym_id=rec.get("d4_sym_id"),
            requested_action=result.requested_action,
            executed_action=executed_action,
            response_valid=result.response_valid,
            action_is_legal=move_eval.is_legal and result.response_valid,
            action_is_optimal=move_eval.is_optimal and result.response_valid,
            legal_regret=move_eval.legal_regret if result.response_valid else None,
            effective_regret=move_eval.effective_regret,
            minimax_regret=move_eval.effective_regret,
            action_probabilities=result.action_probabilities,
            confidence=result.confidence,
            optimal_mass=optimal_mass,
            illegal_mass=illegal_mass,
            retries=result.retries,
            error_type=result.error_type,
            error_message=result.error_message,
            model_latency_ms=model_latency,
            end_to_end_latency_ms=e2e_latency,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            thought_tokens=result.thought_tokens,
            total_tokens=result.total_tokens,
            api_cost_usd=result.api_cost_usd,
            raw_response=result.raw_response,
        )
        logger.log_decision(log_row)
        progress.record_decision(
            row_id=row_cfg.row_id,
            valid=result.response_valid,
            legal=move_eval.is_legal and result.response_valid,
            cost_usd=result.api_cost_usd,
            error_type=result.error_type,
            error_message=result.error_message,
            episode_id=pos_id,
        )
        progress.advance(1)

    try:
        with progress:
            if pending_items:
                await asyncio.gather(*(_process_item(item) for item in pending_items))
    except Exception as exc:
        run_error = exc
        raise
    finally:
        await close_adapters(adapters)
        if run_error is not None:
            logger.finalize(status="failed")
        elif budget_halted:
            logger.finalize(status="halted_budget_cap")
        else:
            logger.finalize(status="completed")

    return run_dir
