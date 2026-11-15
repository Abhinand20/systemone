"""Track 2 closed-loop match runner for T6 (vs random and minimax)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import time
from typing import Any, Literal

from dm_bench.adapters.factory import build_row_adapters, close_adapters
from dm_bench.config import (
    BudgetExceededError,
    BudgetTracker,
    find_repo_root,
    load_core_matrix,
    load_env,
    load_game_config,
    load_prompt_config,
)
from dm_bench.contracts import (
    DecisionLogRow,
    EpisodeLogRow,
    EpisodeResumeKey,
    ImageFrame,
    PolicyRequest,
    PolicyResult,
    RunManifest,
)
from dm_bench.games.tic_tac_toe.engine import (
    INDEX_TO_KEY,
    Mark,
    TTTState,
    legal_action_keys,
    new_ttt_state,
    other_mark,
    step_ttt,
)
from dm_bench.games.tic_tac_toe.formats import format_state
from dm_bench.games.tic_tac_toe.oracle import (
    analyze_position,
    choose_minimax_action,
    evaluate_move,
)
from dm_bench.games.tic_tac_toe.renderer import render_ttt_png
from dm_bench.logging_io import (
    MediaStore,
    RunLogger,
    generate_run_id,
    sha256_json,
    utc_now_iso,
)

OPPONENT_TYPES: tuple[Literal["random", "minimax"], ...] = ("random", "minimax")
DEFAULT_ACTION_MODES: tuple[Literal["all_9", "legal_only"], ...] = ("all_9", "legal_only")
TURN_SCHEDULE_BLOCK: tuple[tuple[Mark, bool], ...] = (
    ("X", True),
    ("O", False),
    ("O", True),
    ("X", False),
)


@dataclass(frozen=True, slots=True)
class _MatchSpec:
    row_id: str
    episode_id: str
    opponent_type: Literal["random", "minimax"]
    action_mode: Literal["all_9", "legal_only"]
    game_idx: int
    episode_seed: int
    agent_mark: Mark
    goes_first: bool
    starting_player: Mark


def _games_per_opponent(action_mode: str, *, pilot: bool) -> int:
    if pilot:
        return 20
    if action_mode == "all_9":
        return 100
    if action_mode == "legal_only":
        return 50
    raise ValueError(f"Unsupported action_mode: {action_mode!r}")


def _deterministic_seed(action_mode: str, opponent_type: str, game_idx: int) -> int:
    digest = hashlib.sha256(f"{action_mode}|{opponent_type}|{game_idx}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


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


def _build_match_schedule(
    row_ids: list[str],
    action_modes: list[Literal["all_9", "legal_only"]],
    *,
    pilot: bool,
) -> list[_MatchSpec]:
    specs: list[_MatchSpec] = []
    for action_mode in action_modes:
        n_games = _games_per_opponent(action_mode, pilot=pilot)
        for opponent_type in OPPONENT_TYPES:
            for game_idx in range(n_games):
                agent_mark, goes_first = TURN_SCHEDULE_BLOCK[game_idx % 4]
                starting_player: Mark = agent_mark if goes_first else other_mark(agent_mark)
                episode_id = f"t6_{action_mode}_{opponent_type}_{game_idx:03d}"
                episode_seed = _deterministic_seed(action_mode, opponent_type, game_idx)
                for row_id in row_ids:
                    specs.append(
                        _MatchSpec(
                            row_id=row_id,
                            episode_id=episode_id,
                            opponent_type=opponent_type,
                            action_mode=action_mode,
                            game_idx=game_idx,
                            episode_seed=episode_seed,
                            agent_mark=agent_mark,
                            goes_first=goes_first,
                            starting_player=starting_player,
                        )
                    )
    return specs


def _choose_opponent_action(
    state: TTTState,
    opponent_type: Literal["random", "minimax"],
    rng: random.Random,
) -> str:
    if opponent_type == "random":
        return rng.choice(legal_action_keys(state.board))
    return choose_minimax_action(state.board, state.player_to_move, rng=rng)


async def run_matches_experiment(
    repo_root: Path | None = None,
    *,
    pilot: bool = False,
    mock: bool = False,
    concurrency: int = 4,
    max_cost_usd: float = 10.0,
    resume_run_id: str | None = None,
    row_ids: list[str] | None = None,
    action_modes: list[str] | None = None,
) -> Path:
    """Run Track 2 closed-loop T6 matches vs random and minimax with checkpoint resume."""
    root = find_repo_root(repo_root)
    load_env(root)

    experiment_id = "T6"
    game_cfg = load_game_config("tic_tac_toe", root)
    prompt_cfg = load_prompt_config(game_cfg.prompt_template, root)
    matrix_cfg = load_core_matrix(root)

    target_row_ids = list(row_ids) if row_ids is not None else matrix_cfg.row_ids()
    selected_modes: list[Literal["all_9", "legal_only"]] = []
    for mode in (action_modes if action_modes is not None else list(DEFAULT_ACTION_MODES)):
        if mode not in ("all_9", "legal_only"):
            raise ValueError(f"Unsupported action_mode: {mode!r}")
        selected_modes.append(mode)  # type: ignore[arg-type]

    results_base = root / "experiments" / "results" / experiment_id
    if resume_run_id is not None:
        run_id = resume_run_id
        run_dir = results_base / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
        logger = RunLogger(run_dir)
        existing_decisions = logger.load_decisions()
        existing_episode_keys: set[EpisodeResumeKey] = logger.load_existing_episode_keys()
        initial_spent_usd = sum(d.api_cost_usd for d in existing_decisions)
        existing_manifest = logger.load_manifest()
    else:
        run_id = generate_run_id(experiment_id)
        run_dir = results_base / run_id
        logger = RunLogger(run_dir)
        existing_episode_keys = set()
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
                "updated_at_utc": now,
                "completed_at_utc": None,
                "max_cost_usd": max_cost_usd,
                "model_configs": model_configs_dump or existing_manifest.model_configs,
            }
        )
    else:
        manifest = RunManifest(
            experiment_id=experiment_id,
            run_id=run_id,
            game=game_cfg.game,
            status="running",
            created_at_utc=now,
            updated_at_utc=now,
            mock_mode=mock,
            pilot_mode=pilot,
            rows=target_row_ids,
            render_variants=["clean_default"],
            text_formats=["json_canonical"],
            action_modes=list(selected_modes),
            max_cost_usd=max_cost_usd,
            total_spent_usd=initial_spent_usd,
            model_configs=model_configs_dump,
        )
    logger.write_manifest(manifest)

    media_store = MediaStore(repo_root=root)
    frame_cache: dict[tuple[tuple[str, ...], Mark], ImageFrame] = {}

    all_specs = _build_match_schedule(target_row_ids, selected_modes, pilot=pilot)
    pending_specs = [
        spec
        for spec in all_specs
        if (spec.row_id, spec.episode_id, spec.opponent_type, spec.action_mode)
        not in existing_episode_keys
    ]

    sem = asyncio.Semaphore(max(1, concurrency))
    budget_halted = False
    run_error: Exception | None = None

    def _get_clean_frame(state: TTTState) -> ImageFrame:
        cache_key = (state.board, state.player_to_move)
        cached = frame_cache.get(cache_key)
        if cached is not None:
            return cached
        png_bytes = render_ttt_png(
            board=state.board,
            player_to_move=state.player_to_move,
            variant="clean_default",
            position_id="t6",
        )
        saved = media_store.save_frame(ImageFrame(png_bytes=png_bytes))
        frame_cache[cache_key] = saved
        return saved

    async def _play_episode(spec: _MatchSpec) -> None:
        nonlocal budget_halted
        async with sem:
            if budget_halted:
                return

            row_cfg, model_cfg, adapter = adapters[spec.row_id]
            opponent_mark = other_mark(spec.agent_mark)
            rng = random.Random(spec.episode_seed)
            state = new_ttt_state(starting_player=spec.starting_player)

            num_agent_turns = 0
            illegal_moves = 0
            invalid_responses = 0
            optimal_agent_turns = 0
            total_cost_usd = 0.0
            total_input_tokens = 0
            total_output_tokens = 0
            total_thought_tokens = 0
            total_latency_ms = 0.0
            effective_regrets: list[float] = []

            outcome: Literal["win", "draw", "loss_by_board", "loss_by_illegal_move"] | None = None
            winner: str | None = None
            termination_reason = "unknown"
            agent_reward = 0.0

            while outcome is None:
                if state.player_to_move != spec.agent_mark:
                    opp_action = _choose_opponent_action(state, spec.opponent_type, rng)
                    opp_step = step_ttt(state, opp_action)
                    state = opp_step.next_state
                    if opp_step.done:
                        if opp_step.termination_reason == "win":
                            outcome = "loss_by_board"
                            winner = opponent_mark
                            termination_reason = "win"
                            agent_reward = -1.0
                        else:
                            outcome = "draw"
                            winner = None
                            termination_reason = "draw"
                            agent_reward = 0.0
                    continue

                if budget_halted:
                    return

                legal_keys = legal_action_keys(state.board)
                action_criteria = prompt_cfg.get_action_criteria(
                    legal_actions=list(legal_keys),
                    action_mode=spec.action_mode,
                )
                render_variant = "clean_default" if row_cfg.use_vision else None
                text_format = "json_canonical" if row_cfg.use_text else None
                frames: list[ImageFrame] = [_get_clean_frame(state)] if row_cfg.use_vision else []
                state_text = (
                    format_state(state, text_format="json_canonical")
                    if row_cfg.use_text
                    else None
                )

                step_idx = len(state.move_history)
                req_id = f"T6:{run_id}:{spec.row_id}:{spec.episode_id}:{step_idx}"
                req = PolicyRequest(
                    request_id=req_id,
                    game=game_cfg.game,
                    episode_id=spec.episode_id,
                    step_idx=step_idx,
                    row_id=row_cfg.row_id,
                    condition_code=row_cfg.condition_code,
                    model_key=model_cfg.model_key,
                    objective_instructions=prompt_cfg.objective_instructions,
                    action_criteria=action_criteria,
                    frames=frames,
                    state_text=state_text,
                    render_variant=render_variant,
                    text_format=text_format,
                    action_mode=spec.action_mode,
                    player_to_move=state.player_to_move,
                    legal_actions=list(legal_keys),
                    metadata={
                        "opponent_type": spec.opponent_type,
                        "agent_mark": spec.agent_mark,
                        "goes_first": spec.goes_first,
                        "episode_seed": spec.episode_seed,
                    },
                )

                try:
                    budget_tracker.check_can_proceed()
                    t0 = time.perf_counter()
                    result = await adapter.decide(req)
                    measured_e2e_ms = (time.perf_counter() - t0) * 1000.0
                except BudgetExceededError:
                    budget_halted = True
                    return
                except Exception as exc:
                    measured_e2e_ms = 0.0
                    result = PolicyResult(
                        requested_action=None,
                        response_valid=False,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )

                analysis = analyze_position(state.board, state.player_to_move)
                candidate_action = result.requested_action if result.response_valid else None
                move_eval = evaluate_move(state.board, state.player_to_move, candidate_action)
                step_out = step_ttt(state, candidate_action)

                is_legal = bool(result.response_valid and step_out.action_is_legal)
                is_optimal = bool(result.response_valid and move_eval.is_optimal)
                executed_action = candidate_action if is_legal else None

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
                        "episode_id": spec.episode_id,
                        "step_idx": step_idx,
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

                decision_row = DecisionLogRow(
                    experiment_id=experiment_id,
                    run_id=run_id,
                    row_id=row_cfg.row_id,
                    condition_code=row_cfg.condition_code,
                    model_key=model_cfg.model_key,
                    model_id=result.model_id or model_cfg.model_id,
                    family=row_cfg.family,
                    episode_id=spec.episode_id,
                    step_idx=step_idx,
                    timestamp_utc=utc_now_iso(),
                    game=game_cfg.game,
                    render_variant=render_variant,
                    text_format=text_format,
                    action_mode=spec.action_mode,
                    frame_paths=frame_paths,
                    frame_hashes=frame_hashes,
                    state_text=state_text,
                    request_payload_hash=payload_hash,
                    action_criteria_keys=list(action_criteria.keys()),
                    board=list(state.board),
                    player_to_move=state.player_to_move,
                    starting_player=state.starting_player,
                    ply=len(state.move_history),
                    move_history=[INDEX_TO_KEY[idx] for idx in state.move_history],
                    legal_actions=list(legal_keys),
                    state_value=analysis.state_value,
                    optimal_actions=list(analysis.optimal_actions),
                    action_values=dict(analysis.action_values),
                    tactical_category=analysis.tactical_category,
                    requested_action=result.requested_action,
                    executed_action=executed_action,
                    response_valid=result.response_valid,
                    action_is_legal=is_legal,
                    action_is_optimal=is_optimal,
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
                    opponent_type=spec.opponent_type,
                    agent_mark=spec.agent_mark,
                    goes_first=spec.goes_first,
                    termination_reason=step_out.termination_reason if step_out.done else None,
                    raw_response=result.raw_response,
                )
                logger.log_decision(decision_row)

                num_agent_turns += 1
                if not result.response_valid:
                    invalid_responses += 1
                if not is_legal:
                    illegal_moves += 1
                if is_optimal:
                    optimal_agent_turns += 1
                total_cost_usd += result.api_cost_usd
                total_input_tokens += result.input_tokens
                total_output_tokens += result.output_tokens
                total_thought_tokens += result.thought_tokens
                total_latency_ms += e2e_latency
                effective_regrets.append(float(move_eval.effective_regret))

                state = step_out.next_state
                if not is_legal:
                    outcome = "loss_by_illegal_move"
                    winner = opponent_mark
                    termination_reason = "illegal_forfeit"
                    agent_reward = -1.0
                elif step_out.done:
                    if step_out.termination_reason == "win":
                        outcome = "win"
                        winner = spec.agent_mark
                        termination_reason = "win"
                        agent_reward = 1.0
                    else:
                        outcome = "draw"
                        winner = None
                        termination_reason = "draw"
                        agent_reward = 0.0

            mean_regret = (
                sum(effective_regrets) / len(effective_regrets)
                if effective_regrets
                else None
            )
            episode_row = EpisodeLogRow(
                experiment_id=experiment_id,
                run_id=run_id,
                episode_id=spec.episode_id,
                row_id=row_cfg.row_id,
                condition_code=row_cfg.condition_code,
                model_key=model_cfg.model_key,
                model_id=model_cfg.model_id,
                family=row_cfg.family,
                game=game_cfg.game,
                action_mode=spec.action_mode,
                opponent_type=spec.opponent_type,
                episode_seed=spec.episode_seed,
                agent_mark=spec.agent_mark,
                starting_player=spec.starting_player,
                goes_first=spec.goes_first,
                outcome=outcome,
                winner=winner,
                termination_reason=termination_reason,
                agent_reward=agent_reward,
                num_plies=len(state.move_history),
                num_agent_turns=num_agent_turns,
                illegal_moves=illegal_moves,
                invalid_responses=invalid_responses,
                optimal_agent_turns=optimal_agent_turns,
                total_cost_usd=round(total_cost_usd, 8),
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_thought_tokens=total_thought_tokens,
                total_latency_ms=round(total_latency_ms, 4),
                mean_effective_regret=mean_regret,
                timestamp_utc=utc_now_iso(),
            )
            logger.log_episode(episode_row)

    try:
        if pending_specs:
            await asyncio.gather(*(_play_episode(spec) for spec in pending_specs))
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
