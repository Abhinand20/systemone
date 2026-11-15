"""Head-to-head model-vs-model closed-loop Tic-Tac-Toe runner (`H2H`)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import itertools
from pathlib import Path
import re
import time
from typing import Any, Literal

from rich.console import Console

from dm_bench.adapters.factory import build_row_adapters, close_adapters
from dm_bench.adapters.retry import classify_error, infer_error_category, is_infra_error
from dm_bench.config import (
    BudgetExceededError,
    BudgetTracker,
    ConditionRowConfig,
    CoreMatrixConfig,
    ModelConfig,
    find_repo_root,
    load_all_model_configs,
    load_core_matrix,
    load_env,
    load_game_config,
    load_prompt_config,
)
from dm_bench.contracts import (
    DecisionLogRow,
    H2HMatchLogRow,
    H2HResumeKey,
    ImageFrame,
    Modality,
    PlayerH2HOutcome,
    PolicyRequest,
    PolicyResult,
    RunManifest,
)
from dm_bench.games.tic_tac_toe.engine import (
    CELL_KEYS,
    INDEX_TO_KEY,
    Mark,
    TTTState,
    is_terminal,
    legal_action_keys,
    new_ttt_state,
    step_ttt,
)
from dm_bench.games.tic_tac_toe.formats import format_state
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
from dm_bench.runners.matches import _compute_probability_masses
from dm_bench.runners.progress import EvalProgressTracker

OpeningMode = Literal["standard", "empty"]
DEFAULT_H2H_ACTION_MODES: tuple[Literal["all_9", "legal_only"], ...] = ("all_9",)
DEFAULT_MODALITIES: tuple[Modality, ...] = ("image", "text", "image+text")

# 4-game seat rotation block guaranteeing swapped marks and swapped first movers per pair.
H2H_SEAT_CYCLE: tuple[tuple[Mark, Mark, Mark], ...] = (
    ("X", "X", "O"),  # (starting_mark, mark_a, mark_b)
    ("X", "O", "X"),
    ("O", "O", "X"),
    ("O", "X", "O"),
)

# Diverse 1-ply openings (all 9 squares on an empty 3x3 board have exact minimax state_value == 0).
ONE_PLY_OPENING_ORDER: tuple[str, ...] = (
    "R2C2",
    "R1C1",
    "R1C2",
    "R3C3",
    "R1C3",
    "R2C1",
    "R3C1",
    "R2C3",
    "R3C2",
)

MODEL_ALIASES: dict[str, str] = {
    "jev": "jev_1_14_0",
    "typesafe": "jev_1_14_0",
    "typesafe-jev": "jev_1_14_0",
    "jev-1.14.0": "jev_1_14_0",
    "jev_1_14_0": "jev_1_14_0",
    "gdec": "gemini_decision_1",
    "gemini-decision": "gemini_decision_1",
    "gemini_decision": "gemini_decision_1",
    "gemini-decision-1": "gemini_decision_1",
    "gemini_decision_1": "gemini_decision_1",
    "ghyd": "gemini_4_hydrogen",
    "hydrogen": "gemini_4_hydrogen",
    "gemini-hydrogen": "gemini_4_hydrogen",
    "gemini-4-hydrogen": "gemini_4_hydrogen",
    "gemini_4_hydrogen": "gemini_4_hydrogen",
}

MODALITY_ALIASES: dict[str, Modality] = {
    "v": "image",
    "vision": "image",
    "image": "image",
    "t": "text",
    "text": "text",
    "vt": "image+text",
    "tv": "image+text",
    "hybrid": "image+text",
    "image+text": "image+text",
    "vision+text": "image+text",
    "v+t": "image+text",
}

_VS_SPLIT_RE = re.compile(r"\s+vs\.?\s+", flags=re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class H2HGameSpec:
    """Deterministic schedule slot for a single head-to-head Tic-Tac-Toe game."""

    matchup_id: str
    row_a: ConditionRowConfig
    row_b: ConditionRowConfig
    action_mode: Literal["all_9", "legal_only"]
    game_idx: int
    episode_id: str
    opening_mode: OpeningMode
    opening_moves: tuple[str, ...]
    starting_mark: Mark
    mark_a: Mark
    mark_b: Mark
    a_goes_first: bool


def row_modality(row: ConditionRowConfig) -> Modality:
    """Return canonical modality string for a `ConditionRowConfig`."""
    if row.use_vision and row.use_text:
        return "image+text"
    if row.use_vision:
        return "image"
    return "text"


def resolve_model_key(token: str, model_configs: dict[str, ModelConfig]) -> str:
    """Resolve shorthand model alias (`jev`, `gdec`, `ghyd`), model_id, or exact model_key."""
    cleaned = token.strip()
    if not cleaned:
        raise ValueError("Model identifier cannot be empty")
    if cleaned in model_configs:
        return cleaned

    lowered = cleaned.lower()
    alias_target = MODEL_ALIASES.get(lowered)
    if alias_target and alias_target in model_configs:
        return alias_target

    for key, cfg in model_configs.items():
        if key.lower() == lowered or cfg.model_id.lower() == lowered:
            return key

    available = sorted(set(model_configs.keys()) | {"jev", "gdec", "ghyd"})
    raise ValueError(
        f"Unknown model or alias {token!r}. Supported values include: {', '.join(available)}"
    )


def resolve_modalities(tokens: Sequence[str] | None) -> list[Modality]:
    """Normalize modality tokens (`V`, `T`, `VT`, `image`, `text`, `image+text`)."""
    if tokens is None:
        return list(DEFAULT_MODALITIES)
    out: list[Modality] = []
    for raw in tokens:
        cleaned = raw.strip().lower()
        if not cleaned:
            continue
        mod = MODALITY_ALIASES.get(cleaned)
        if mod is None:
            raise ValueError(
                f"Unsupported modality {raw!r}. Expected one of V/image, T/text, VT/hybrid/image+text."
            )
        if mod not in out:
            out.append(mod)
    if not out:
        raise ValueError("modalities must be non-empty")
    return out


def _split_matchup_token(token: str) -> tuple[str, str]:
    cleaned = token.strip()
    if ":" in cleaned:
        parts = [p.strip() for p in cleaned.split(":")]
    else:
        parts = [p.strip() for p in _VS_SPLIT_RE.split(cleaned)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid matchup specification {token!r}. Use 'lhs:rhs' (e.g. 'jev:gdec' or 'A_jev_V:A_gdec_V') "
            "or 'lhs vs rhs'."
        )
    return parts[0], parts[1]


def resolve_h2h_matchups(
    matrix: CoreMatrixConfig,
    model_configs: dict[str, ModelConfig],
    *,
    matchups: Sequence[str] | None = None,
    rows: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    modalities: Sequence[str] | None = None,
) -> list[tuple[ConditionRowConfig, ConditionRowConfig]]:
    """Resolve CLI/programmatic H2H selections into ordered `(row_a, row_b)` condition pairs."""
    row_by_id = {row.row_id: row for row in matrix.rows}
    row_by_model_mod = {(row.model_key, row_modality(row)): row for row in matrix.rows}
    selected_mods = resolve_modalities(modalities)

    pairs: list[tuple[ConditionRowConfig, ConditionRowConfig]] = []
    seen_pairs: set[tuple[str, str]] = set()

    def _add_pair(row_a: ConditionRowConfig, row_b: ConditionRowConfig) -> None:
        if row_a.row_id == row_b.row_id:
            raise ValueError(
                f"Self-play within the same condition row ({row_a.row_id!r}) is not supported; "
                "choose two different models or condition rows."
            )
        key = (row_a.row_id, row_b.row_id)
        if key not in seen_pairs:
            seen_pairs.add(key)
            pairs.append((row_a, row_b))

    if matchups:
        if rows is not None or models is not None:
            raise ValueError("Cannot combine --matchups with --rows or --models")
        for raw_item in matchups:
            lhs, rhs = _split_matchup_token(raw_item)
            lhs_row = row_by_id.get(lhs)
            rhs_row = row_by_id.get(rhs)

            if lhs_row is not None and rhs_row is not None:
                _add_pair(lhs_row, rhs_row)
            elif lhs_row is not None:
                rhs_model = resolve_model_key(rhs, model_configs)
                mod = row_modality(lhs_row)
                resolved_rhs = row_by_model_mod.get((rhs_model, mod))
                if resolved_rhs is None:
                    raise ValueError(
                        f"Model {rhs!r} ({rhs_model}) has no condition row with modality {mod!r}"
                    )
                _add_pair(lhs_row, resolved_rhs)
            elif rhs_row is not None:
                lhs_model = resolve_model_key(lhs, model_configs)
                mod = row_modality(rhs_row)
                resolved_lhs = row_by_model_mod.get((lhs_model, mod))
                if resolved_lhs is None:
                    raise ValueError(
                        f"Model {lhs!r} ({lhs_model}) has no condition row with modality {mod!r}"
                    )
                _add_pair(resolved_lhs, rhs_row)
            else:
                model_a = resolve_model_key(lhs, model_configs)
                model_b = resolve_model_key(rhs, model_configs)
                if model_a == model_b:
                    raise ValueError(
                        f"Self-play within model {model_a!r} is not supported; provide distinct models or row IDs."
                    )
                added_for_spec = 0
                for mod in selected_mods:
                    ra = row_by_model_mod.get((model_a, mod))
                    rb = row_by_model_mod.get((model_b, mod))
                    if ra is not None and rb is not None:
                        _add_pair(ra, rb)
                        added_for_spec += 1
                if added_for_spec == 0:
                    raise ValueError(
                        f"No overlapping condition rows found for {lhs!r} vs {rhs!r} in modalities {selected_mods}"
                    )
        return pairs

    if rows is not None:
        if models is not None:
            raise ValueError("Specify either --rows or --models, not both")
        selected_rows: list[ConditionRowConfig] = []
        seen_rows: set[str] = set()
        for rid in rows:
            if rid not in row_by_id:
                raise ValueError(
                    f"Unknown condition row_id {rid!r}. Available: {', '.join(matrix.row_ids())}"
                )
            cfg = row_by_id[rid]
            if modalities is not None and row_modality(cfg) not in selected_mods:
                continue
            if rid not in seen_rows:
                seen_rows.add(rid)
                selected_rows.append(cfg)
        if len(selected_rows) < 2:
            raise ValueError("Need at least two distinct condition rows to form H2H matchups")
        for ra, rb in itertools.combinations(selected_rows, 2):
            _add_pair(ra, rb)
        return pairs

    if models is not None:
        resolved_models: list[str] = []
        for tok in models:
            mkey = resolve_model_key(tok, model_configs)
            if mkey not in resolved_models:
                resolved_models.append(mkey)
        if len(resolved_models) < 2:
            raise ValueError("Need at least two distinct models for head-to-head evaluation")
    else:
        resolved_models = list(dict.fromkeys(r.model_key for r in matrix.rows))

    for model_a, model_b in itertools.combinations(resolved_models, 2):
        for mod in selected_mods:
            ra = row_by_model_mod.get((model_a, mod))
            rb = row_by_model_mod.get((model_b, mod))
            if ra is not None and rb is not None:
                _add_pair(ra, rb)

    if not pairs:
        raise ValueError(
            f"No head-to-head matchups could be formed for models={resolved_models} and modalities={selected_mods}"
        )
    return pairs


def _build_two_ply_fair_openings() -> list[tuple[str, ...]]:
    """Build deterministic non-terminal 2-ply openings with exact minimax state_value == 0."""
    prioritized: list[tuple[str, str]] = [
        ("R2C2", "R1C1"),
        ("R1C1", "R2C2"),
        ("R2C2", "R1C3"),
        ("R1C3", "R2C2"),
        ("R2C2", "R3C1"),
        ("R3C1", "R2C2"),
        ("R2C2", "R3C3"),
        ("R3C3", "R2C2"),
        ("R1C2", "R2C2"),
        ("R2C1", "R2C2"),
        ("R2C3", "R2C2"),
        ("R3C2", "R2C2"),
        ("R1C2", "R3C2"),
        ("R2C1", "R2C3"),
    ]
    prioritized_set = set(prioritized)
    candidates = prioritized + [
        (m1, m2)
        for m1 in CELL_KEYS
        for m2 in CELL_KEYS
        if m1 != m2 and (m1, m2) not in prioritized_set
    ]
    fair: list[tuple[str, ...]] = []
    for m1, m2 in candidates:
        state = step_ttt(new_ttt_state("X"), m1).next_state
        state = step_ttt(state, m2).next_state
        if not is_terminal(state.board):
            analysis = analyze_position(state.board, state.player_to_move)
            if analysis.state_value == 0:
                fair.append((m1, m2))
    return fair


def _opening_book_for_blocks(needed_blocks: int, opening_mode: OpeningMode) -> list[tuple[str, ...]]:
    if opening_mode == "empty":
        return [() for _ in range(max(1, needed_blocks))]
    if opening_mode != "standard":
        raise ValueError(f"Unsupported opening_mode: {opening_mode!r}")

    pool: list[tuple[str, ...]] = [(), *((cell,) for cell in ONE_PLY_OPENING_ORDER)]
    if needed_blocks > len(pool):
        pool.extend(_build_two_ply_fair_openings())
    return [pool[idx % len(pool)] for idx in range(max(1, needed_blocks))]


def build_h2h_schedule(
    matchups: Sequence[tuple[ConditionRowConfig, ConditionRowConfig]],
    action_modes: Sequence[Literal["all_9", "legal_only"]],
    n_games: int,
    opening_mode: OpeningMode = "standard",
) -> list[H2HGameSpec]:
    """Build seat-balanced, opening-balanced schedule across matchups and action modes."""
    if n_games < 1:
        raise ValueError(f"games must be >= 1, got {n_games}")
    needed_blocks = (n_games + 3) // 4
    block_openings = _opening_book_for_blocks(needed_blocks, opening_mode)

    specs: list[H2HGameSpec] = []
    for action_mode in action_modes:
        for row_a, row_b in matchups:
            matchup_id = f"{row_a.row_id}_vs_{row_b.row_id}"
            for game_idx in range(n_games):
                block_idx = game_idx // 4
                slot_idx = game_idx % 4
                starting_mark, mark_a, mark_b = H2H_SEAT_CYCLE[slot_idx]
                opening_moves = block_openings[block_idx]

                initial_state = new_ttt_state(starting_player=starting_mark)
                for move_key in opening_moves:
                    initial_state = step_ttt(initial_state, move_key).next_state
                a_goes_first = initial_state.player_to_move == mark_a

                specs.append(
                    H2HGameSpec(
                        matchup_id=matchup_id,
                        row_a=row_a,
                        row_b=row_b,
                        action_mode=action_mode,
                        game_idx=game_idx,
                        episode_id=f"h2h_{action_mode}_{game_idx:03d}",
                        opening_mode=opening_mode,
                        opening_moves=opening_moves,
                        starting_mark=starting_mark,
                        mark_a=mark_a,
                        mark_b=mark_b,
                        a_goes_first=a_goes_first,
                    )
                )
    return specs


async def run_h2h_experiment(
    repo_root: Path | None = None,
    *,
    pilot: bool = False,
    mock: bool = False,
    concurrency: int = 4,
    max_cost_usd: float | None = None,
    resume_run_id: str | None = None,
    models: Sequence[str] | None = None,
    row_ids: Sequence[str] | None = None,
    matchups: Sequence[str] | None = None,
    modalities: Sequence[str] | None = None,
    action_modes: Sequence[str] | None = None,
    games: int | None = None,
    opening_mode: OpeningMode = "standard",
    run_name: str | None = None,
    show_progress: bool = True,
    console: Console | None = None,
) -> Path:
    """Run closed-loop head-to-head model-vs-model evaluation (`H2H`) with checkpoint resume."""
    root = find_repo_root(repo_root)
    load_env(root)
    clean_run_name = normalize_run_name(run_name)

    experiment_id = "H2H"
    game_cfg = load_game_config("tic_tac_toe", root)
    prompt_cfg = load_prompt_config(game_cfg.prompt_template, root)
    matrix_cfg = load_core_matrix(root)
    model_configs = load_all_model_configs(root)

    selected_modes: list[Literal["all_9", "legal_only"]] = []
    for mode in dict.fromkeys(
        action_modes if action_modes is not None else list(DEFAULT_H2H_ACTION_MODES)
    ):
        if mode not in ("all_9", "legal_only"):
            raise ValueError(f"Unsupported action_mode: {mode!r}")
        selected_modes.append(mode)  # type: ignore[arg-type]
    if not selected_modes:
        raise ValueError("action_modes must be non-empty")

    if opening_mode not in ("standard", "empty"):
        raise ValueError(f"Unsupported opening_mode: {opening_mode!r}")

    effective_games = games if games is not None else (20 if pilot else 40)
    if effective_games < 1:
        raise ValueError(f"games must be >= 1, got {effective_games}")

    effective_max_cost = 10.0 if max_cost_usd is None else float(max_cost_usd)

    resolved_pairs = resolve_h2h_matchups(
        matrix_cfg,
        model_configs,
        matchups=matchups,
        rows=row_ids,
        models=models,
        modalities=modalities,
    )
    target_row_ids = list(
        dict.fromkeys(
            rid
            for row_a, row_b in resolved_pairs
            for rid in (row_a.row_id, row_b.row_id)
        )
    )

    results_base = root / "experiments" / "results" / experiment_id
    if resume_run_id is not None:
        run_id = resume_run_id
        run_dir = results_base / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
        logger = RunLogger(run_dir)
        existing_decisions = logger.load_decisions()
        existing_h2h_keys: set[H2HResumeKey] = logger.load_existing_h2h_keys()
        existing_manifest = logger.load_manifest()
        initial_spent_usd = max(
            sum(d.api_cost_usd for d in existing_decisions),
            float(existing_manifest.total_spent_usd) if existing_manifest is not None else 0.0,
        )
    else:
        run_id = generate_run_id(experiment_id, name=clean_run_name)
        run_dir = results_base / run_id
        logger = RunLogger(run_dir)
        existing_h2h_keys = set()
        initial_spent_usd = 0.0
        existing_manifest = None

    budget_tracker = BudgetTracker(
        max_cost_usd=effective_max_cost,
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

    matchup_labels = [f"{a.row_id}:{b.row_id}" for a, b in resolved_pairs]
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
                "max_cost_usd": effective_max_cost,
                "model_configs": model_configs_dump or existing_manifest.model_configs,
            }
        )
    else:
        manifest = RunManifest(
            experiment_id=experiment_id,
            run_id=run_id,
            run_name=clean_run_name,
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
            max_cost_usd=effective_max_cost,
            total_spent_usd=initial_spent_usd,
            model_configs=model_configs_dump,
            extra={
                "matchups": matchup_labels,
                "opening_mode": opening_mode,
                "games_per_matchup": effective_games,
            },
        )
    logger.write_manifest(manifest)

    media_store = MediaStore(repo_root=root)
    frame_cache: dict[tuple[tuple[str, ...], Mark], ImageFrame] = {}

    all_specs = build_h2h_schedule(
        resolved_pairs,
        selected_modes,
        effective_games,
        opening_mode=opening_mode,
    )
    pending_specs = [
        spec
        for spec in all_specs
        if (spec.row_a.row_id, spec.row_b.row_id, spec.episode_id, spec.action_mode)
        not in existing_h2h_keys
    ]

    sem_limit = max(1, concurrency)
    sems_by_model: dict[tuple[str, str], asyncio.Semaphore] = {
        (model_cfg.provider, model_cfg.model_key): asyncio.Semaphore(sem_limit)
        for _, model_cfg, _ in adapters.values()
    }
    game_sem = asyncio.Semaphore(max(1, sem_limit * 2))
    budget_halted = False
    run_error: Exception | None = None

    progress = EvalProgressTracker(
        experiment_id=experiment_id,
        run_id=run_id,
        unit_label="games",
        total_units=len(all_specs),
        already_completed=len(all_specs) - len(pending_specs),
        rows=target_row_ids,
        concurrency=concurrency,
        max_cost_usd=effective_max_cost,
        initial_spent_usd=initial_spent_usd,
        run_dir=run_dir,
        console=console,
        enabled=show_progress,
    )

    def _get_clean_frame(state: TTTState) -> ImageFrame:
        cache_key = (state.board, state.player_to_move)
        cached = frame_cache.get(cache_key)
        if cached is not None:
            return cached
        png_bytes = render_ttt_png(
            board=state.board,
            player_to_move=state.player_to_move,
            variant="clean_default",
            position_id="h2h",
        )
        saved = media_store.save_frame(ImageFrame(png_bytes=png_bytes))
        frame_cache[cache_key] = saved
        return saved

    async def _play_single_h2h_game(spec: H2HGameSpec) -> None:
        nonlocal budget_halted
        row_a, model_a, adapter_a = adapters[spec.row_a.row_id]
        row_b, model_b, adapter_b = adapters[spec.row_b.row_id]

        async with game_sem:
            if budget_halted:
                return

            state = new_ttt_state(starting_player=spec.starting_mark)
            for move_key in spec.opening_moves:
                state = step_ttt(state, move_key).next_state

            pending_decisions: list[DecisionLogRow] = []
            model_turn_idx = 0

            turns_a = 0
            turns_b = 0
            optimal_a = 0
            optimal_b = 0
            illegal_a = 0
            illegal_b = 0
            invalid_a = 0
            invalid_b = 0
            regrets_a: list[float] = []
            regrets_b: list[float] = []
            in_tok_a = out_tok_a = th_tok_a = 0
            in_tok_b = out_tok_b = th_tok_b = 0
            cost_a = cost_b = 0.0
            lat_a = lat_b = 0.0

            outcome: Literal["player_a_win", "player_b_win", "draw"] | None = None
            outcome_a: PlayerH2HOutcome = "draw"
            outcome_b: PlayerH2HOutcome = "draw"
            winner_row_id: str | None = None
            winner_mark: Mark | None = None
            forfeited_by_row_id: str | None = None
            termination_reason: Literal["win", "draw", "illegal_forfeit"] = "draw"

            while outcome is None:
                if budget_halted:
                    return

                active_is_a = state.player_to_move == spec.mark_a
                if active_is_a:
                    active_row, active_model, active_adapter = row_a, model_a, adapter_a
                    opp_row, opp_model = row_b, model_b
                    active_mark, opp_mark = spec.mark_a, spec.mark_b
                    active_goes_first = spec.a_goes_first
                else:
                    active_row, active_model, active_adapter = row_b, model_b, adapter_b
                    opp_row, opp_model = row_a, model_a
                    active_mark, opp_mark = spec.mark_b, spec.mark_a
                    active_goes_first = not spec.a_goes_first

                legal_keys = legal_action_keys(state.board)
                action_criteria = prompt_cfg.get_action_criteria(
                    legal_actions=list(legal_keys),
                    action_mode=spec.action_mode,
                )
                render_variant = "clean_default" if active_row.use_vision else None
                text_format = "json_canonical" if active_row.use_text else None
                frames: list[ImageFrame] = [_get_clean_frame(state)] if active_row.use_vision else []
                state_text = (
                    format_state(state, text_format="json_canonical")
                    if active_row.use_text
                    else None
                )

                ply = len(state.move_history)
                req_id = (
                    f"H2H:{run_id}:{spec.matchup_id}:{spec.episode_id}:"
                    f"{model_turn_idx}:{active_row.row_id}"
                )
                req = PolicyRequest(
                    request_id=req_id,
                    game=game_cfg.game,
                    episode_id=spec.episode_id,
                    step_idx=model_turn_idx,
                    row_id=active_row.row_id,
                    condition_code=active_row.condition_code,
                    model_key=active_model.model_key,
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
                        "matchup_id": spec.matchup_id,
                        "opponent_row_id": opp_row.row_id,
                        "opponent_model_key": opp_model.model_key,
                        "agent_mark": active_mark,
                        "goes_first": active_goes_first,
                        "opening_mode": spec.opening_mode,
                        "opening_moves": list(spec.opening_moves),
                    },
                )

                sem = sems_by_model[(active_model.provider, active_model.model_key)]
                async with sem:
                    if budget_halted:
                        return
                    try:
                        budget_tracker.check_can_proceed()
                        t0 = time.perf_counter()
                        result = await active_adapter.decide(req)
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
                            error_category=classify_error(exc),
                        )

                analysis = analyze_position(state.board, state.player_to_move)
                candidate_action = result.requested_action if result.response_valid else None
                move_eval = evaluate_move(state.board, state.player_to_move, candidate_action)
                step_out = step_ttt(state, candidate_action)

                decision_error_category = (
                    None
                    if result.response_valid
                    else infer_error_category(
                        result.error_type,
                        result.error_message,
                        result.error_category,
                    )
                )
                is_infra_failure = bool(
                    not result.response_valid
                    and is_infra_error(
                        result.error_type,
                        result.error_message,
                        decision_error_category,
                    )
                )
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
                        "row_id": active_row.row_id,
                        "matchup_id": spec.matchup_id,
                        "episode_id": spec.episode_id,
                        "step_idx": model_turn_idx,
                        "ply": ply,
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

                turn_termination: str | None = None
                if is_infra_failure:
                    turn_termination = decision_error_category
                elif not is_legal:
                    turn_termination = "illegal_forfeit"
                elif step_out.done:
                    turn_termination = step_out.termination_reason

                decision_row = DecisionLogRow(
                    experiment_id=experiment_id,
                    run_id=run_id,
                    row_id=active_row.row_id,
                    condition_code=active_row.condition_code,
                    model_key=active_model.model_key,
                    model_id=result.model_id or active_model.model_id,
                    family=active_row.family,
                    episode_id=spec.episode_id,
                    step_idx=model_turn_idx,
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
                    ply=ply,
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
                    effective_regret=None if is_infra_failure else move_eval.effective_regret,
                    minimax_regret=None if is_infra_failure else move_eval.effective_regret,
                    action_probabilities=result.action_probabilities,
                    confidence=result.confidence,
                    optimal_mass=optimal_mass,
                    illegal_mass=illegal_mass,
                    retries=result.retries,
                    error_type=result.error_type,
                    error_message=result.error_message,
                    error_category=decision_error_category,
                    model_latency_ms=model_latency,
                    end_to_end_latency_ms=e2e_latency,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    thought_tokens=result.thought_tokens,
                    total_tokens=result.total_tokens,
                    api_cost_usd=result.api_cost_usd,
                    opponent_type=f"h2h:{opp_row.row_id}",
                    opponent_row_id=opp_row.row_id,
                    opponent_model_key=opp_model.model_key,
                    matchup_id=spec.matchup_id,
                    agent_mark=active_mark,
                    goes_first=active_goes_first,
                    termination_reason=turn_termination,
                    raw_response=result.raw_response,
                )
                pending_decisions.append(decision_row)
                progress.record_decision(
                    row_id=active_row.row_id,
                    valid=result.response_valid,
                    legal=is_legal,
                    cost_usd=result.api_cost_usd,
                    error_type=result.error_type,
                    error_message=result.error_message,
                    episode_id=f"{spec.matchup_id}/{spec.episode_id}",
                )

                if is_infra_failure:
                    progress.advance(1)
                    return

                if active_is_a:
                    turns_a += 1
                    if not result.response_valid:
                        invalid_a += 1
                    if not is_legal:
                        illegal_a += 1
                    if is_optimal:
                        optimal_a += 1
                    regrets_a.append(float(move_eval.effective_regret))
                    in_tok_a += result.input_tokens
                    out_tok_a += result.output_tokens
                    th_tok_a += result.thought_tokens
                    cost_a += result.api_cost_usd
                    lat_a += e2e_latency
                else:
                    turns_b += 1
                    if not result.response_valid:
                        invalid_b += 1
                    if not is_legal:
                        illegal_b += 1
                    if is_optimal:
                        optimal_b += 1
                    regrets_b.append(float(move_eval.effective_regret))
                    in_tok_b += result.input_tokens
                    out_tok_b += result.output_tokens
                    th_tok_b += result.thought_tokens
                    cost_b += result.api_cost_usd
                    lat_b += e2e_latency

                model_turn_idx += 1
                state = step_out.next_state

                if not is_legal:
                    termination_reason = "illegal_forfeit"
                    forfeited_by_row_id = active_row.row_id
                    winner_row_id = opp_row.row_id
                    winner_mark = opp_mark
                    if active_is_a:
                        outcome = "player_b_win"
                        outcome_a = "loss_by_illegal_move"
                        outcome_b = "win_by_forfeit"
                    else:
                        outcome = "player_a_win"
                        outcome_a = "win_by_forfeit"
                        outcome_b = "loss_by_illegal_move"
                elif step_out.done:
                    if step_out.termination_reason == "win":
                        termination_reason = "win"
                        winner_row_id = active_row.row_id
                        winner_mark = active_mark
                        if active_is_a:
                            outcome = "player_a_win"
                            outcome_a = "win"
                            outcome_b = "loss"
                        else:
                            outcome = "player_b_win"
                            outcome_a = "loss"
                            outcome_b = "win"
                    else:
                        termination_reason = "draw"
                        outcome = "draw"
                        outcome_a = "draw"
                        outcome_b = "draw"
                        winner_row_id = None
                        winner_mark = None

            for drow in pending_decisions:
                logger.log_decision(drow)

            match_row = H2HMatchLogRow(
                run_id=run_id,
                experiment_id=experiment_id,
                matchup_id=spec.matchup_id,
                episode_id=spec.episode_id,
                game_idx=spec.game_idx,
                action_mode=spec.action_mode,
                opening_mode=spec.opening_mode,
                opening_moves=list(spec.opening_moves),
                row_a=row_a.row_id,
                condition_code_a=row_a.condition_code,
                model_key_a=model_a.model_key,
                model_id_a=model_a.model_id,
                modality_a=row_modality(row_a),
                mark_a=spec.mark_a,
                row_b=row_b.row_id,
                condition_code_b=row_b.condition_code,
                model_key_b=model_b.model_key,
                model_id_b=model_b.model_id,
                modality_b=row_modality(row_b),
                mark_b=spec.mark_b,
                starting_mark=spec.starting_mark,
                a_goes_first=spec.a_goes_first,
                outcome=outcome,
                outcome_a=outcome_a,
                outcome_b=outcome_b,
                winner_row_id=winner_row_id,
                winner_mark=winner_mark,
                forfeited_by_row_id=forfeited_by_row_id,
                termination_reason=termination_reason,
                num_plies=len(state.move_history),
                num_model_turns=turns_a + turns_b,
                turns_a=turns_a,
                turns_b=turns_b,
                optimal_turns_a=optimal_a,
                optimal_turns_b=optimal_b,
                illegal_moves_a=illegal_a,
                illegal_moves_b=illegal_b,
                invalid_responses_a=invalid_a,
                invalid_responses_b=invalid_b,
                mean_effective_regret_a=(sum(regrets_a) / len(regrets_a)) if regrets_a else 0.0,
                mean_effective_regret_b=(sum(regrets_b) / len(regrets_b)) if regrets_b else 0.0,
                input_tokens_a=in_tok_a,
                output_tokens_a=out_tok_a,
                thought_tokens_a=th_tok_a,
                input_tokens_b=in_tok_b,
                output_tokens_b=out_tok_b,
                thought_tokens_b=th_tok_b,
                cost_usd_a=round(cost_a, 8),
                cost_usd_b=round(cost_b, 8),
                total_cost_usd=round(cost_a + cost_b, 8),
                latency_ms_a=round(lat_a, 4),
                latency_ms_b=round(lat_b, 4),
                total_latency_ms=round(lat_a + lat_b, 4),
                final_board="".join(state.board),
                move_history=[INDEX_TO_KEY[idx] for idx in state.move_history],
                timestamp_utc=utc_now_iso(),
            )
            logger.log_h2h_match(match_row)
            progress.advance(1)

    try:
        with progress:
            if pending_specs:
                await asyncio.gather(*(_play_single_h2h_game(spec) for spec in pending_specs))
    except Exception as exc:
        run_error = exc
        raise
    finally:
        await close_adapters(adapters)
        final_spent = max(
            budget_tracker.spent_usd,
            sum(d.api_cost_usd for d in logger.load_decisions()),
        )
        if run_error is not None:
            logger.finalize(status="failed", total_spent_usd=final_spent)
        elif budget_halted:
            logger.finalize(status="halted_budget_cap", total_spent_usd=final_spent)
        else:
            logger.finalize(status="completed", total_spent_usd=final_spent)

    return run_dir
