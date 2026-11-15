"""Statistical aggregation, paired bootstrap CIs, McNemar tests, and DM calibration."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dm_bench.adapters.retry import infer_error_category, is_infra_error
from dm_bench.contracts import DecisionLogRow, EpisodeLogRow, H2HMatchLogRow, RunManifest
from dm_bench.logging_io import RunLogger, reconcile_decisions, reconcile_episodes_with_decisions

DEFAULT_BOOTSTRAP_SAMPLES: int = 10_000
DEFAULT_BOOTSTRAP_SEED: int = 20261114

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

TACTICAL_CATEGORY_ORDER: tuple[str, ...] = (
    "immediate_win",
    "forced_block",
    "fork_create",
    "fork_defend",
    "forced_draw",
    "losing_position",
)

RENDER_VARIANT_ORDER: tuple[str, ...] = (
    "clean_default",
    "labeled_margins",
    "dark_low_contrast",
    "thin_alt_font",
    "tilt_blur",
)

TEXT_FORMAT_ORDER: tuple[str, ...] = (
    "json_canonical",
    "ascii_grid",
    "coord_list",
    "json_with_history",
)


def sort_row_ids(row_ids: Sequence[str]) -> list[str]:
    """Sort row IDs using canonical matrix ordering first, then lexicographically."""
    order_map = {row_id: idx for idx, row_id in enumerate(CORE_ROW_ORDER)}
    return sorted(set(row_ids), key=lambda rid: (order_map.get(rid, len(CORE_ROW_ORDER)), rid))


@dataclass(frozen=True, slots=True)
class LoadedRunsData:
    """Container for deduplicated decision logs, episode logs, H2H matches, and run manifests."""

    run_dirs: tuple[Path, ...]
    decisions: tuple[DecisionLogRow, ...]
    episodes: tuple[EpisodeLogRow, ...]
    manifests: tuple[RunManifest, ...]
    h2h_matches: tuple[H2HMatchLogRow, ...] = ()


def _resolve_run_dirs(run_dirs: Sequence[Path | str]) -> list[Path]:
    """Expand experiment parent folders to child run directories when needed."""
    resolved: list[Path] = []
    seen: set[Path] = set()
    run_marker_files = ("decisions.jsonl", "episodes.jsonl", "h2h_matches.jsonl", "manifest.yaml")
    for raw in run_dirs:
        path = Path(raw)
        has_run_files = any((path / name).is_file() for name in run_marker_files)
        if has_run_files or not path.is_dir():
            norm = path.resolve()
            if norm not in seen:
                seen.add(norm)
                resolved.append(path)
            continue

        child_runs = [
            child
            for child in sorted(path.iterdir())
            if child.is_dir() and any((child / name).is_file() for name in run_marker_files)
        ]
        if child_runs:
            for child in child_runs:
                norm = child.resolve()
                if norm not in seen:
                    seen.add(norm)
                    resolved.append(child)
        else:
            norm = path.resolve()
            if norm not in seen:
                seen.add(norm)
                resolved.append(path)
    return resolved


def load_runs_data(run_dirs: Sequence[Path | str]) -> LoadedRunsData:
    """Load and deduplicate decisions, episodes, H2H matches, and manifests from run directories."""
    resolved_dirs = _resolve_run_dirs(run_dirs)
    decisions_map: dict[
        tuple[str, str, str, int, str | None, str | None, str, str | None, str | None],
        DecisionLogRow,
    ] = {}
    episodes_map: dict[tuple[str, str, str, str, str], EpisodeLogRow] = {}
    h2h_map: dict[tuple[str, str, str, str, str], H2HMatchLogRow] = {}
    manifests: list[RunManifest] = []

    for run_dir in resolved_dirs:
        logger = RunLogger(run_dir)
        manifest = logger.load_manifest()
        if manifest is not None:
            manifests.append(manifest)
        for row in logger.load_decisions():
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
            decisions_map[dkey] = row
        for ep in logger.load_episodes():
            ekey = (
                ep.experiment_id,
                ep.row_id,
                ep.episode_id,
                ep.opponent_type,
                ep.action_mode,
            )
            episodes_map[ekey] = ep
        for match in logger.load_h2h_matches():
            hkey = (
                match.experiment_id,
                match.row_a,
                match.row_b,
                match.episode_id,
                match.action_mode,
            )
            h2h_map[hkey] = match

    reconciled_decisions = reconcile_decisions(decisions_map.values())
    reconciled_episodes = reconcile_episodes_with_decisions(
        episodes_map.values(), reconciled_decisions
    )

    return LoadedRunsData(
        run_dirs=tuple(resolved_dirs),
        decisions=tuple(reconciled_decisions),
        episodes=tuple(reconciled_episodes),
        manifests=tuple(manifests),
        h2h_matches=tuple(h2h_map.values()),
    )


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """Single-sample bootstrap confidence interval result."""

    mean: float
    ci_low: float
    ci_high: float
    n: int
    n_bootstrap: int
    seed: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairedBootstrapCI:
    """Paired two-sample bootstrap confidence interval for mean_x - mean_y."""

    mean_x: float
    mean_y: float
    delta: float
    ci_low: float
    ci_high: float
    n: int
    n_bootstrap: int
    seed: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_ci(
    values: Sequence[float | int | bool],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Compute percentile bootstrap confidence interval for the sample mean."""
    if not values:
        raise ValueError("Cannot compute bootstrap CI on empty sequence")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    chunk_size = 2000
    offset = 0
    while offset < n_bootstrap:
        count = min(chunk_size, n_bootstrap - offset)
        idx = rng.integers(0, n, size=(count, n))
        boot_means[offset : offset + count] = arr[idx].mean(axis=1)
        offset += count

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.percentile(boot_means, 100.0 * alpha))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return BootstrapCI(
        mean=mean_val,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
        n_bootstrap=n_bootstrap,
        seed=seed,
        confidence=confidence,
    )


def paired_bootstrap_ci(
    values_x: Sequence[float | int | bool],
    values_y: Sequence[float | int | bool],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> PairedBootstrapCI:
    """Compute paired bootstrap CI for delta = mean(x) - mean(y) over shared indices."""
    if len(values_x) != len(values_y):
        raise ValueError(
            f"Paired sequences must have equal length, got {len(values_x)} and {len(values_y)}"
        )
    if not values_x:
        raise ValueError("Cannot compute paired bootstrap CI on empty sequences")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    arr_x = np.asarray(values_x, dtype=np.float64)
    arr_y = np.asarray(values_y, dtype=np.float64)
    diffs = arr_x - arr_y
    n = len(diffs)
    mean_x = float(np.mean(arr_x))
    mean_y = float(np.mean(arr_y))
    delta = float(mean_x - mean_y)

    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(n_bootstrap, dtype=np.float64)
    chunk_size = 2000
    offset = 0
    while offset < n_bootstrap:
        count = min(chunk_size, n_bootstrap - offset)
        idx = rng.integers(0, n, size=(count, n))
        boot_deltas[offset : offset + count] = diffs[idx].mean(axis=1)
        offset += count

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.percentile(boot_deltas, 100.0 * alpha))
    ci_high = float(np.percentile(boot_deltas, 100.0 * (1.0 - alpha)))
    return PairedBootstrapCI(
        mean_x=mean_x,
        mean_y=mean_y,
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
        n_bootstrap=n_bootstrap,
        seed=seed,
        confidence=confidence,
    )


def mcnemar_exact_from_counts(n10: int, n01: int) -> float:
    """Compute exact two-sided McNemar p-value from discordant counts (n10, n01)."""
    if n10 < 0 or n01 < 0:
        raise ValueError(f"Discordant counts must be non-negative, got ({n10}, {n01})")
    total = n10 + n01
    if total == 0:
        return 1.0
    k = min(n10, n01)
    log_half_n = -total * math.log(2.0)
    log_fact_n = math.lgamma(total + 1)
    tail = math.fsum(
        math.exp(log_fact_n - math.lgamma(i + 1) - math.lgamma(total - i + 1) + log_half_n)
        for i in range(k + 1)
    )
    return min(1.0, 2.0 * tail)


def mcnemar_exact_pvalue(
    outcomes_x: Sequence[bool | int],
    outcomes_y: Sequence[bool | int],
) -> float:
    """Compute exact two-sided McNemar p-value on aligned binary outcomes."""
    if len(outcomes_x) != len(outcomes_y):
        raise ValueError(
            f"Outcome sequences must have equal length, got {len(outcomes_x)} and {len(outcomes_y)}"
        )
    if not outcomes_x:
        return 1.0
    n10 = 0
    n01 = 0
    for x_val, y_val in zip(outcomes_x, outcomes_y, strict=True):
        bx = bool(x_val)
        by = bool(y_val)
        if bx and not by:
            n10 += 1
        elif not bx and by:
            n01 += 1
    return mcnemar_exact_from_counts(n10, n01)


Track1Key = tuple[str, str, str | None, str | None, str]
Track2Key = tuple[str, str, str]
H2HMatchupKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class Track1Metrics:
    """Aggregated Track 1 metrics for one `(experiment_id, row_id, render_variant, text_format, action_mode)` slice."""

    experiment_id: str
    row_id: str
    condition_code: str
    model_key: str
    model_id: str
    family: str
    render_variant: str | None
    text_format: str | None
    action_mode: str
    n: int
    optimal_count: int
    legal_count: int
    illegal_count: int
    invalid_count: int
    optimal_accuracy: float
    conditional_optimal_accuracy: float | None
    illegal_rate: float
    invalid_response_rate: float
    mean_effective_regret: float | None
    mean_legal_regret: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    mean_latency_ms: float | None
    model_p50_latency_ms: float | None
    model_p95_latency_ms: float | None
    model_mean_latency_ms: float | None
    total_cost_usd: float
    cost_per_100_decisions_usd: float
    optimal_decisions_per_usd: float | None
    infra_error_count: int = 0
    rate_limit_error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Track2Metrics:
    """Aggregated Track 2 (`T6`) metrics for one `(row_id, action_mode, opponent_type)` slice."""

    row_id: str
    condition_code: str
    model_key: str
    model_id: str
    family: str
    action_mode: str
    opponent_type: str
    games: int
    wins: int
    draws: int
    losses_by_board: int
    losses_by_illegal_move: int
    win_rate: float
    draw_rate: float
    non_loss_rate: float
    loss_by_board_rate: float
    loss_by_illegal_move_rate: float
    total_cost_usd: float
    cost_per_episode_usd: float
    non_loss_games_per_usd: float | None
    mean_plies: float
    mean_agent_turns: float
    mean_latency_ms: float
    mean_effective_regret: float | None
    aborted_errors: int = 0
    rate_limit_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_infra_decision(row: DecisionLogRow) -> bool:
    """Return True when a decision failed due to infrastructure, API, or rate-limit errors."""
    return bool(
        not row.response_valid
        and is_infra_error(row.error_type, row.error_message, row.error_category)
    )


def filter_evaluated_decisions(decisions: Sequence[DecisionLogRow]) -> list[DecisionLogRow]:
    """Return decisions excluding infrastructure, API, and rate-limit failures."""
    return [d for d in decisions if not is_infra_decision(d)]


def _summarize_decision_group(
    key: Track1Key,
    rows: Sequence[DecisionLogRow],
) -> Track1Metrics:
    experiment_id, row_id, render_variant, text_format, action_mode = key
    first = rows[0]
    infra_rows = [r for r in rows if is_infra_decision(r)]
    eval_rows = [r for r in rows if not is_infra_decision(r)]
    infra_error_count = len(infra_rows)
    rate_limit_error_count = sum(
        1
        for r in infra_rows
        if infer_error_category(r.error_type, r.error_message, r.error_category) == "rate_limit"
    )
    n = len(eval_rows)

    optimal_count = sum(1 for r in eval_rows if r.action_is_optimal)
    legal_rows = [r for r in eval_rows if r.action_is_legal]
    legal_count = len(legal_rows)
    illegal_count = sum(1 for r in eval_rows if r.response_valid and not r.action_is_legal)
    invalid_count = sum(1 for r in eval_rows if not r.response_valid)

    optimal_accuracy = optimal_count / n if n > 0 else 0.0
    conditional_optimal_accuracy = (
        sum(1 for r in legal_rows if r.action_is_optimal) / legal_count
        if legal_count > 0
        else None
    )
    illegal_rate = illegal_count / n if n > 0 else 0.0
    invalid_response_rate = invalid_count / n if n > 0 else 0.0

    eff_regrets = [
        float(r.effective_regret)
        for r in eval_rows
        if r.effective_regret is not None
    ]
    mean_effective_regret = float(np.mean(eff_regrets)) if eff_regrets else None

    legal_regrets = [
        float(r.legal_regret)
        for r in legal_rows
        if r.legal_regret is not None
    ]
    mean_legal_regret = float(np.mean(legal_regrets)) if legal_regrets else None

    e2e_latencies = [
        float(r.end_to_end_latency_ms if r.end_to_end_latency_ms is not None else r.model_latency_ms)
        for r in eval_rows
        if (r.end_to_end_latency_ms is not None or r.model_latency_ms is not None)
    ]
    if e2e_latencies:
        lat_arr = np.asarray(e2e_latencies, dtype=np.float64)
        p50_latency_ms: float | None = float(np.percentile(lat_arr, 50))
        p95_latency_ms: float | None = float(np.percentile(lat_arr, 95))
        mean_latency_ms: float | None = float(np.mean(lat_arr))
    else:
        p50_latency_ms = None
        p95_latency_ms = None
        mean_latency_ms = None

    model_latencies = [float(r.model_latency_ms) for r in eval_rows if r.model_latency_ms is not None]
    if model_latencies:
        mlat_arr = np.asarray(model_latencies, dtype=np.float64)
        model_p50_latency_ms: float | None = float(np.percentile(mlat_arr, 50))
        model_p95_latency_ms: float | None = float(np.percentile(mlat_arr, 95))
        model_mean_latency_ms: float | None = float(np.mean(mlat_arr))
    else:
        model_p50_latency_ms = None
        model_p95_latency_ms = None
        model_mean_latency_ms = None

    total_cost_usd = float(sum(r.api_cost_usd for r in eval_rows))
    cost_per_100_decisions_usd = (total_cost_usd / n) * 100.0 if n > 0 else 0.0
    optimal_decisions_per_usd = (
        optimal_count / total_cost_usd if total_cost_usd > 0.0 else None
    )

    return Track1Metrics(
        experiment_id=experiment_id,
        row_id=row_id,
        condition_code=first.condition_code,
        model_key=first.model_key,
        model_id=first.model_id,
        family=first.family,
        render_variant=render_variant,
        text_format=text_format,
        action_mode=action_mode,
        n=n,
        optimal_count=optimal_count,
        legal_count=legal_count,
        illegal_count=illegal_count,
        invalid_count=invalid_count,
        optimal_accuracy=optimal_accuracy,
        conditional_optimal_accuracy=conditional_optimal_accuracy,
        illegal_rate=illegal_rate,
        invalid_response_rate=invalid_response_rate,
        mean_effective_regret=mean_effective_regret,
        mean_legal_regret=mean_legal_regret,
        p50_latency_ms=p50_latency_ms,
        p95_latency_ms=p95_latency_ms,
        mean_latency_ms=mean_latency_ms,
        model_p50_latency_ms=model_p50_latency_ms,
        model_p95_latency_ms=model_p95_latency_ms,
        model_mean_latency_ms=model_mean_latency_ms,
        total_cost_usd=total_cost_usd,
        cost_per_100_decisions_usd=cost_per_100_decisions_usd,
        optimal_decisions_per_usd=optimal_decisions_per_usd,
        infra_error_count=infra_error_count,
        rate_limit_error_count=rate_limit_error_count,
    )


def compute_track1_metrics(
    decisions: Sequence[DecisionLogRow],
) -> dict[Track1Key, Track1Metrics]:
    """Compute Track 1 metrics grouped by `(experiment_id, row_id, render_variant, text_format, action_mode)`."""
    grouped: dict[Track1Key, list[DecisionLogRow]] = defaultdict(list)
    for row in decisions:
        key: Track1Key = (
            row.experiment_id,
            row.row_id,
            row.render_variant,
            row.text_format,
            row.action_mode,
        )
        grouped[key].append(row)

    results: dict[Track1Key, Track1Metrics] = {}
    for key in sorted(
        grouped.keys(),
        key=lambda k: (
            k[0],
            CORE_ROW_ORDER.index(k[1]) if k[1] in CORE_ROW_ORDER else len(CORE_ROW_ORDER),
            k[1],
            k[2] or "",
            k[3] or "",
            k[4],
        ),
    ):
        results[key] = _summarize_decision_group(key, grouped[key])
    return results


def compute_tactical_metrics(
    decisions: Sequence[DecisionLogRow],
) -> dict[tuple[str, str], Track1Metrics]:
    """Compute Track 1 metrics stratified by `(row_id, tactical_category)`."""
    grouped: dict[tuple[str, str], list[DecisionLogRow]] = defaultdict(list)
    for row in decisions:
        if not row.tactical_category:
            continue
        grouped[(row.row_id, row.tactical_category)].append(row)

    results: dict[tuple[str, str], Track1Metrics] = {}
    for (row_id, category), rows in grouped.items():
        first = rows[0]
        synth_key: Track1Key = (
            first.experiment_id,
            row_id,
            first.render_variant,
            first.text_format,
            first.action_mode,
        )
        results[(row_id, category)] = _summarize_decision_group(synth_key, rows)
    return results


def compute_track2_metrics(
    episodes: Sequence[EpisodeLogRow],
) -> dict[Track2Key, Track2Metrics]:
    """Compute Track 2 (`T6`) metrics grouped by `(row_id, action_mode, opponent_type)`."""
    grouped: dict[Track2Key, list[EpisodeLogRow]] = defaultdict(list)
    for ep in episodes:
        key: Track2Key = (ep.row_id, ep.action_mode, ep.opponent_type)
        grouped[key].append(ep)

    results: dict[Track2Key, Track2Metrics] = {}
    for key in sorted(
        grouped.keys(),
        key=lambda k: (
            CORE_ROW_ORDER.index(k[0]) if k[0] in CORE_ROW_ORDER else len(CORE_ROW_ORDER),
            k[0],
            k[1],
            k[2],
        ),
    ):
        eps = grouped[key]
        row_id, action_mode, opponent_type = key
        first = eps[0]
        aborted_eps = [e for e in eps if e.outcome == "aborted_error"]
        eval_eps = [e for e in eps if e.outcome != "aborted_error"]
        aborted_errors = len(aborted_eps)
        rate_limit_errors = sum(
            1
            for e in aborted_eps
            if e.termination_reason == "rate_limit"
            or infer_error_category(e.error_type, e.error_message, e.error_category) == "rate_limit"
        )
        games = len(eval_eps)
        wins = sum(1 for e in eval_eps if e.outcome == "win")
        draws = sum(1 for e in eval_eps if e.outcome == "draw")
        losses_by_board = sum(1 for e in eval_eps if e.outcome == "loss_by_board")
        losses_by_illegal_move = sum(
            1 for e in eval_eps if e.outcome == "loss_by_illegal_move"
        )
        non_losses = wins + draws
        total_cost_usd = float(sum(e.total_cost_usd for e in eval_eps))

        eff_regrets = [
            float(e.mean_effective_regret)
            for e in eval_eps
            if e.mean_effective_regret is not None
        ]

        results[key] = Track2Metrics(
            row_id=row_id,
            condition_code=first.condition_code,
            model_key=first.model_key,
            model_id=first.model_id,
            family=first.family,
            action_mode=action_mode,
            opponent_type=opponent_type,
            games=games,
            wins=wins,
            draws=draws,
            losses_by_board=losses_by_board,
            losses_by_illegal_move=losses_by_illegal_move,
            win_rate=wins / games if games > 0 else 0.0,
            draw_rate=draws / games if games > 0 else 0.0,
            non_loss_rate=non_losses / games if games > 0 else 0.0,
            loss_by_board_rate=losses_by_board / games if games > 0 else 0.0,
            loss_by_illegal_move_rate=losses_by_illegal_move / games if games > 0 else 0.0,
            total_cost_usd=total_cost_usd,
            cost_per_episode_usd=total_cost_usd / games if games > 0 else 0.0,
            non_loss_games_per_usd=(
                non_losses / total_cost_usd if total_cost_usd > 0.0 else None
            ),
            mean_plies=float(np.mean([e.num_plies for e in eval_eps])) if eval_eps else 0.0,
            mean_agent_turns=float(np.mean([e.num_agent_turns for e in eval_eps])) if eval_eps else 0.0,
            mean_latency_ms=float(np.mean([e.total_latency_ms for e in eval_eps])) if eval_eps else 0.0,
            mean_effective_regret=float(np.mean(eff_regrets)) if eff_regrets else None,
            aborted_errors=aborted_errors,
            rate_limit_errors=rate_limit_errors,
        )
    return results


@dataclass(frozen=True, slots=True)
class H2HMatchupMetrics:
    """Aggregated Head-to-Head (`H2H`) metrics for one `(row_a, row_b, action_mode)` matchup."""

    matchup_id: str
    row_a: str
    row_b: str
    condition_code_a: str
    condition_code_b: str
    model_key_a: str
    model_key_b: str
    model_id_a: str
    model_id_b: str
    modality_a: str
    modality_b: str
    action_mode: str
    games: int
    wins_a: int
    wins_b: int
    draws: int
    board_wins_a: int
    board_wins_b: int
    forfeit_wins_a: int
    forfeit_wins_b: int
    win_rate_a: float
    win_rate_b: float
    draw_rate: float
    score_a: float
    score_a_ci_low: float
    score_a_ci_high: float
    score_b: float
    p_value_decisive: float
    turns_a: int
    turns_b: int
    optimal_turns_a: int
    optimal_turns_b: int
    optimal_rate_a: float
    optimal_rate_b: float
    illegal_moves_a: int
    illegal_moves_b: int
    invalid_responses_a: int
    invalid_responses_b: int
    mean_effective_regret_a: float
    mean_effective_regret_b: float
    mean_latency_ms_a: float
    mean_latency_ms_b: float
    cost_usd_a: float
    cost_usd_b: float
    total_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class H2HStandingMetrics:
    """Overall H2H arena standing aggregated per `row_id` across all opponents played."""

    row_id: str
    condition_code: str
    model_key: str
    model_id: str
    modality: str
    games: int
    wins: int
    draws: int
    losses: int
    board_wins: int
    forfeit_wins: int
    losses_by_board: int
    losses_by_illegal_move: int
    points: float
    score_rate: float
    win_rate: float
    draw_rate: float
    optimal_rate: float
    mean_effective_regret: float
    total_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _row_sort_index(row_id: str) -> tuple[int, str]:
    idx = CORE_ROW_ORDER.index(row_id) if row_id in CORE_ROW_ORDER else len(CORE_ROW_ORDER)
    return (idx, row_id)


def compute_h2h_metrics(
    matches: Sequence[H2HMatchLogRow],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[H2HMatchupKey, H2HMatchupMetrics]:
    """Compute head-to-head matchup metrics grouped by `(row_a, row_b, action_mode)`."""
    if not matches:
        return {}

    grouped: dict[H2HMatchupKey, list[H2HMatchLogRow]] = defaultdict(list)
    for m in matches:
        grouped[(m.row_a, m.row_b, m.action_mode)].append(m)

    results: dict[H2HMatchupKey, H2HMatchupMetrics] = {}
    for key in sorted(
        grouped.keys(),
        key=lambda k: (_row_sort_index(k[0]), _row_sort_index(k[1]), k[2]),
    ):
        grp = grouped[key]
        row_a, row_b, action_mode = key
        first = grp[0]
        games = len(grp)

        wins_a = sum(1 for m in grp if m.outcome == "player_a_win")
        wins_b = sum(1 for m in grp if m.outcome == "player_b_win")
        draws = sum(1 for m in grp if m.outcome == "draw")

        board_wins_a = sum(1 for m in grp if m.outcome_a == "win")
        board_wins_b = sum(1 for m in grp if m.outcome_b == "win")
        forfeit_wins_a = sum(1 for m in grp if m.outcome_a == "win_by_forfeit")
        forfeit_wins_b = sum(1 for m in grp if m.outcome_b == "win_by_forfeit")

        per_game_scores_a = [
            1.0 if m.outcome == "player_a_win" else (0.5 if m.outcome == "draw" else 0.0)
            for m in grp
        ]
        score_ci = bootstrap_ci(
            per_game_scores_a,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        score_a = score_ci.mean
        p_val = mcnemar_exact_from_counts(wins_a, wins_b) if (wins_a + wins_b) > 0 else 1.0

        turns_a = sum(m.turns_a for m in grp)
        turns_b = sum(m.turns_b for m in grp)
        opt_a = sum(m.optimal_turns_a for m in grp)
        opt_b = sum(m.optimal_turns_b for m in grp)

        regret_sum_a = sum(m.mean_effective_regret_a * m.turns_a for m in grp)
        regret_sum_b = sum(m.mean_effective_regret_b * m.turns_b for m in grp)
        lat_sum_a = sum(m.latency_ms_a for m in grp)
        lat_sum_b = sum(m.latency_ms_b for m in grp)
        cost_a = float(sum(m.cost_usd_a for m in grp))
        cost_b = float(sum(m.cost_usd_b for m in grp))

        results[key] = H2HMatchupMetrics(
            matchup_id=first.matchup_id,
            row_a=row_a,
            row_b=row_b,
            condition_code_a=first.condition_code_a,
            condition_code_b=first.condition_code_b,
            model_key_a=first.model_key_a,
            model_key_b=first.model_key_b,
            model_id_a=first.model_id_a,
            model_id_b=first.model_id_b,
            modality_a=first.modality_a,
            modality_b=first.modality_b,
            action_mode=action_mode,
            games=games,
            wins_a=wins_a,
            wins_b=wins_b,
            draws=draws,
            board_wins_a=board_wins_a,
            board_wins_b=board_wins_b,
            forfeit_wins_a=forfeit_wins_a,
            forfeit_wins_b=forfeit_wins_b,
            win_rate_a=wins_a / games,
            win_rate_b=wins_b / games,
            draw_rate=draws / games,
            score_a=score_a,
            score_a_ci_low=score_ci.ci_low,
            score_a_ci_high=score_ci.ci_high,
            score_b=1.0 - score_a,
            p_value_decisive=p_val,
            turns_a=turns_a,
            turns_b=turns_b,
            optimal_turns_a=opt_a,
            optimal_turns_b=opt_b,
            optimal_rate_a=(opt_a / turns_a) if turns_a > 0 else 0.0,
            optimal_rate_b=(opt_b / turns_b) if turns_b > 0 else 0.0,
            illegal_moves_a=sum(m.illegal_moves_a for m in grp),
            illegal_moves_b=sum(m.illegal_moves_b for m in grp),
            invalid_responses_a=sum(m.invalid_responses_a for m in grp),
            invalid_responses_b=sum(m.invalid_responses_b for m in grp),
            mean_effective_regret_a=(regret_sum_a / turns_a) if turns_a > 0 else 0.0,
            mean_effective_regret_b=(regret_sum_b / turns_b) if turns_b > 0 else 0.0,
            mean_latency_ms_a=(lat_sum_a / turns_a) if turns_a > 0 else 0.0,
            mean_latency_ms_b=(lat_sum_b / turns_b) if turns_b > 0 else 0.0,
            cost_usd_a=cost_a,
            cost_usd_b=cost_b,
            total_cost_usd=float(sum(m.total_cost_usd for m in grp)),
        )
    return results


def compute_h2h_standings(
    matches: Sequence[H2HMatchLogRow],
    *,
    action_mode: str | None = None,
) -> list[H2HStandingMetrics]:
    """Compute overall H2H standings per `row_id` aggregated across all opponents."""
    filtered = [m for m in matches if action_mode is None or m.action_mode == action_mode]
    if not filtered:
        return []

    accum: dict[str, dict[str, Any]] = {}

    def _record_side(
        row_id: str,
        condition_code: str,
        model_key: str,
        model_id: str,
        modality: str,
        outcome: PlayerH2HOutcome | str,
        turns: int,
        optimal_turns: int,
        mean_regret: float,
        cost_usd: float,
    ) -> None:
        bucket = accum.setdefault(
            row_id,
            {
                "row_id": row_id,
                "condition_code": condition_code,
                "model_key": model_key,
                "model_id": model_id,
                "modality": modality,
                "games": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "board_wins": 0,
                "forfeit_wins": 0,
                "losses_by_board": 0,
                "losses_by_illegal_move": 0,
                "turns": 0,
                "optimal_turns": 0,
                "regret_turn_sum": 0.0,
                "total_cost_usd": 0.0,
            },
        )
        bucket["games"] += 1
        if outcome in ("win", "win_by_forfeit"):
            bucket["wins"] += 1
            if outcome == "win":
                bucket["board_wins"] += 1
            else:
                bucket["forfeit_wins"] += 1
        elif outcome == "draw":
            bucket["draws"] += 1
        else:
            bucket["losses"] += 1
            if outcome == "loss":
                bucket["losses_by_board"] += 1
            elif outcome == "loss_by_illegal_move":
                bucket["losses_by_illegal_move"] += 1
        bucket["turns"] += turns
        bucket["optimal_turns"] += optimal_turns
        bucket["regret_turn_sum"] += mean_regret * turns
        bucket["total_cost_usd"] += cost_usd

    for m in filtered:
        _record_side(
            m.row_a,
            m.condition_code_a,
            m.model_key_a,
            m.model_id_a,
            m.modality_a,
            m.outcome_a,
            m.turns_a,
            m.optimal_turns_a,
            m.mean_effective_regret_a,
            m.cost_usd_a,
        )
        _record_side(
            m.row_b,
            m.condition_code_b,
            m.model_key_b,
            m.model_id_b,
            m.modality_b,
            m.outcome_b,
            m.turns_b,
            m.optimal_turns_b,
            m.mean_effective_regret_b,
            m.cost_usd_b,
        )

    standings: list[H2HStandingMetrics] = []
    for bucket in accum.values():
        games = bucket["games"]
        turns = bucket["turns"]
        points = bucket["wins"] + 0.5 * bucket["draws"]
        standings.append(
            H2HStandingMetrics(
                row_id=bucket["row_id"],
                condition_code=bucket["condition_code"],
                model_key=bucket["model_key"],
                model_id=bucket["model_id"],
                modality=bucket["modality"],
                games=games,
                wins=bucket["wins"],
                draws=bucket["draws"],
                losses=bucket["losses"],
                board_wins=bucket["board_wins"],
                forfeit_wins=bucket["forfeit_wins"],
                losses_by_board=bucket["losses_by_board"],
                losses_by_illegal_move=bucket["losses_by_illegal_move"],
                points=points,
                score_rate=points / games,
                win_rate=bucket["wins"] / games,
                draw_rate=bucket["draws"] / games,
                optimal_rate=(bucket["optimal_turns"] / turns) if turns > 0 else 0.0,
                mean_effective_regret=(bucket["regret_turn_sum"] / turns) if turns > 0 else 0.0,
                total_cost_usd=bucket["total_cost_usd"],
            )
        )

    standings.sort(
        key=lambda s: (
            -s.score_rate,
            -s.wins,
            s.losses_by_illegal_move,
            _row_sort_index(s.row_id),
        )
    )
    return standings


@dataclass(frozen=True, slots=True)
class PairedComparisonResult:
    """Paired statistical comparison between two conditions on shared `episode_id` positions."""

    label: str
    row_x: str
    row_y: str
    shared_n: int
    optimal_ci: PairedBootstrapCI
    mcnemar_pvalue: float
    n10: int
    n01: int
    illegal_ci: PairedBootstrapCI
    effective_regret_ci: PairedBootstrapCI | None
    latency_ci: PairedBootstrapCI | None
    cost_ci: PairedBootstrapCI

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "row_x": self.row_x,
            "row_y": self.row_y,
            "shared_n": self.shared_n,
            "optimal_ci": self.optimal_ci.to_dict(),
            "mcnemar_pvalue": self.mcnemar_pvalue,
            "n10": self.n10,
            "n01": self.n01,
            "illegal_ci": self.illegal_ci.to_dict(),
            "effective_regret_ci": (
                self.effective_regret_ci.to_dict() if self.effective_regret_ci else None
            ),
            "latency_ci": self.latency_ci.to_dict() if self.latency_ci else None,
            "cost_ci": self.cost_ci.to_dict(),
        }


def compare_paired_decisions(
    rows_x: Sequence[DecisionLogRow],
    rows_y: Sequence[DecisionLogRow],
    *,
    label: str = "",
    row_x_label: str = "",
    row_y_label: str = "",
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PairedComparisonResult | None:
    """Compare two decision sets aligned on shared `episode_id` positions."""
    map_x = {r.episode_id: r for r in filter_evaluated_decisions(rows_x)}
    map_y = {r.episode_id: r for r in filter_evaluated_decisions(rows_y)}
    shared_ids = sorted(set(map_x.keys()) & set(map_y.keys()))
    if not shared_ids:
        return None

    aligned_x = [map_x[eid] for eid in shared_ids]
    aligned_y = [map_y[eid] for eid in shared_ids]

    opt_x = [1 if r.action_is_optimal else 0 for r in aligned_x]
    opt_y = [1 if r.action_is_optimal else 0 for r in aligned_y]
    opt_ci = paired_bootstrap_ci(opt_x, opt_y, n_bootstrap=n_bootstrap, seed=seed)

    n10 = sum(1 for bx, by in zip(opt_x, opt_y, strict=True) if bx and not by)
    n01 = sum(1 for bx, by in zip(opt_x, opt_y, strict=True) if not bx and by)
    pval = mcnemar_exact_from_counts(n10, n01)

    ill_x = [1 if (r.response_valid and not r.action_is_legal) else 0 for r in aligned_x]
    ill_y = [1 if (r.response_valid and not r.action_is_legal) else 0 for r in aligned_y]
    illegal_ci = paired_bootstrap_ci(ill_x, ill_y, n_bootstrap=n_bootstrap, seed=seed)

    reg_pairs = [
        (float(rx.effective_regret), float(ry.effective_regret))
        for rx, ry in zip(aligned_x, aligned_y, strict=True)
        if rx.effective_regret is not None and ry.effective_regret is not None
    ]
    effective_regret_ci = (
        paired_bootstrap_ci(
            [p[0] for p in reg_pairs],
            [p[1] for p in reg_pairs],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        if reg_pairs
        else None
    )

    lat_pairs = []
    for rx, ry in zip(aligned_x, aligned_y, strict=True):
        lx = rx.end_to_end_latency_ms if rx.end_to_end_latency_ms is not None else rx.model_latency_ms
        ly = ry.end_to_end_latency_ms if ry.end_to_end_latency_ms is not None else ry.model_latency_ms
        if lx is not None and ly is not None:
            lat_pairs.append((float(lx), float(ly)))
    latency_ci = (
        paired_bootstrap_ci(
            [p[0] for p in lat_pairs],
            [p[1] for p in lat_pairs],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        if lat_pairs
        else None
    )

    cost_ci = paired_bootstrap_ci(
        [float(r.api_cost_usd) for r in aligned_x],
        [float(r.api_cost_usd) for r in aligned_y],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    rx_name = row_x_label or aligned_x[0].row_id
    ry_name = row_y_label or aligned_y[0].row_id
    comp_label = label or f"{rx_name} vs {ry_name}"
    return PairedComparisonResult(
        label=comp_label,
        row_x=rx_name,
        row_y=ry_name,
        shared_n=len(shared_ids),
        optimal_ci=opt_ci,
        mcnemar_pvalue=pval,
        n10=n10,
        n01=n01,
        illegal_ci=illegal_ci,
        effective_regret_ci=effective_regret_ci,
        latency_ci=latency_ci,
        cost_ci=cost_ci,
    )


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One fixed-width calibration reliability bin."""

    bin_index: int
    bin_low: float
    bin_high: float
    bin_center: float
    count: int
    mean_confidence: float | None
    empirical_accuracy: float | None
    calibration_gap: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AbstentionPoint:
    """One threshold point along the selective abstention curve."""

    tau: float
    coverage: float
    retained_count: int
    abstained_count: int
    accuracy: float | None
    illegal_rate: float | None
    hybrid_oracle_accuracy: float
    hybrid_fallback_accuracy: float | None = None
    hybrid_fallback_cost_per_100_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DMCalibrationResult:
    """Multi-optimal calibration and selective abstention metrics for Decision Models."""

    n: int
    ece: float
    brier_score: float
    top1_prob_ece: float
    top1_prob_brier_score: float
    mean_confidence: float
    mean_top1_prob: float
    empirical_accuracy: float
    mean_optimal_mass: float | None
    mean_illegal_mass: float | None
    reliability_bins: tuple[ReliabilityBin, ...]
    top1_prob_bins: tuple[ReliabilityBin, ...]
    abstention_curve: tuple[AbstentionPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "ece": self.ece,
            "brier_score": self.brier_score,
            "top1_prob_ece": self.top1_prob_ece,
            "top1_prob_brier_score": self.top1_prob_brier_score,
            "mean_confidence": self.mean_confidence,
            "mean_top1_prob": self.mean_top1_prob,
            "empirical_accuracy": self.empirical_accuracy,
            "mean_optimal_mass": self.mean_optimal_mass,
            "mean_illegal_mass": self.mean_illegal_mass,
            "reliability_bins": [b.to_dict() for b in self.reliability_bins],
            "top1_prob_bins": [b.to_dict() for b in self.top1_prob_bins],
            "abstention_curve": [p.to_dict() for p in self.abstention_curve],
        }


def _compute_reliability_bins(
    scores: Sequence[float],
    labels: Sequence[float],
    n_bins: int,
) -> tuple[float, float, tuple[ReliabilityBin, ...]]:
    n = len(scores)
    bin_scores: list[list[float]] = [[] for _ in range(n_bins)]
    bin_labels: list[list[float]] = [[] for _ in range(n_bins)]

    for score, label in zip(scores, labels, strict=True):
        clamped = min(1.0, max(0.0, float(score)))
        idx = min(n_bins - 1, int(clamped * n_bins))
        bin_scores[idx].append(clamped)
        bin_labels[idx].append(float(label))

    bins: list[ReliabilityBin] = []
    ece = 0.0
    for idx in range(n_bins):
        low = idx / n_bins
        high = (idx + 1) / n_bins
        center = (low + high) / 2.0
        cnt = len(bin_scores[idx])
        if cnt > 0:
            mean_conf = float(np.mean(bin_scores[idx]))
            emp_acc = float(np.mean(bin_labels[idx]))
            gap = abs(emp_acc - mean_conf)
            ece += (cnt / n) * gap
        else:
            mean_conf = None
            emp_acc = None
            gap = 0.0
        bins.append(
            ReliabilityBin(
                bin_index=idx,
                bin_low=low,
                bin_high=high,
                bin_center=center,
                count=cnt,
                mean_confidence=mean_conf,
                empirical_accuracy=emp_acc,
                calibration_gap=gap,
            )
        )

    brier = float(
        np.mean([(float(s) - float(y)) ** 2 for s, y in zip(scores, labels, strict=True)])
    )
    return float(ece), brier, tuple(bins)


def compute_dm_calibration(
    decisions: Sequence[DecisionLogRow],
    *,
    n_bins: int = 10,
    taus: Sequence[float] | None = None,
    fallback_decisions: Sequence[DecisionLogRow] | None = None,
) -> DMCalibrationResult:
    """Compute top-1 reliability (ECE, Brier), multi-optimal/illegal mass, and abstention sweep."""
    eval_decisions = filter_evaluated_decisions(decisions)
    if not eval_decisions:
        raise ValueError("Cannot compute DM calibration on empty decision sequence")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    decisions = eval_decisions
    if fallback_decisions is not None:
        fallback_decisions = filter_evaluated_decisions(fallback_decisions)

    tau_grid = (
        [float(t) for t in taus]
        if taus is not None
        else [round(i * 0.05, 4) for i in range(21)]
    )

    confs: list[float] = []
    top1_probs: list[float] = []
    labels: list[float] = []
    optimal_masses: list[float] = []
    illegal_masses: list[float] = []

    for row in decisions:
        labels.append(1.0 if row.action_is_optimal else 0.0)

        if row.action_probabilities:
            if row.requested_action and row.requested_action in row.action_probabilities:
                p1 = float(row.action_probabilities[row.requested_action])
            else:
                p1 = float(max(row.action_probabilities.values()))
        elif row.confidence is not None:
            p1 = float(row.confidence)
        else:
            p1 = 0.0
        p1 = min(1.0, max(0.0, p1))
        top1_probs.append(p1)

        if row.confidence is not None:
            c = min(1.0, max(0.0, float(row.confidence)))
        else:
            c = p1
        confs.append(c)

        if row.optimal_mass is not None:
            optimal_masses.append(float(row.optimal_mass))
        elif row.action_probabilities is not None and row.optimal_actions:
            optimal_masses.append(
                float(sum(row.action_probabilities.get(a, 0.0) for a in row.optimal_actions))
            )

        if row.illegal_mass is not None:
            illegal_masses.append(float(row.illegal_mass))
        elif row.action_probabilities is not None and row.legal_actions:
            legal_set = set(row.legal_actions)
            illegal_masses.append(
                float(sum(p for k, p in row.action_probabilities.items() if k not in legal_set))
            )

    ece, brier, rel_bins = _compute_reliability_bins(confs, labels, n_bins=n_bins)
    top1_ece, top1_brier, top1_bins = _compute_reliability_bins(
        top1_probs, labels, n_bins=n_bins
    )

    fallback_map = (
        {fb.episode_id: fb for fb in fallback_decisions}
        if fallback_decisions is not None
        else None
    )

    n = len(decisions)
    abstention_points: list[AbstentionPoint] = []
    for tau in tau_grid:
        retained = [
            row for row, c in zip(decisions, confs, strict=True) if c >= tau
        ]
        retained_cnt = len(retained)
        abstained_cnt = n - retained_cnt
        coverage = retained_cnt / n
        if retained_cnt > 0:
            retained_opt = sum(1 for r in retained if r.action_is_optimal)
            acc: float | None = retained_opt / retained_cnt
            ill_rate: float | None = (
                sum(1 for r in retained if r.response_valid and not r.action_is_legal)
                / retained_cnt
            )
        else:
            retained_opt = 0
            acc = None
            ill_rate = None

        hybrid_oracle_acc = (retained_opt + abstained_cnt) / n

        fb_acc: float | None = None
        fb_cost_100: float | None = None
        if fallback_map is not None:
            hybrid_hits = 0
            hybrid_cost = 0.0
            valid_hybrid = 0
            for row, c in zip(decisions, confs, strict=True):
                if c >= tau:
                    hybrid_hits += 1 if row.action_is_optimal else 0
                    hybrid_cost += float(row.api_cost_usd)
                    valid_hybrid += 1
                elif row.episode_id in fallback_map:
                    fb = fallback_map[row.episode_id]
                    hybrid_hits += 1 if fb.action_is_optimal else 0
                    hybrid_cost += float(row.api_cost_usd) + float(fb.api_cost_usd)
                    valid_hybrid += 1
            if valid_hybrid > 0:
                fb_acc = hybrid_hits / valid_hybrid
                fb_cost_100 = (hybrid_cost / valid_hybrid) * 100.0

        abstention_points.append(
            AbstentionPoint(
                tau=tau,
                coverage=coverage,
                retained_count=retained_cnt,
                abstained_count=abstained_cnt,
                accuracy=acc,
                illegal_rate=ill_rate,
                hybrid_oracle_accuracy=hybrid_oracle_acc,
                hybrid_fallback_accuracy=fb_acc,
                hybrid_fallback_cost_per_100_usd=fb_cost_100,
            )
        )

    return DMCalibrationResult(
        n=n,
        ece=ece,
        brier_score=brier,
        top1_prob_ece=top1_ece,
        top1_prob_brier_score=top1_brier,
        mean_confidence=float(np.mean(confs)),
        mean_top1_prob=float(np.mean(top1_probs)),
        empirical_accuracy=float(np.mean(labels)),
        mean_optimal_mass=float(np.mean(optimal_masses)) if optimal_masses else None,
        mean_illegal_mass=float(np.mean(illegal_masses)) if illegal_masses else None,
        reliability_bins=rel_bins,
        top1_prob_bins=top1_bins,
        abstention_curve=tuple(abstention_points),
    )


def select_core_t2_decisions(decisions: Sequence[DecisionLogRow]) -> list[DecisionLogRow]:
    """Select baseline Track 1 (T1+T2) decisions in clean_default + json_canonical + all_9."""
    primary = [
        d
        for d in decisions
        if d.experiment_id in {"T2", "T1_T2", "T1+T2"}
        and d.action_mode == "all_9"
        and d.render_variant in (None, "clean_default")
        and d.text_format in (None, "json_canonical")
    ]
    if primary:
        return primary

    t1_only = [
        d
        for d in decisions
        if d.experiment_id == "T1"
        and d.action_mode == "all_9"
        and d.render_variant in (None, "clean_default")
        and d.text_format in (None, "json_canonical")
    ]
    if t1_only:
        return t1_only

    return [
        d
        for d in decisions
        if d.experiment_id not in {"M0_smoke", "T3", "T4", "T5", "T6", "H2H"}
        and d.action_mode == "all_9"
        and d.render_variant in (None, "clean_default")
        and d.text_format in (None, "json_canonical")
    ]


def select_t3_decisions(decisions: Sequence[DecisionLogRow]) -> list[DecisionLogRow]:
    """Select T3 rendering-robustness decisions, reusing clean_default baseline from T2 when absent."""
    t3_rows = [
        d
        for d in decisions
        if d.experiment_id == "T3"
        or (
            d.experiment_id not in {"M0_smoke", "T6", "H2H"}
            and d.render_variant in {"labeled_margins", "dark_low_contrast", "thin_alt_font", "tilt_blur"}
        )
    ]
    if not t3_rows:
        return []

    t3_row_ids = {d.row_id for d in t3_rows}
    t3_eps = {d.episode_id for d in t3_rows}
    has_clean_in_t3 = {
        d.row_id for d in t3_rows if d.render_variant == "clean_default"
    }
    missing_clean_rows = t3_row_ids - has_clean_in_t3
    if not missing_clean_rows:
        return t3_rows

    core_t2 = select_core_t2_decisions(decisions)
    combined = list(t3_rows)
    for row_id in sort_row_ids(list(missing_clean_rows)):
        candidates = [
            d
            for d in core_t2
            if d.row_id == row_id and d.render_variant in (None, "clean_default")
        ]
        paired = [d for d in candidates if d.episode_id in t3_eps]
        combined.extend(paired if paired else candidates)
    return combined


def select_t4_decisions(
    decisions: Sequence[DecisionLogRow],
) -> tuple[list[DecisionLogRow], list[DecisionLogRow]]:
    """Return `(all_9_baseline, legal_only_t4)` paired lists for T4 schema comparison."""
    decisions = filter_evaluated_decisions(decisions)
    legal_only = [
        d
        for d in decisions
        if (d.experiment_id == "T4" and d.action_mode == "legal_only")
        or (d.experiment_id not in {"M0_smoke", "T6", "H2H"} and d.action_mode == "legal_only")
    ]
    if not legal_only:
        return [], []

    lo_map: dict[tuple[str, str], DecisionLogRow] = {}
    for d in legal_only:
        key = (d.row_id, d.episode_id)
        if key not in lo_map or d.experiment_id == "T4":
            lo_map[key] = d
    legal_only_dedup = list(lo_map.values())

    t4_all9 = [
        d for d in decisions if d.experiment_id == "T4" and d.action_mode == "all_9"
    ]
    core_t2 = select_core_t2_decisions(decisions)
    baseline_pool = t4_all9 if t4_all9 else core_t2

    t4_row_ids = {d.row_id for d in legal_only_dedup}
    t4_eps = {d.episode_id for d in legal_only_dedup}
    all9_rows = [
        d
        for d in baseline_pool
        if d.row_id in t4_row_ids and (not t4_eps or d.episode_id in t4_eps)
    ]
    if not all9_rows:
        all9_rows = [d for d in baseline_pool if d.row_id in t4_row_ids]
    return all9_rows, legal_only_dedup


def select_t5_decisions(decisions: Sequence[DecisionLogRow]) -> list[DecisionLogRow]:
    """Select T5 prompt-format decisions, reusing json_canonical baseline from T2 when absent."""
    t5_rows = [
        d
        for d in decisions
        if d.experiment_id == "T5"
        or (
            d.experiment_id not in {"M0_smoke", "T6", "H2H"}
            and d.text_format in {"ascii_grid", "coord_list", "json_with_history"}
        )
    ]
    if not t5_rows:
        return []

    t5_row_ids = {d.row_id for d in t5_rows}
    t5_eps = {d.episode_id for d in t5_rows}
    has_canon_in_t5 = {
        d.row_id for d in t5_rows if d.text_format == "json_canonical"
    }
    missing_canon_rows = t5_row_ids - has_canon_in_t5
    if not missing_canon_rows:
        return t5_rows

    core_t2 = select_core_t2_decisions(decisions)
    combined = list(t5_rows)
    for row_id in sort_row_ids(list(missing_canon_rows)):
        candidates = [
            d
            for d in core_t2
            if d.row_id == row_id and d.text_format in (None, "json_canonical")
        ]
        paired = [d for d in candidates if d.episode_id in t5_eps]
        combined.extend(paired if paired else candidates)
    return combined


def select_dm_decisions(decisions: Sequence[DecisionLogRow]) -> list[DecisionLogRow]:
    """Select Decision Model decisions from Track 1 core (all_9) for calibration analysis."""
    decisions = filter_evaluated_decisions(decisions)
    core = select_core_t2_decisions(decisions)
    pool = core if core else [
        d for d in decisions if d.experiment_id not in {"M0_smoke", "T6", "H2H"} and d.action_mode == "all_9"
    ]
    return [
        d
        for d in pool
        if d.family == "decision_model" or d.action_probabilities is not None
    ]
