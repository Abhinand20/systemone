"""Markdown summary report, paired hypothesis comparisons, and Milestone 1 memo generator."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dm_bench.analysis.plots import PlotArtifact, generate_all_plots
from dm_bench.analysis.stats import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    RENDER_VARIANT_ORDER,
    TACTICAL_CATEGORY_ORDER,
    TEXT_FORMAT_ORDER,
    DMCalibrationResult,
    H2HMatchupMetrics,
    H2HStandingMetrics,
    PairedComparisonResult,
    Track1Metrics,
    Track2Metrics,
    compare_paired_decisions,
    compute_dm_calibration,
    compute_h2h_metrics,
    compute_h2h_standings,
    compute_tactical_metrics,
    compute_track1_metrics,
    compute_track2_metrics,
    is_infra_decision,
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
from dm_bench.logging_io import utc_now_iso


def _fmt_pct(val: float | None, decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100.0:.{decimals}f}%"


def _fmt_float(val: float | None, decimals: int = 3) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _fmt_ms(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.1f}"


def _fmt_usd(val: float | None, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    return f"${val:.{decimals}f}"


def _fmt_per_usd(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.1f}"


def _fmt_board_inline(board: Sequence[str]) -> str:
    if len(board) != 9:
        return "".join(board)
    return "/".join("".join(board[i : i + 3]) for i in (0, 3, 6))


def _compute_hypothesis_comparisons(
    core_decisions: Sequence[DecisionLogRow],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, list[PairedComparisonResult]]:
    by_row: dict[str, list[DecisionLogRow]] = defaultdict(list)
    for row in core_decisions:
        by_row[row.row_id].append(row)

    specs: dict[str, list[tuple[str, str, str]]] = {
        "H1_structured_vs_vision": [
            ("Jev Text vs Vision (B_jev_T - A_jev_V)", "B_jev_T", "A_jev_V"),
            ("Gemini Decision 1 Text vs Vision (B_gdec_T - A_gdec_V)", "B_gdec_T", "A_gdec_V"),
            ("Gemini 4 Hydrogen Text vs Vision (F_ghyd_T - C_ghyd_V)", "F_ghyd_T", "C_ghyd_V"),
        ],
        "H2_hybrid_vs_single": [
            ("Jev Hybrid vs Vision (D_jev_VT - A_jev_V)", "D_jev_VT", "A_jev_V"),
            ("Jev Hybrid vs Text (D_jev_VT - B_jev_T)", "D_jev_VT", "B_jev_T"),
            ("Gemini Decision 1 Hybrid vs Vision (D_gdec_VT - A_gdec_V)", "D_gdec_VT", "A_gdec_V"),
            ("Gemini Decision 1 Hybrid vs Text (D_gdec_VT - B_gdec_T)", "D_gdec_VT", "B_gdec_T"),
            ("Gemini 4 Hydrogen Hybrid vs Vision (E_ghyd_VT - C_ghyd_V)", "E_ghyd_VT", "C_ghyd_V"),
            ("Gemini 4 Hydrogen Hybrid vs Text (E_ghyd_VT - F_ghyd_T)", "E_ghyd_VT", "F_ghyd_T"),
        ],
        "H3_dm_vs_generative": [
            ("Vision: Jev vs Hydrogen (A_jev_V - C_ghyd_V)", "A_jev_V", "C_ghyd_V"),
            ("Text: Jev vs Hydrogen (B_jev_T - F_ghyd_T)", "B_jev_T", "F_ghyd_T"),
            ("Hybrid: Jev vs Hydrogen (D_jev_VT - E_ghyd_VT)", "D_jev_VT", "E_ghyd_VT"),
            ("Vision: Gemini Decision 1 vs Hydrogen (A_gdec_V - C_ghyd_V)", "A_gdec_V", "C_ghyd_V"),
            ("Text: Gemini Decision 1 vs Hydrogen (B_gdec_T - F_ghyd_T)", "B_gdec_T", "F_ghyd_T"),
            ("Hybrid: Gemini Decision 1 vs Hydrogen (D_gdec_VT - E_ghyd_VT)", "D_gdec_VT", "E_ghyd_VT"),
        ],
        "H4_dm_head_to_head": [
            ("Vision: Jev vs Gemini Decision 1 (A_jev_V - A_gdec_V)", "A_jev_V", "A_gdec_V"),
            ("Text: Jev vs Gemini Decision 1 (B_jev_T - B_gdec_T)", "B_jev_T", "B_gdec_T"),
            ("Hybrid: Jev vs Gemini Decision 1 (D_jev_VT - D_gdec_VT)", "D_jev_VT", "D_gdec_VT"),
        ],
    }

    results: dict[str, list[PairedComparisonResult]] = {}
    for group_name, pairs in specs.items():
        group_results: list[PairedComparisonResult] = []
        for label, rx, ry in pairs:
            if rx in by_row and ry in by_row:
                comp = compare_paired_decisions(
                    by_row[rx],
                    by_row[ry],
                    label=label,
                    row_x_label=rx,
                    row_y_label=ry,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                if comp is not None:
                    group_results.append(comp)
        results[group_name] = group_results
    return results


def _select_representative_failures(
    decisions: Sequence[DecisionLogRow],
    episodes: Sequence[EpisodeLogRow],
    *,
    limit_per_kind: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    track1 = [d for d in decisions if d.experiment_id not in {"M0_smoke", "T6", "H2H"}]

    invalid_rows = [
        d for d in track1 if not d.response_valid and not is_infra_decision(d)
    ][:limit_per_kind]
    illegal_rows = [
        d for d in track1 if d.response_valid and not d.action_is_legal
    ][:limit_per_kind]
    suboptimal_rows = sorted(
        [
            d
            for d in track1
            if d.response_valid and d.action_is_legal and not d.action_is_optimal
        ],
        key=lambda d: (-(float(d.effective_regret or 0.0)), -(d.confidence or 0.0), d.episode_id),
    )[:limit_per_kind]

    t6_failures = [
        e
        for e in episodes
        if e.outcome in ("loss_by_illegal_move", "loss_by_board")
    ]
    t6_failures_sorted = sorted(
        t6_failures,
        key=lambda e: (0 if e.outcome == "loss_by_illegal_move" else 1, e.row_id, e.episode_id),
    )[:limit_per_kind]

    def _serialize_decision(d: DecisionLogRow) -> dict[str, Any]:
        return {
            "experiment_id": d.experiment_id,
            "row_id": d.row_id,
            "episode_id": d.episode_id,
            "render_variant": d.render_variant,
            "text_format": d.text_format,
            "action_mode": d.action_mode,
            "tactical_category": d.tactical_category,
            "player_to_move": d.player_to_move,
            "board": _fmt_board_inline(d.board),
            "requested_action": d.requested_action,
            "optimal_actions": d.optimal_actions,
            "effective_regret": d.effective_regret,
            "confidence": d.confidence,
            "error_type": d.error_type,
        }

    def _serialize_episode(e: EpisodeLogRow) -> dict[str, Any]:
        return {
            "row_id": e.row_id,
            "episode_id": e.episode_id,
            "opponent_type": e.opponent_type,
            "action_mode": e.action_mode,
            "outcome": e.outcome,
            "termination_reason": e.termination_reason,
            "num_plies": e.num_plies,
            "illegal_moves": e.illegal_moves,
            "mean_effective_regret": e.mean_effective_regret,
        }

    return {
        "invalid_responses": [_serialize_decision(d) for d in invalid_rows],
        "illegal_moves": [_serialize_decision(d) for d in illegal_rows],
        "suboptimal_legal_moves": [_serialize_decision(d) for d in suboptimal_rows],
        "t6_match_losses": [_serialize_episode(e) for e in t6_failures_sorted],
    }


def _assess_hypothesis_status(comps: Sequence[PairedComparisonResult]) -> str:
    if not comps:
        return "Insufficient paired data in loaded runs."
    pos_sig = sum(1 for c in comps if c.optimal_ci.ci_low > 0)
    neg_sig = sum(1 for c in comps if c.optimal_ci.ci_high < 0)
    if pos_sig == len(comps):
        return f"Supported ({pos_sig}/{len(comps)} pairs show statistically positive delta with 95% CI > 0)."
    if neg_sig == len(comps):
        return f"Not supported ({neg_sig}/{len(comps)} pairs show statistically negative delta)."
    if pos_sig > 0 and neg_sig == 0:
        return f"Partially supported ({pos_sig}/{len(comps)} pairs significantly positive; remaining pairs inconclusive)."
    if pos_sig > 0 or neg_sig > 0:
        return f"Mixed evidence ({pos_sig} positive, {neg_sig} negative out of {len(comps)} pairs)."
    return f"Inconclusive / parity (0/{len(comps)} pairs reach 95% CI separation)."


def _render_markdown_report(
    *,
    tag: str,
    run_dirs: Sequence[Path],
    experiment_ids: Sequence[str],
    decisions_count: int,
    episodes_count: int,
    h2h_matches_count: int,
    total_spend_usd: float,
    mock_mode: bool,
    core_metrics: Sequence[Track1Metrics],
    tactical_metrics: dict[tuple[str, str], Track1Metrics],
    t3_metrics: Sequence[Track1Metrics],
    t4_metrics: list[dict[str, Any]],
    t5_metrics: Sequence[Track1Metrics],
    track2_metrics: Sequence[Track2Metrics],
    h2h_matchups: Sequence[H2HMatchupMetrics],
    h2h_standings: Sequence[H2HStandingMetrics],
    comparisons: dict[str, list[PairedComparisonResult]],
    dm_cal_by_model: dict[str, DMCalibrationResult],
    dm_cal_by_row: dict[str, DMCalibrationResult],
    failures: dict[str, list[dict[str, Any]]],
    plots: Sequence[PlotArtifact],
) -> str:
    lines: list[str] = [
        f"# dm-bench Milestone 1 Analysis Summary (`{tag}`)",
        "",
        f"- **Generated (UTC):** `{utc_now_iso()}`",
        f"- **Experiments present:** {', '.join(experiment_ids) if experiment_ids else 'None'}",
        f"- **Total decisions loaded:** `{decisions_count}`",
        f"- **Total T6 episodes loaded:** `{episodes_count}`",
        f"- **Total H2H matches loaded:** `{h2h_matches_count}`",
        f"- **Observed API spend:** `${total_spend_usd:.6f}`" + (" *(mock mode)*" if mock_mode else ""),
        f"- **Run directories:** {', '.join(f'`{p.as_posix()}`' for p in run_dirs)}",
        "",
    ]

    # 1. T2 Core Matrix
    lines.extend(["## 1. T2 Core Matrix (Clean Default · JSON Canonical · all_9)", ""])
    if core_metrics:
        lines.extend(
            [
                "| Row | Cond | Model | n | Opt Acc | Cond Opt Acc | Illegal | Invalid | Eff Regret | Legal Regret | p50 (ms) | p95 (ms) | Mean (ms) | $/100 Dec | Opt/USD |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for m in core_metrics:
            lines.append(
                f"| `{m.row_id}` | `{m.condition_code}` | `{m.model_key}` | {m.n} | "
                f"{_fmt_pct(m.optimal_accuracy)} | {_fmt_pct(m.conditional_optimal_accuracy)} | "
                f"{_fmt_pct(m.illegal_rate)} | {_fmt_pct(m.invalid_response_rate)} | "
                f"{_fmt_float(m.mean_effective_regret)} | {_fmt_float(m.mean_legal_regret)} | "
                f"{_fmt_ms(m.p50_latency_ms)} | {_fmt_ms(m.p95_latency_ms)} | {_fmt_ms(m.mean_latency_ms)} | "
                f"{_fmt_usd(m.cost_per_100_decisions_usd)} | {_fmt_per_usd(m.optimal_decisions_per_usd)} |"
            )
        infra_notes = [
            f"`{m.row_id}`: {m.infra_error_count} excluded ({m.rate_limit_error_count} rate-limit)"
            for m in core_metrics
            if m.infra_error_count > 0
        ]
        if infra_notes:
            lines.append("")
            lines.append(
                "> Excluded infrastructure/API decision errors (not counted in denominators): "
                + "; ".join(infra_notes)
            )
    else:
        lines.append("*No T2 core matrix decisions found in loaded runs.*")
    lines.append("")

    # 2. T1 Tactical Breakdown
    lines.extend(["## 2. T1 Tactical Category Breakdown (Optimal Accuracy)", ""])
    if tactical_metrics:
        t1_rows = sort_row_ids(list({k[0] for k in tactical_metrics}))
        cats = [c for c in TACTICAL_CATEGORY_ORDER if any((r, c) in tactical_metrics for r in t1_rows)]
        lines.append("| Row | " + " | ".join(cats) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(cats)) + "|")
        for rid in t1_rows:
            cells = []
            for cat in cats:
                cell = tactical_metrics.get((rid, cat))
                cells.append(f"{_fmt_pct(cell.optimal_accuracy)} ({cell.n})" if cell else "N/A")
            lines.append(f"| `{rid}` | " + " | ".join(cells) + " |")
    else:
        lines.append("*No tactical category annotations found.*")
    lines.append("")

    # 3. T3 Rendering Robustness
    lines.extend(["## 3. T3 Rendering Robustness", ""])
    if t3_metrics:
        lines.extend(
            [
                "| Row | Render Variant | n | Opt Acc | Cond Opt Acc | Illegal | Eff Regret | p50 (ms) |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for m in t3_metrics:
            lines.append(
                f"| `{m.row_id}` | `{m.render_variant or 'clean_default'}` | {m.n} | "
                f"{_fmt_pct(m.optimal_accuracy)} | {_fmt_pct(m.conditional_optimal_accuracy)} | "
                f"{_fmt_pct(m.illegal_rate)} | {_fmt_float(m.mean_effective_regret)} | {_fmt_ms(m.p50_latency_ms)} |"
            )
    else:
        lines.append("*T3 rendering perturbation data not present in loaded runs.*")
    lines.append("")

    # 4. T4 Action-Choice Schema
    lines.extend(["## 4. T4 Action-Choice Schema (`all_9` vs `legal_only`)", ""])
    if t4_metrics:
        lines.extend(
            [
                "| Row | Shared n | `all_9` Acc | `legal_only` Acc | Delta (legal - all_9) | 95% Bootstrap CI | McNemar p |",
                "|---|---:|---:|---:|---:|---|---:|",
            ]
        )
        for item in t4_metrics:
            ci = item["paired"]["optimal_ci"]
            lines.append(
                f"| `{item['row_id']}` | {item['paired']['shared_n']} | "
                f"{_fmt_pct(ci['mean_y'])} | {_fmt_pct(ci['mean_x'])} | "
                f"{ci['delta'] * 100.0:+.1f}pp | [{ci['ci_low'] * 100.0:+.1f}pp, {ci['ci_high'] * 100.0:+.1f}pp] | "
                f"{item['paired']['mcnemar_pvalue']:.4g} |"
            )
    else:
        lines.append("*T4 `legal_only` ablation data not present in loaded runs.*")
    lines.append("")

    # 5. T5 Prompt Representation
    lines.extend(["## 5. T5 Prompt Format Sensitivity", ""])
    if t5_metrics:
        lines.extend(
            [
                "| Row | Text Format | n | Opt Acc | Cond Opt Acc | Illegal | Eff Regret | p50 (ms) |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for m in t5_metrics:
            lines.append(
                f"| `{m.row_id}` | `{m.text_format or 'json_canonical'}` | {m.n} | "
                f"{_fmt_pct(m.optimal_accuracy)} | {_fmt_pct(m.conditional_optimal_accuracy)} | "
                f"{_fmt_pct(m.illegal_rate)} | {_fmt_float(m.mean_effective_regret)} | {_fmt_ms(m.p50_latency_ms)} |"
            )
    else:
        lines.append("*T5 prompt representation data not present in loaded runs.*")
    lines.append("")

    # 6. T6 Closed-Loop Matches
    lines.extend(["## 6. T6 Closed-Loop Gameplay Outcomes", ""])
    if track2_metrics:
        lines.extend(
            [
                "| Row | Mode | Opponent | Games | Win | Draw | Non-Loss | Loss (Board) | Loss (Illegal) | $/Episode | Non-Loss/USD |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for m in track2_metrics:
            lines.append(
                f"| `{m.row_id}` | `{m.action_mode}` | `{m.opponent_type}` | {m.games} | "
                f"{_fmt_pct(m.win_rate)} | {_fmt_pct(m.draw_rate)} | {_fmt_pct(m.non_loss_rate)} | "
                f"{_fmt_pct(m.loss_by_board_rate)} | {_fmt_pct(m.loss_by_illegal_move_rate)} | "
                f"{_fmt_usd(m.cost_per_episode_usd)} | {_fmt_per_usd(m.non_loss_games_per_usd)} |"
            )
        aborted_notes = [
            f"`{m.row_id}` (`{m.action_mode}` vs `{m.opponent_type}`): {m.aborted_errors} aborted ({m.rate_limit_errors} rate-limit)"
            for m in track2_metrics
            if m.aborted_errors > 0
        ]
        if aborted_notes:
            lines.append("")
            lines.append(
                "> Excluded aborted episodes due to API/rate-limit errors (not counted in Games or forfeits): "
                + "; ".join(aborted_notes)
            )
    else:
        lines.append("*T6 closed-loop match episodes not present in loaded runs.*")
    lines.append("")

    # 6b. Head-to-Head Model-vs-Model Arena (H2H)
    lines.extend(["## 6b. Head-to-Head Model-vs-Model Arena (H2H)", ""])
    if h2h_matchups:
        lines.extend(
            [
                "### Matchup Scoreboard",
                "",
                "| Matchup (A vs B) | Mode | Games | W-D-L (A) | Score A (95% CI) | p (decisive) | Board Wins A/B | Forfeit Wins A/B | Opt% A / B | Total USD |",
                "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for m in h2h_matchups:
            lines.append(
                f"| `{m.row_a}` vs `{m.row_b}` | `{m.action_mode}` | {m.games} | "
                f"{m.wins_a}-{m.draws}-{m.wins_b} | "
                f"{m.score_a:.3f} [{m.score_a_ci_low:.2f}, {m.score_a_ci_high:.2f}] | "
                f"{m.p_value_decisive:.4g} | {m.board_wins_a}/{m.board_wins_b} | "
                f"{m.forfeit_wins_a}/{m.forfeit_wins_b} | "
                f"{_fmt_pct(m.optimal_rate_a)} / {_fmt_pct(m.optimal_rate_b)} | {_fmt_usd(m.total_cost_usd)} |"
            )
        lines.append("")
        if h2h_standings:
            lines.extend(
                [
                    "### Overall H2H Arena Standings",
                    "",
                    "| Row | Model | Modality | Games | W-D-L | Points | Score% | Opt% | Eff Regret | Forfeits Lost | USD |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for s in h2h_standings:
                lines.append(
                    f"| `{s.row_id}` | `{s.model_key}` | `{s.modality}` | {s.games} | "
                    f"{s.wins}-{s.draws}-{s.losses} | {s.points:.1f} | {_fmt_pct(s.score_rate)} | "
                    f"{_fmt_pct(s.optimal_rate)} | {_fmt_float(s.mean_effective_regret)} | "
                    f"{s.losses_by_illegal_move} | {_fmt_usd(s.total_cost_usd)} |"
                )
    else:
        lines.append("*H2H head-to-head match logs not present in loaded runs.*")
    lines.append("")

    # 7. Key Paired Hypothesis Comparisons
    lines.extend(
        [
            "## 7. Key Paired Comparisons (95% Bootstrap CI, B=10,000, Exact McNemar)",
            "",
        ]
    )
    titles = {
        "H1_structured_vs_vision": "### H1 — Structured State vs Vision-Only",
        "H2_hybrid_vs_single": "### H2 — Hybrid (V+T) vs Single-Modality (V or T)",
        "H3_dm_vs_generative": "### H3 — Decision Models vs Generative VLM/LLM (Quality, Latency & Cost)",
        "H4_dm_head_to_head": "### H4 — Decision Model vs Decision Model (Jev vs Gemini Decision 1)",
    }
    for key, heading in titles.items():
        lines.extend([heading, ""])
        comps = comparisons.get(key, [])
        if comps:
            lines.extend(
                [
                    "| Comparison | n | Acc X | Acc Y | Delta Acc | 95% CI | McNemar p | Delta Latency (ms) | $/100 Delta |",
                    "|---|---:|---:|---:|---:|---|---:|---:|---:|",
                ]
            )
            for c in comps:
                lat_str = (
                    f"{c.latency_ci.delta:+.1f} [{c.latency_ci.ci_low:+.1f}, {c.latency_ci.ci_high:+.1f}]"
                    if c.latency_ci
                    else "N/A"
                )
                cost_100_delta = c.cost_ci.delta * 100.0
                lines.append(
                    f"| {c.label} | {c.shared_n} | {_fmt_pct(c.optimal_ci.mean_x)} | "
                    f"{_fmt_pct(c.optimal_ci.mean_y)} | {c.optimal_ci.delta * 100.0:+.1f}pp | "
                    f"[{c.optimal_ci.ci_low * 100.0:+.1f}pp, {c.optimal_ci.ci_high * 100.0:+.1f}pp] | "
                    f"{c.mcnemar_pvalue:.4g} | {lat_str} | {cost_100_delta:+.4f} |"
                )
        else:
            lines.append("*No shared paired positions available for this comparison group.*")
        lines.append("")

    # 8. Decision-Model Calibration & Abstention
    lines.extend(["## 8. Decision-Model Multi-Optimal Calibration & Selective Abstention", ""])
    if dm_cal_by_model:
        lines.extend(
            [
                "### Per-Model Calibration Summary",
                "",
                "| Model | n | Opt Acc | Mean Conf | ECE | Brier | Top-1 Prob ECE | Mean Optimal Mass | Mean Illegal Mass |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for mkey, cal in dm_cal_by_model.items():
            lines.append(
                f"| `{mkey}` | {cal.n} | {_fmt_pct(cal.empirical_accuracy)} | "
                f"{_fmt_float(cal.mean_confidence)} | {_fmt_float(cal.ece, 4)} | "
                f"{_fmt_float(cal.brier_score, 4)} | {_fmt_float(cal.top1_prob_ece, 4)} | "
                f"{_fmt_float(cal.mean_optimal_mass)} | {_fmt_float(cal.mean_illegal_mass)} |"
            )
        lines.append("")
        if dm_cal_by_row:
            lines.extend(
                [
                    "### Per-Condition Calibration & Abstention Highlights",
                    "",
                    "| Row | n | ECE | Brier | Optimal Mass | Illegal Mass | Cov @ τ=0.70 | Acc @ τ=0.70 |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for rid, cal in dm_cal_by_row.items():
                pt_70 = min(cal.abstention_curve, key=lambda p: abs(p.tau - 0.70))
                lines.append(
                    f"| `{rid}` | {cal.n} | {_fmt_float(cal.ece, 4)} | {_fmt_float(cal.brier_score, 4)} | "
                    f"{_fmt_float(cal.mean_optimal_mass)} | {_fmt_float(cal.mean_illegal_mass)} | "
                    f"{_fmt_pct(pt_70.coverage)} | {_fmt_pct(pt_70.accuracy)} |"
                )
            lines.append("")
    else:
        lines.append("*No Decision Model probability/confidence logs available.*")
        lines.append("")

    # 9. Representative Failure Examples
    lines.extend(["## 9. Representative Failure Examples", ""])
    has_any_failure = any(failures.values())
    if has_any_failure:
        for category, items in failures.items():
            if not items:
                continue
            lines.append(f"### {category.replace('_', ' ').title()}")
            lines.append("")
            for item in items:
                if category == "t6_match_losses":
                    lines.append(
                        f"- `{item['row_id']}` vs `{item['opponent_type']}` (`{item['action_mode']}`, `{item['episode_id']}`): "
                        f"outcome=`{item['outcome']}`, plies={item['num_plies']}, illegal_moves={item['illegal_moves']}"
                    )
                else:
                    lines.append(
                        f"- `{item['row_id']}` (`{item['experiment_id']}`, `{item['episode_id']}`, `{item['tactical_category']}`): "
                        f"to_move=`{item['player_to_move']}`, board=`{item['board']}`, "
                        f"chosen=`{item['requested_action']}`, optimal=`{item['optimal_actions']}`, "
                        f"regret={item['effective_regret']}, conf={item['confidence']}, err=`{item['error_type']}`"
                    )
            lines.append("")
    else:
        lines.append("*Zero failures observed across loaded runs.*")
        lines.append("")

    # 10. Plots Index
    lines.extend(["## 10. Generated Visualization Artifacts", ""])
    for plot in plots:
        if plot.generated and plot.png_path is not None:
            lines.append(f"- `{plot.name}`: `{plot.png_path.as_posix()}` (+ `.svg`)")
        else:
            lines.append(f"- `{plot.name}`: *Skipped ({plot.reason})*")
    lines.append("")

    # 11. Milestone 1 Interim Memo Scaffold
    h1_status = _assess_hypothesis_status(comparisons.get("H1_structured_vs_vision", []))
    h2_status = _assess_hypothesis_status(comparisons.get("H2_hybrid_vs_single", []))
    h3_comps = comparisons.get("H3_dm_vs_generative", [])
    h3_quality = _assess_hypothesis_status(h3_comps)
    best_row = max(core_metrics, key=lambda m: (m.optimal_accuracy, -m.illegal_rate), default=None)
    cheapest_row = min(core_metrics, key=lambda m: (m.cost_per_100_decisions_usd, -m.optimal_accuracy), default=None)

    lines.extend(
        [
            "## 11. Milestone 1 Interim Memo Scaffold",
            "",
            "### Executive Summary",
            f"- **Top Track 1 Policy:** `{best_row.row_id}` (`{_fmt_pct(best_row.optimal_accuracy)}` optimal, `{_fmt_pct(best_row.illegal_rate)}` illegal)" if best_row else "- **Top Track 1 Policy:** *Pending T2 run.*",
            f"- **Lowest-Cost Policy:** `{cheapest_row.row_id}` (`{_fmt_usd(cheapest_row.cost_per_100_decisions_usd)}` per 100 decisions)" if cheapest_row else "- **Lowest-Cost Policy:** *Pending T2 run.*",
            "- **T6 Closed-Loop Readiness:** "
            + (
                f"Evaluated `{len(track2_metrics)}` `(row, mode, opponent)` slices across `{episodes_count}` matches."
                if track2_metrics
                else "Pending T6 match evaluation."
            ),
            "",
            "### Hypothesis Assessment",
            f"- **H1 (Structured State Advantage):** {h1_status}",
            f"- **H2 (Hybrid Input Advantage with Caveats):** {h2_status}",
            f"- **H3 (DM vs Generative Quality/Latency/Cost Trade-off):** Quality parity/difference: {h3_quality}",
            "",
            "### Calibration & Selective Control Takeaway",
            *(
                [
                    f"- `{mkey}`: ECE=`{cal.ece:.4f}`, Brier=`{cal.brier_score:.4f}`, "
                    f"mean optimal-set mass=`{_fmt_float(cal.mean_optimal_mass)}`, "
                    f"mean illegal mass=`{_fmt_float(cal.mean_illegal_mass)}`."
                    for mkey, cal in dm_cal_by_model.items()
                ]
                if dm_cal_by_model
                else ["- *Pending Decision Model calibration evaluation.*"]
            ),
            "",
            "### Next Steps Toward Milestone 2 (Breakout)",
            "- Inspect top failure modes in the Trace Viewer (`bench view`) to separate visual perception errors from minimax tactical errors.",
            "- Carry forward validated representation lessons (`all_9` vs `legal_only`, structured vs pixel) into `ALE/Breakout-v5` frame-history experiments.",
            "",
        ]
    )

    return "\n".join(lines)


def analyze_runs(
    run_dirs: Sequence[Path | str],
    repo_root: Path | None = None,
    *,
    tag: str = "latest",
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Analyze one or more experiment run directories, generating metrics, plots, and Markdown report."""
    root = Path(repo_root) if repo_root is not None else find_repo_root()
    loaded = load_runs_data(run_dirs)
    decisions = loaded.decisions
    episodes = loaded.episodes
    h2h_matches = loaded.h2h_matches

    all_t1_metrics = compute_track1_metrics(decisions)
    core_decisions = select_core_t2_decisions(decisions)
    core_metrics = list(compute_track1_metrics(core_decisions).values())
    tactical_metrics = compute_tactical_metrics(core_decisions)

    t3_decisions = select_t3_decisions(decisions)
    t3_grouped: dict[tuple[str, str], list[DecisionLogRow]] = defaultdict(list)
    for d in t3_decisions:
        t3_grouped[(d.row_id, d.render_variant or "clean_default")].append(d)
    t3_metrics: list[Track1Metrics] = []
    for rid in sort_row_ids(list({k[0] for k in t3_grouped})):
        variants = [v for v in RENDER_VARIANT_ORDER if (rid, v) in t3_grouped]
        variants.extend(sorted({k[1] for k in t3_grouped if k[0] == rid} - set(variants)))
        for var in variants:
            vals = list(compute_track1_metrics(t3_grouped[(rid, var)]).values())
            if vals:
                t3_metrics.append(vals[0])

    all9_t4, legal_only_t4 = select_t4_decisions(decisions)
    t4_comparisons: list[dict[str, Any]] = []
    if all9_t4 and legal_only_t4:
        a9_by_row: dict[str, list[DecisionLogRow]] = defaultdict(list)
        lo_by_row: dict[str, list[DecisionLogRow]] = defaultdict(list)
        for d in all9_t4:
            a9_by_row[d.row_id].append(d)
        for d in legal_only_t4:
            lo_by_row[d.row_id].append(d)
        for rid in sort_row_ids(list(set(a9_by_row) & set(lo_by_row))):
            paired = compare_paired_decisions(
                lo_by_row[rid],
                a9_by_row[rid],
                label=f"{rid}: legal_only vs all_9",
                row_x_label=f"{rid} (legal_only)",
                row_y_label=f"{rid} (all_9)",
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            if paired is not None:
                t4_comparisons.append({"row_id": rid, "paired": paired.to_dict()})

    t5_decisions = select_t5_decisions(decisions)
    t5_grouped: dict[tuple[str, str], list[DecisionLogRow]] = defaultdict(list)
    for d in t5_decisions:
        t5_grouped[(d.row_id, d.text_format or "json_canonical")].append(d)
    t5_metrics: list[Track1Metrics] = []
    for rid in sort_row_ids(list({k[0] for k in t5_grouped})):
        fmts = [f for f in TEXT_FORMAT_ORDER if (rid, f) in t5_grouped]
        fmts.extend(sorted({k[1] for k in t5_grouped if k[0] == rid} - set(fmts)))
        for fmt in fmts:
            vals = list(compute_track1_metrics(t5_grouped[(rid, fmt)]).values())
            if vals:
                t5_metrics.append(vals[0])

    track2_metrics = list(compute_track2_metrics(episodes).values())
    comparisons = _compute_hypothesis_comparisons(
        core_decisions,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    dm_decisions = select_dm_decisions(decisions)
    dm_by_model: dict[str, list[DecisionLogRow]] = defaultdict(list)
    dm_by_row: dict[str, list[DecisionLogRow]] = defaultdict(list)
    for d in dm_decisions:
        dm_by_model[d.model_key].append(d)
        dm_by_row[d.row_id].append(d)

    dm_cal_by_model = {
        mkey: compute_dm_calibration(dm_by_model[mkey])
        for mkey in sorted(dm_by_model.keys())
    }
    dm_cal_by_row = {
        rid: compute_dm_calibration(dm_by_row[rid])
        for rid in sort_row_ids(list(dm_by_row.keys()))
    }

    failures = _select_representative_failures(decisions, episodes)
    h2h_matchups = list(
        compute_h2h_metrics(h2h_matches, n_bootstrap=n_bootstrap, seed=seed).values()
    )
    h2h_standings = compute_h2h_standings(h2h_matches)

    plot_dir = root / "analysis" / "plots" / tag
    plot_artifacts = generate_all_plots(
        decisions,
        episodes,
        plot_dir,
        h2h_matches=h2h_matches,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    generated_plot_paths: list[Path] = []
    for art in plot_artifacts:
        if art.generated:
            if art.png_path is not None:
                generated_plot_paths.append(art.png_path)
            if art.svg_path is not None:
                generated_plot_paths.append(art.svg_path)

    experiment_ids = sorted(
        {d.experiment_id for d in decisions}
        | {e.experiment_id for e in episodes}
        | {m.experiment_id for m in h2h_matches}
    )
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
    mock_mode = any(m.mock_mode for m in loaded.manifests)

    report_md = _render_markdown_report(
        tag=tag,
        run_dirs=loaded.run_dirs,
        experiment_ids=experiment_ids,
        decisions_count=len(decisions),
        episodes_count=len(episodes),
        h2h_matches_count=len(h2h_matches),
        total_spend_usd=total_spend_usd,
        mock_mode=mock_mode,
        core_metrics=core_metrics,
        tactical_metrics=tactical_metrics,
        t3_metrics=t3_metrics,
        t4_metrics=t4_comparisons,
        t5_metrics=t5_metrics,
        track2_metrics=track2_metrics,
        h2h_matchups=h2h_matchups,
        h2h_standings=h2h_standings,
        comparisons=comparisons,
        dm_cal_by_model=dm_cal_by_model,
        dm_cal_by_row=dm_cal_by_row,
        failures=failures,
        plots=plot_artifacts,
    )

    report_path = root / "analysis" / "reports" / f"{tag}_summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")

    summary: dict[str, Any] = {
        "tag": tag,
        "experiment_ids": experiment_ids,
        "total_decisions": len(decisions),
        "total_episodes": len(episodes),
        "total_h2h_matches": len(h2h_matches),
        "total_spend_usd": total_spend_usd,
        "track1_metrics": [m.to_dict() for m in all_t1_metrics.values()],
        "core_t2_metrics": [m.to_dict() for m in core_metrics],
        "t1_tactical_metrics": {
            f"{rid}:{cat}": m.to_dict() for (rid, cat), m in sorted(tactical_metrics.items())
        },
        "t3_metrics": [m.to_dict() for m in t3_metrics],
        "t4_comparisons": t4_comparisons,
        "t5_metrics": [m.to_dict() for m in t5_metrics],
        "track2_metrics": [m.to_dict() for m in track2_metrics],
        "h2h_matchups": [m.to_dict() for m in h2h_matchups],
        "h2h_standings": [s.to_dict() for s in h2h_standings],
        "paired_comparisons": {
            key: [c.to_dict() for c in comps] for key, comps in comparisons.items()
        },
        "calibration": {
            "by_model": {k: v.to_dict() for k, v in dm_cal_by_model.items()},
            "by_row": {k: v.to_dict() for k, v in dm_cal_by_row.items()},
        },
        "failures": failures,
        "plot_status": [
            {
                "name": art.name,
                "generated": art.generated,
                "png_path": str(art.png_path) if art.png_path else None,
                "svg_path": str(art.svg_path) if art.svg_path else None,
                "reason": art.reason,
            }
            for art in plot_artifacts
        ],
    }

    return {
        "report_path": report_path,
        "plots": generated_plot_paths,
        "summary": summary,
    }
