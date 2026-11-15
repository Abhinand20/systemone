"""Unit and integration tests for dm_bench.analysis (stats, calibration, plots, report)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from dm_bench.analysis import (
    CHART_NAMES,
    CORE_ROW_ORDER,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    RENDER_VARIANT_ORDER,
    TACTICAL_CATEGORY_ORDER,
    TEXT_FORMAT_ORDER,
    analyze_runs,
    compare_paired_decisions,
    compute_dm_calibration,
    compute_track1_metrics,
    compute_track2_metrics,
    mcnemar_exact_from_counts,
    mcnemar_exact_pvalue,
    paired_bootstrap_ci,
)
from dm_bench.contracts import DecisionLogRow, EpisodeLogRow, RunManifest
from dm_bench.logging_io import RunLogger

ROW_META: dict[str, tuple[str, str, str, str]] = {
    "A_jev_V": ("A-DM-V", "jev_1_14_0", "jev-1.14.0", "decision_model"),
    "B_jev_T": ("B-DM-T", "jev_1_14_0", "jev-1.14.0", "decision_model"),
    "D_jev_VT": ("D-DM-VT", "jev_1_14_0", "jev-1.14.0", "decision_model"),
    "A_gdec_V": ("A-DM-V", "gemini_decision_1", "gemini-decision-1", "decision_model"),
    "B_gdec_T": ("B-DM-T", "gemini_decision_1", "gemini-decision-1", "decision_model"),
    "D_gdec_VT": ("D-DM-VT", "gemini_decision_1", "gemini-decision-1", "decision_model"),
    "C_ghyd_V": ("C-VLM-V", "gemini_4_hydrogen", "gemini-4-hydrogen", "generative"),
    "E_ghyd_VT": ("E-VLM-VT", "gemini_4_hydrogen", "gemini-4-hydrogen", "generative"),
    "F_ghyd_T": ("F-LLM-T", "gemini_4_hydrogen", "gemini-4-hydrogen", "generative"),
}


def _make_decision(
    *,
    experiment_id: str = "T2",
    run_id: str = "run_test",
    row_id: str = "A_jev_V",
    episode_id: str = "pos_0001",
    render_variant: str | None = "clean_default",
    text_format: str | None = "json_canonical",
    action_mode: str = "all_9",
    tactical_category: str = "immediate_win",
    response_valid: bool = True,
    action_is_legal: bool = True,
    action_is_optimal: bool = True,
    requested_action: str | None = "R1C1",
    legal_regret: float | None = 0.0,
    effective_regret: float = 0.0,
    end_to_end_latency_ms: float = 120.0,
    model_latency_ms: float = 100.0,
    api_cost_usd: float = 0.0001,
    confidence: float | None = None,
    action_probabilities: dict[str, float] | None = None,
    optimal_mass: float | None = None,
    illegal_mass: float | None = None,
    optimal_actions: list[str] | None = None,
    legal_actions: list[str] | None = None,
    error_type: str | None = None,
) -> DecisionLogRow:
    cond_code, model_key, model_id, family = ROW_META[row_id]
    return DecisionLogRow(
        experiment_id=experiment_id,
        run_id=run_id,
        row_id=row_id,
        condition_code=cond_code,
        model_key=model_key,
        model_id=model_id,
        family=family,
        episode_id=episode_id,
        step_idx=0,
        timestamp_utc="2026-11-15T00:00:00Z",
        render_variant=render_variant,
        text_format=text_format,
        action_mode=action_mode,  # type: ignore[arg-type]
        board=["X", "O", ".", ".", "X", ".", ".", ".", "O"],
        player_to_move="X",
        starting_player="X",
        ply=4,
        legal_actions=legal_actions or ["R1C3", "R2C1", "R2C3", "R3C1", "R3C2"],
        state_value=1,
        optimal_actions=optimal_actions or ["R1C1"],
        action_values={"R1C3": 1, "R2C1": 0, "R2C3": 0, "R3C1": 0, "R3C2": -1},
        tactical_category=tactical_category,
        requested_action=requested_action,
        executed_action=requested_action if action_is_legal else None,
        response_valid=response_valid,
        action_is_legal=action_is_legal,
        action_is_optimal=action_is_optimal,
        legal_regret=legal_regret if action_is_legal else None,
        effective_regret=effective_regret,
        action_probabilities=action_probabilities,
        confidence=confidence,
        optimal_mass=optimal_mass,
        illegal_mass=illegal_mass,
        end_to_end_latency_ms=end_to_end_latency_ms,
        model_latency_ms=model_latency_ms,
        api_cost_usd=api_cost_usd,
        error_type=error_type,
    )


def _make_episode(
    *,
    run_id: str = "run_t6",
    row_id: str = "A_jev_V",
    episode_id: str = "ep_0001",
    action_mode: str = "all_9",
    opponent_type: str = "random",
    outcome: str = "win",
    total_cost_usd: float = 0.0004,
) -> EpisodeLogRow:
    cond_code, model_key, model_id, family = ROW_META[row_id]
    reward = 1.0 if outcome == "win" else (0.0 if outcome == "draw" else -1.0)
    term = "illegal_forfeit" if outcome == "loss_by_illegal_move" else ("draw" if outcome == "draw" else "win")
    return EpisodeLogRow(
        experiment_id="T6",
        run_id=run_id,
        episode_id=episode_id,
        row_id=row_id,
        condition_code=cond_code,
        model_key=model_key,
        model_id=model_id,
        family=family,
        action_mode=action_mode,  # type: ignore[arg-type]
        opponent_type=opponent_type,  # type: ignore[arg-type]
        episode_seed=42,
        agent_mark="X",
        starting_player="X",
        goes_first=True,
        outcome=outcome,  # type: ignore[arg-type]
        winner="X" if outcome == "win" else (None if outcome == "draw" else "O"),
        termination_reason=term,
        agent_reward=reward,
        num_plies=5,
        num_agent_turns=3,
        illegal_moves=1 if outcome == "loss_by_illegal_move" else 0,
        invalid_responses=0,
        optimal_agent_turns=2 if outcome == "win" else 1,
        total_cost_usd=total_cost_usd,
        total_latency_ms=300.0,
        mean_effective_regret=0.25,
        timestamp_utc="2026-11-15T00:00:00Z",
    )


def test_track1_and_track2_metrics_exact_formulas() -> None:
    # 4 decisions in one group:
    # 2 optimal legal, 1 suboptimal legal (regret=1), 1 illegal valid (eff_regret=2), 1 invalid (eff_regret=2) -> wait 5 total
    decisions = [
        _make_decision(episode_id="p1", action_is_optimal=True, action_is_legal=True, response_valid=True, legal_regret=0, effective_regret=0, end_to_end_latency_ms=100.0, api_cost_usd=0.01),
        _make_decision(episode_id="p2", action_is_optimal=True, action_is_legal=True, response_valid=True, legal_regret=0, effective_regret=0, end_to_end_latency_ms=200.0, api_cost_usd=0.01),
        _make_decision(episode_id="p3", action_is_optimal=False, action_is_legal=True, response_valid=True, legal_regret=1, effective_regret=1, end_to_end_latency_ms=300.0, api_cost_usd=0.01),
        _make_decision(episode_id="p4", action_is_optimal=False, action_is_legal=False, response_valid=True, legal_regret=None, effective_regret=2, end_to_end_latency_ms=400.0, api_cost_usd=0.01),
        _make_decision(episode_id="p5", action_is_optimal=False, action_is_legal=False, response_valid=False, requested_action=None, legal_regret=None, effective_regret=2, end_to_end_latency_ms=500.0, api_cost_usd=0.01, error_type="ParseError"),
    ]
    t1 = compute_track1_metrics(decisions)
    key = ("T2", "A_jev_V", "clean_default", "json_canonical", "all_9")
    assert key in t1
    m = t1[key]
    assert m.n == 5
    assert m.optimal_accuracy == pytest.approx(2 / 5)
    assert m.conditional_optimal_accuracy == pytest.approx(2 / 3)
    assert m.illegal_rate == pytest.approx(1 / 5)
    assert m.invalid_response_rate == pytest.approx(1 / 5)
    assert m.mean_effective_regret == pytest.approx((0 + 0 + 1 + 2 + 2) / 5)
    assert m.mean_legal_regret == pytest.approx((0 + 0 + 1) / 3)
    assert m.p50_latency_ms == pytest.approx(300.0)
    assert m.p95_latency_ms == pytest.approx(480.0)
    assert m.mean_latency_ms == pytest.approx(300.0)
    assert m.cost_per_100_decisions_usd == pytest.approx(1.0)
    assert m.optimal_decisions_per_usd == pytest.approx(2 / 0.05)

    episodes = [
        _make_episode(episode_id="e1", outcome="win", total_cost_usd=0.02),
        _make_episode(episode_id="e2", outcome="draw", total_cost_usd=0.02),
        _make_episode(episode_id="e3", outcome="loss_by_board", total_cost_usd=0.02),
        _make_episode(episode_id="e4", outcome="loss_by_illegal_move", total_cost_usd=0.02),
    ]
    t2 = compute_track2_metrics(episodes)
    t2_key = ("A_jev_V", "all_9", "random")
    assert t2_key in t2
    e = t2[t2_key]
    assert e.games == 4
    assert e.win_rate == pytest.approx(0.25)
    assert e.draw_rate == pytest.approx(0.25)
    assert e.non_loss_rate == pytest.approx(0.50)
    assert e.loss_by_board_rate == pytest.approx(0.25)
    assert e.loss_by_illegal_move_rate == pytest.approx(0.25)
    assert e.cost_per_episode_usd == pytest.approx(0.02)
    assert e.non_loss_games_per_usd == pytest.approx(2 / 0.08)


def test_paired_bootstrap_ci_and_mcnemar_exact() -> None:
    # Exact determinism with default B=10_000, seed=20261114
    x = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]
    res1 = paired_bootstrap_ci(x, y)
    res2 = paired_bootstrap_ci(x, y)
    assert res1 == res2
    assert res1.n_bootstrap == DEFAULT_BOOTSTRAP_SAMPLES == 10_000
    assert res1.seed == DEFAULT_BOOTSTRAP_SEED == 20261114
    assert res1.delta == pytest.approx(sum(x) / len(x) - sum(y) / len(y))
    assert res1.ci_low > 0.0
    assert res1.ci_low <= res1.delta <= res1.ci_high

    # Identical sequences -> 0 delta and [0, 0] CI, McNemar p=1.0
    eq = paired_bootstrap_ci(x, x)
    assert eq.delta == 0.0
    assert eq.ci_low == 0.0
    assert eq.ci_high == 0.0
    assert mcnemar_exact_pvalue(x, x) == 1.0

    # Exact McNemar checks: n10=10, n01=0 -> 2 * 0.5**10
    assert mcnemar_exact_from_counts(0, 0) == 1.0
    assert mcnemar_exact_from_counts(5, 5) == 1.0
    assert mcnemar_exact_from_counts(10, 0) == pytest.approx(2.0 * (0.5**10))
    # From binary sequences: n10=10, n01=0
    bx = [1] * 10 + [0] * 5
    by = [0] * 10 + [0] * 5
    assert mcnemar_exact_pvalue(bx, by) == pytest.approx(2.0 * (0.5**10))

    # Balanced discordant pairs (n10=2, n01=2)
    bx2 = [1, 1, 0, 0, 1]
    by2 = [0, 0, 1, 1, 1]
    expected_p = min(1.0, 2.0 * sum(math.comb(4, i) * (0.5**4) for i in range(3)))
    assert mcnemar_exact_pvalue(bx2, by2) == pytest.approx(expected_p)

    # Alignment by shared episode_id
    rows_x = [
        _make_decision(row_id="B_jev_T", episode_id=f"ep_{i:02d}", action_is_optimal=(i < 12))
        for i in range(15)
    ]
    rows_y = [
        _make_decision(row_id="A_jev_V", episode_id=f"ep_{i:02d}", action_is_optimal=(i < 2))
        for i in range(2, 17)
    ]
    comp = compare_paired_decisions(rows_x, rows_y)
    assert comp is not None
    assert comp.shared_n == 13  # ep_02 .. ep_14
    assert comp.n10 == 10  # ep_02..ep_11 optimal in X, not Y
    assert comp.n01 == 0
    assert comp.mcnemar_pvalue == pytest.approx(2.0 * (0.5**10))


def test_dm_multi_optimal_calibration_and_abstention() -> None:
    # Multi-optimal position: optimal_actions = ["R1C1", "R3C3"], probs split 0.45 + 0.45,
    # illegal leaked mass = 0.10 on occupied R1C2.
    # Leave optimal_mass=None and illegal_mass=None on row 1 to test on-the-fly computation too!
    r1 = _make_decision(
        episode_id="ep1",
        confidence=0.90,
        action_is_optimal=True,
        requested_action="R1C1",
        optimal_actions=["R1C1", "R3C3"],
        legal_actions=["R1C1", "R3C3", "R2C2"],
        action_probabilities={"R1C1": 0.45, "R3C3": 0.45, "R1C2": 0.10},
        optimal_mass=None,
        illegal_mass=None,
        api_cost_usd=0.001,
    )
    r2 = _make_decision(
        episode_id="ep2",
        confidence=0.80,
        action_is_optimal=True,
        requested_action="R2C2",
        optimal_actions=["R2C2"],
        legal_actions=["R2C2", "R1C3"],
        action_probabilities={"R2C2": 0.80, "R1C3": 0.20},
        optimal_mass=0.80,
        illegal_mass=0.00,
        api_cost_usd=0.001,
    )
    r3 = _make_decision(
        episode_id="ep3",
        confidence=0.35,
        action_is_optimal=False,
        requested_action="R1C3",
        optimal_actions=["R2C2"],
        legal_actions=["R2C2", "R1C3"],
        action_probabilities={"R1C3": 0.35, "R2C2": 0.25, "R1C1": 0.40},
        optimal_mass=0.25,
        illegal_mass=0.40,
        api_cost_usd=0.001,
    )
    r4 = _make_decision(
        episode_id="ep4",
        confidence=0.40,
        action_is_optimal=False,
        action_is_legal=False,
        requested_action="R1C1",
        optimal_actions=["R2C2"],
        legal_actions=["R2C2"],
        action_probabilities={"R1C1": 0.40, "R2C2": 0.60},
        optimal_mass=0.60,
        illegal_mass=0.40,
        api_cost_usd=0.001,
    )
    fallback = [
        _make_decision(row_id="E_ghyd_VT", episode_id="ep3", action_is_optimal=True, api_cost_usd=0.01),
        _make_decision(row_id="E_ghyd_VT", episode_id="ep4", action_is_optimal=True, api_cost_usd=0.01),
    ]

    cal = compute_dm_calibration(
        [r1, r2, r3, r4],
        taus=[0.0, 0.5, 0.85, 1.0],
        fallback_decisions=fallback,
    )
    assert cal.n == 4
    assert cal.empirical_accuracy == pytest.approx(0.5)
    assert cal.mean_confidence == pytest.approx((0.90 + 0.80 + 0.35 + 0.40) / 4)
    assert cal.mean_optimal_mass == pytest.approx((0.90 + 0.80 + 0.25 + 0.60) / 4)
    assert cal.mean_illegal_mass == pytest.approx((0.10 + 0.00 + 0.40 + 0.40) / 4)
    expected_brier = ((0.9 - 1) ** 2 + (0.8 - 1) ** 2 + (0.35 - 0) ** 2 + (0.40 - 0) ** 2) / 4
    assert cal.brier_score == pytest.approx(expected_brier)
    assert 0.0 <= cal.ece <= 1.0

    # Abstention curve checks
    pts = {pt.tau: pt for pt in cal.abstention_curve}
    assert pts[0.0].coverage == pytest.approx(1.0)
    assert pts[0.0].accuracy == pytest.approx(0.5)
    assert pts[0.5].coverage == pytest.approx(0.5)
    assert pts[0.5].accuracy == pytest.approx(1.0)
    assert pts[0.5].hybrid_oracle_accuracy == pytest.approx(1.0)
    assert pts[0.5].hybrid_fallback_accuracy == pytest.approx(1.0)
    assert pts[0.5].hybrid_fallback_cost_per_100_usd == pytest.approx((0.004 + 0.02) / 4 * 100.0)
    assert pts[0.85].coverage == pytest.approx(0.25)
    assert pts[0.85].accuracy == pytest.approx(1.0)
    assert pts[1.0].coverage == pytest.approx(0.0)
    assert pts[1.0].accuracy is None


def test_analyze_runs_generates_all_9_charts_and_report(tmp_path: Path) -> None:
    repo_root = tmp_path
    t2_dir = repo_root / "experiments" / "results" / "T2" / "run_t2"
    t3_dir = repo_root / "experiments" / "results" / "T3" / "run_t3"
    t4_dir = repo_root / "experiments" / "results" / "T4" / "run_t4"
    t5_dir = repo_root / "experiments" / "results" / "T5" / "run_t5"
    t6_dir = repo_root / "experiments" / "results" / "T6" / "run_t6"

    # Build T1+T2 across all 9 rows and all 6 tactical categories (12 positions)
    t2_logger = RunLogger(
        t2_dir,
        RunManifest(
            experiment_id="T2",
            run_id="run_t2",
            created_at_utc="2026-11-15T00:00:00Z",
            updated_at_utc="2026-11-15T00:00:00Z",
            mock_mode=True,
            rows=list(CORE_ROW_ORDER),
        ),
    )
    for row_idx, rid in enumerate(CORE_ROW_ORDER):
        is_dm = ROW_META[rid][3] == "decision_model"
        is_text = rid.endswith("_T") or rid.endswith("_VT")
        for i in range(12):
            cat = TACTICAL_CATEGORY_ORDER[i % len(TACTICAL_CATEGORY_ORDER)]
            is_opt = (i < (10 if is_text else 7))
            is_legal = True if is_text else (i != 11)
            is_valid = not (rid == "C_ghyd_V" and i == 10)
            if not is_valid:
                is_legal = False
                is_opt = False
            elif not is_legal:
                is_opt = False
            conf = (0.88 if is_opt else 0.42) if is_dm else None
            probs = (
                {"R1C1": conf, "R2C2": round(1.0 - conf, 4)}
                if (is_dm and conf is not None)
                else None
            )
            t2_logger.log_decision(
                _make_decision(
                    experiment_id="T2",
                    run_id="run_t2",
                    row_id=rid,
                    episode_id=f"pos_{i:04d}",
                    tactical_category=cat,
                    response_valid=is_valid,
                    action_is_legal=is_legal,
                    action_is_optimal=is_opt,
                    requested_action="R1C1" if is_valid else None,
                    effective_regret=0.0 if is_opt else (1.0 if is_legal else 2.0),
                    end_to_end_latency_ms=80.0 + row_idx * 35.0 + i * 2.0,
                    api_cost_usd=0.00005 if is_dm else 0.0015,
                    confidence=conf,
                    action_probabilities=probs,
                    optimal_mass=conf if (is_dm and is_opt) else (0.2 if is_dm else None),
                    illegal_mass=0.02 if is_dm else None,
                    error_type="MalformedJSON" if not is_valid else None,
                )
            )
    t2_logger.finalize()

    # T3 rendering perturbations (reusing clean_default from T2)
    t3_logger = RunLogger(t3_dir)
    for rid in ("A_jev_V", "A_gdec_V", "C_ghyd_V"):
        for var in RENDER_VARIANT_ORDER[1:]:
            for i in range(6):
                is_opt = i < 4 if var != "tilt_blur" else i < 3
                t3_logger.log_decision(
                    _make_decision(
                        experiment_id="T3",
                        run_id="run_t3",
                        row_id=rid,
                        episode_id=f"pos_{i:04d}",
                        render_variant=var,
                        action_is_optimal=is_opt,
                        effective_regret=0.0 if is_opt else 1.0,
                    )
                )

    # T4 legal_only
    t4_logger = RunLogger(t4_dir)
    for rid in CORE_ROW_ORDER:
        for i in range(12):
            is_opt = i < 10
            t4_logger.log_decision(
                _make_decision(
                    experiment_id="T4",
                    run_id="run_t4",
                    row_id=rid,
                    episode_id=f"pos_{i:04d}",
                    action_mode="legal_only",
                    action_is_optimal=is_opt,
                    effective_regret=0.0 if is_opt else 1.0,
                )
            )

    # T5 prompt formats (reusing json_canonical from T2)
    t5_logger = RunLogger(t5_dir)
    for rid in ("B_jev_T", "B_gdec_T", "F_ghyd_T"):
        for fmt in TEXT_FORMAT_ORDER[1:]:
            for i in range(6):
                is_opt = i < 5
                t5_logger.log_decision(
                    _make_decision(
                        experiment_id="T5",
                        run_id="run_t5",
                        row_id=rid,
                        episode_id=f"pos_{i:04d}",
                        text_format=fmt,
                        action_is_optimal=is_opt,
                        effective_regret=0.0 if is_opt else 1.0,
                    )
                )

    # T6 closed-loop episodes
    t6_logger = RunLogger(t6_dir)
    for rid in CORE_ROW_ORDER:
        for mode in ("all_9", "legal_only"):
            for opp in ("random", "minimax"):
                outcomes = (
                    ["win", "win", "draw", "loss_by_illegal_move"]
                    if opp == "random"
                    else ["draw", "draw", "loss_by_board", "loss_by_illegal_move"]
                )
                for idx, out in enumerate(outcomes):
                    t6_logger.log_episode(
                        _make_episode(
                            row_id=rid,
                            episode_id=f"{mode}_{opp}_{idx}",
                            action_mode=mode,
                            opponent_type=opp,
                            outcome=out,
                        )
                    )

    result = analyze_runs(
        [t2_dir, t3_dir, t4_dir, t5_dir, t6_dir],
        repo_root=repo_root,
        tag="m1_synth",
        n_bootstrap=500,
    )

    report_path = Path(result["report_path"])
    assert report_path == repo_root / "analysis" / "reports" / "m1_synth_summary.md"
    assert report_path.is_file()
    md = report_path.read_text(encoding="utf-8")
    assert "## 1. T2 Core Matrix" in md
    assert "## 2. T1 Tactical Category Breakdown" in md
    assert "## 3. T3 Rendering Robustness" in md
    assert "## 4. T4 Action-Choice Schema" in md
    assert "## 5. T5 Prompt Format Sensitivity" in md
    assert "## 6. T6 Closed-Loop Gameplay Outcomes" in md
    assert "### H1 — Structured State vs Vision-Only" in md
    assert "### H2 — Hybrid (V+T) vs Single-Modality" in md
    assert "### H3 — Decision Models vs Generative VLM/LLM" in md
    assert "## 8. Decision-Model Multi-Optimal Calibration" in md
    assert "## 9. Representative Failure Examples" in md
    assert "## 11. Milestone 1 Interim Memo Scaffold" in md

    plot_dir = repo_root / "analysis" / "plots" / "m1_synth"
    plots = result["plots"]
    assert len(plots) == 18  # 9 PNG + 9 SVG
    for chart in CHART_NAMES:
        png = plot_dir / f"{chart}.png"
        svg = plot_dir / f"{chart}.svg"
        assert png.is_file() and png.stat().st_size > 1000
        assert svg.is_file() and svg.stat().st_size > 1000
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    # Verify summary structure
    summary = result["summary"]
    assert len(summary["core_t2_metrics"]) == 9
    assert len(summary["track2_metrics"]) == 36  # 9 rows * 2 modes * 2 opponents
    assert len(summary["paired_comparisons"]["H1_structured_vs_vision"]) == 3
    assert len(summary["paired_comparisons"]["H2_hybrid_vs_single"]) == 6
    assert len(summary["paired_comparisons"]["H3_dm_vs_generative"]) == 6
    assert "jev_1_14_0" in summary["calibration"]["by_model"]
    assert "gemini_decision_1" in summary["calibration"]["by_model"]


def test_analyze_runs_skips_missing_experiments_gracefully(tmp_path: Path) -> None:
    repo_root = tmp_path
    t2_only = repo_root / "experiments" / "results" / "T2" / "run_partial"
    logger = RunLogger(t2_only)
    for i in range(4):
        logger.log_decision(
            _make_decision(
                row_id="C_ghyd_V",
                episode_id=f"ep_{i}",
                action_is_optimal=(i < 2),
                effective_regret=0.0 if i < 2 else 1.0,
            )
        )

    result = analyze_runs([t2_only], repo_root=repo_root, tag="partial", n_bootstrap=200)
    report_path = Path(result["report_path"])
    assert report_path.is_file()
    generated_stems = {Path(p).stem for p in result["plots"]}
    assert generated_stems == {
        "t2_core_accuracy_illegal",
        "t1_tactical_heatmap",
        "latency_cdf",
        "pareto_quality_vs_cost",
    }
    statuses = {s["name"]: s["generated"] for s in result["summary"]["plot_status"]}
    assert statuses["t3_rendering_robustness"] is False
    assert statuses["t4_legality_schema_delta"] is False
    assert statuses["t5_prompt_formats"] is False
    assert statuses["t6_match_outcomes"] is False
    assert statuses["dm_calibration_and_abstention"] is False
