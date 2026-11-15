"""Tests for Track 1 offline runner, Track 2 T6 match runner, resume, budget cap, and CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
from typing import Any

import pytest

from dm_bench import cli
from dm_bench.adapters.base import PolicyAdapter
from dm_bench.config import (
    BudgetTracker,
    ConditionRowConfig,
    ModelConfig,
    find_repo_root,
    load_all_model_configs,
    load_core_matrix,
)
from dm_bench.contracts import PolicyRequest, PolicyResult
from dm_bench.games.tic_tac_toe.engine import CELL_KEYS, other_mark
from dm_bench.games.tic_tac_toe.oracle import choose_minimax_action
from dm_bench.logging_io import RunLogger
from dm_bench.runners.matches import run_matches_experiment
from dm_bench.runners.offline import run_offline_experiment


def _make_temp_repo(tmp_path: Path) -> Path:
    src_root = find_repo_root()
    tmp_repo = tmp_path / "repo"
    (tmp_repo / "experiments").mkdir(parents=True)
    shutil.copy2(src_root / "pyproject.toml", tmp_repo / "pyproject.toml")
    shutil.copytree(src_root / "experiments" / "configs", tmp_repo / "experiments" / "configs")
    shutil.copytree(src_root / "experiments" / "prompts", tmp_repo / "experiments" / "prompts")
    shutil.copytree(src_root / "experiments" / "datasets", tmp_repo / "experiments" / "datasets")
    (tmp_repo / "experiments" / "results").mkdir(parents=True, exist_ok=True)
    (tmp_repo / "experiments" / "media").mkdir(parents=True, exist_ok=True)
    return tmp_repo


class _StubAdapter(PolicyAdapter):
    """Deterministic stub PolicyAdapter for testing runners, resume, budget, and forfeits."""

    def __init__(
        self,
        row_cfg: ConditionRowConfig,
        model_cfg: ModelConfig,
        budget_tracker: BudgetTracker | None = None,
        *,
        cost_per_call: float = 0.01,
        illegal_episode_ids: set[str] | None = None,
        invalid_episode_ids: set[str] | None = None,
    ) -> None:
        super().__init__(model_cfg)
        self.row_cfg = row_cfg
        self.model_cfg = model_cfg
        self.budget_tracker = budget_tracker
        self.cost_per_call = cost_per_call
        self.illegal_episode_ids = illegal_episode_ids or set()
        self.invalid_episode_ids = invalid_episode_ids or set()
        self.calls: list[tuple[str, str, str | None, str | None, str, int]] = []
        self.closed = False

    async def decide(self, request: PolicyRequest) -> PolicyResult:
        if self.budget_tracker is not None:
            self.budget_tracker.check_can_proceed()

        self.calls.append(
            (
                request.row_id,
                request.episode_id,
                request.render_variant,
                request.text_format,
                request.action_mode,
                request.step_idx,
            )
        )

        if request.episode_id in self.invalid_episode_ids:
            if self.budget_tracker is not None:
                self.budget_tracker.record(self.cost_per_call)
            return PolicyResult(
                requested_action=None,
                response_valid=False,
                error_type="MalformedOutputError",
                error_message="Simulated invalid output",
                model_latency_ms=5.0,
                end_to_end_latency_ms=6.0,
                input_tokens=100,
                output_tokens=10,
                total_tokens=110,
                api_cost_usd=self.cost_per_call,
                model_id=self.model_cfg.model_id,
                provider=self.model_cfg.provider,
            )

        if request.episode_id in self.illegal_episode_ids:
            # Pick a non-legal key (occupied or out-of-criteria) to trigger forfeit.
            occupied = [k for k in CELL_KEYS if k not in request.legal_actions]
            action = occupied[0] if occupied else "INVALID_CELL"
        else:
            # Choose first legal key deterministically.
            action = request.legal_actions[0]

        probs: dict[str, float] | None = None
        confidence: float | None = None
        if self.row_cfg.family == "decision_model":
            keys = list(request.action_criteria.keys())
            probs = {k: (0.6 if k == action else 0.4 / max(1, len(keys) - 1)) for k in keys}
            if len(keys) == 1:
                probs = {keys[0]: 1.0}
            confidence = probs.get(action, 0.5)

        if self.budget_tracker is not None:
            self.budget_tracker.record(self.cost_per_call)

        return PolicyResult(
            requested_action=action,
            action_probabilities=probs,
            confidence=confidence,
            response_valid=True,
            model_latency_ms=5.0,
            end_to_end_latency_ms=6.0,
            input_tokens=120,
            output_tokens=0 if self.row_cfg.family == "decision_model" else 15,
            thought_tokens=0,
            total_tokens=120 if self.row_cfg.family == "decision_model" else 135,
            api_cost_usd=self.cost_per_call,
            model_id=self.model_cfg.model_id,
            provider=self.model_cfg.provider,
        )

    async def close(self) -> None:
        self.closed = True


def _build_stub_adapters(
    repo_root: Path | None = None,
    *,
    mock: bool = False,
    budget_tracker: BudgetTracker | None = None,
    row_ids: list[str] | None = None,
    cost_per_call: float = 0.01,
    illegal_episode_ids: set[str] | None = None,
    invalid_episode_ids: set[str] | None = None,
    created_log: dict[str, _StubAdapter] | None = None,
) -> dict[str, tuple[ConditionRowConfig, ModelConfig, _StubAdapter]]:
    matrix = load_core_matrix(repo_root)
    models = load_all_model_configs(repo_root)
    selected = row_ids if row_ids is not None else matrix.row_ids()
    out: dict[str, tuple[ConditionRowConfig, ModelConfig, _StubAdapter]] = {}
    for rid in selected:
        rcfg = matrix.get_row(rid)
        mcfg = models[rcfg.model_key]
        adapter = _StubAdapter(
            rcfg,
            mcfg,
            budget_tracker=budget_tracker,
            cost_per_call=cost_per_call,
            illegal_episode_ids=illegal_episode_ids,
            invalid_episode_ids=invalid_episode_ids,
        )
        out[rid] = (rcfg, mcfg, adapter)
        if created_log is not None:
            created_log[rid] = adapter
    return out


async def test_offline_t2_t3_t4_t5_mock(tmp_path: Path) -> None:
    tmp_repo = _make_temp_repo(tmp_path)

    # T2 pilot (30 * 9 = 270)
    t2_dir = await run_offline_experiment("T2", repo_root=tmp_repo, pilot=True, mock=True)
    t2_logger = RunLogger(t2_dir)
    t2_manifest = t2_logger.load_manifest()
    assert t2_manifest is not None
    assert t2_manifest.status == "completed"
    assert t2_manifest.total_decisions == 270
    t2_rows = t2_logger.load_decisions()
    assert len(t2_rows) == 270
    for r in t2_rows:
        if r.row_id.endswith("_V"):
            assert r.render_variant == "clean_default"
            assert r.text_format is None
            assert len(r.frame_paths) == 1
            assert (tmp_repo / r.frame_paths[0]).is_file()
            assert r.state_text is None
        elif r.row_id.endswith("_T"):
            assert r.render_variant is None
            assert r.text_format == "json_canonical"
            assert r.frame_paths == []
            assert r.state_text is not None
        else:
            assert r.render_variant == "clean_default"
            assert r.text_format == "json_canonical"
            assert len(r.frame_paths) == 1
            assert r.state_text is not None
        if r.family == "decision_model" and r.response_valid:
            assert r.optimal_mass is not None
            assert r.illegal_mass is not None

    # T3 pilot (30 * 3 vision rows * 4 variants = 360)
    t3_dir = await run_offline_experiment("T3", repo_root=tmp_repo, pilot=True, mock=True)
    t3_logger = RunLogger(t3_dir)
    t3_rows = t3_logger.load_decisions()
    assert len(t3_rows) == 360
    assert {r.row_id for r in t3_rows} == {"A_jev_V", "A_gdec_V", "C_ghyd_V"}
    assert {r.render_variant for r in t3_rows} == {
        "labeled_margins",
        "dark_low_contrast",
        "thin_alt_font",
        "tilt_blur",
    }
    assert all(r.text_format is None for r in t3_rows)

    # T4 pilot (30 * 9 rows, legal_only = 270)
    t4_dir = await run_offline_experiment("T4", repo_root=tmp_repo, pilot=True, mock=True)
    t4_logger = RunLogger(t4_dir)
    t4_rows = t4_logger.load_decisions()
    assert len(t4_rows) == 270
    assert all(r.action_mode == "legal_only" for r in t4_rows)
    assert all(r.action_criteria_keys == r.legal_actions for r in t4_rows)

    # T5 pilot (30 * 3 text rows * 3 formats = 270)
    t5_dir = await run_offline_experiment("T5", repo_root=tmp_repo, pilot=True, mock=True)
    t5_logger = RunLogger(t5_dir)
    t5_rows = t5_logger.load_decisions()
    assert len(t5_rows) == 270
    assert {r.row_id for r in t5_rows} == {"B_jev_T", "B_gdec_T", "F_ghyd_T"}
    assert {r.text_format for r in t5_rows} == {
        "ascii_grid",
        "coord_list",
        "json_with_history",
    }
    assert all(r.render_variant is None and r.frame_paths == [] for r in t5_rows)


async def test_offline_budget_halt_and_checkpoint_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_repo = _make_temp_repo(tmp_path)

    created_phase1: dict[str, _StubAdapter] = {}
    monkeypatch.setattr(
        "dm_bench.runners.offline.build_row_adapters",
        lambda repo_root=None, *, mock=False, budget_tracker=None, row_ids=None: _build_stub_adapters(
            repo_root,
            mock=mock,
            budget_tracker=budget_tracker,
            row_ids=row_ids,
            cost_per_call=0.02,
            created_log=created_phase1,
        ),
    )

    # Phase 1: 30 states for 1 row, budget cap $0.05 -> stops after 3 calls ($0.06 >= $0.05)
    run_dir = await run_offline_experiment(
        "T2",
        repo_root=tmp_repo,
        pilot=True,
        concurrency=1,
        max_cost_usd=0.05,
        row_ids=["B_jev_T"],
    )
    logger = RunLogger(run_dir)
    manifest_1 = logger.load_manifest()
    assert manifest_1 is not None
    assert manifest_1.status == "halted_budget_cap"
    assert manifest_1.total_decisions == 3
    assert len(created_phase1["B_jev_T"].calls) == 3
    assert created_phase1["B_jev_T"].closed is True

    # Phase 2: resume with higher budget -> executes only the remaining 27 states (zero duplicates)
    created_phase2: dict[str, _StubAdapter] = {}
    monkeypatch.setattr(
        "dm_bench.runners.offline.build_row_adapters",
        lambda repo_root=None, *, mock=False, budget_tracker=None, row_ids=None: _build_stub_adapters(
            repo_root,
            mock=mock,
            budget_tracker=budget_tracker,
            row_ids=row_ids,
            cost_per_call=0.02,
            created_log=created_phase2,
        ),
    )

    resumed_dir = await run_offline_experiment(
        "T2",
        repo_root=tmp_repo,
        pilot=True,
        concurrency=4,
        max_cost_usd=10.0,
        resume_run_id=run_dir.name,
        row_ids=["B_jev_T"],
    )
    assert resumed_dir == run_dir
    manifest_2 = logger.load_manifest()
    assert manifest_2 is not None
    assert manifest_2.status == "completed"
    assert manifest_2.total_decisions == 30
    assert len(created_phase2["B_jev_T"].calls) == 27
    decisions = logger.load_decisions()
    assert len({d.resume_key() for d in decisions}) == 30

    # Phase 3: resuming a completed run makes zero calls
    created_phase3: dict[str, _StubAdapter] = {}
    monkeypatch.setattr(
        "dm_bench.runners.offline.build_row_adapters",
        lambda repo_root=None, *, mock=False, budget_tracker=None, row_ids=None: _build_stub_adapters(
            repo_root,
            mock=mock,
            budget_tracker=budget_tracker,
            row_ids=row_ids,
            cost_per_call=0.02,
            created_log=created_phase3,
        ),
    )
    await run_offline_experiment(
        "T2",
        repo_root=tmp_repo,
        pilot=True,
        max_cost_usd=10.0,
        resume_run_id=run_dir.name,
        row_ids=["B_jev_T"],
    )
    assert len(created_phase3["B_jev_T"].calls) == 0


async def test_t6_matches_balanced_turn_order_illegal_forfeit_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_repo = _make_temp_repo(tmp_path)

    # Trigger an illegal move on t6_all_9_random_001 (where opponent moves first or second,
    # and invalid response on t6_all_9_minimax_000).
    created_stubs: dict[str, _StubAdapter] = {}
    monkeypatch.setattr(
        "dm_bench.runners.matches.build_row_adapters",
        lambda repo_root=None, *, mock=False, budget_tracker=None, row_ids=None: _build_stub_adapters(
            repo_root,
            mock=mock,
            budget_tracker=budget_tracker,
            row_ids=row_ids,
            cost_per_call=0.005,
            illegal_episode_ids={"t6_all_9_random_001"},
            invalid_episode_ids={"t6_all_9_minimax_000"},
            created_log=created_stubs,
        ),
    )

    run_dir = await run_matches_experiment(
        repo_root=tmp_repo,
        pilot=True,
        concurrency=4,
        max_cost_usd=10.0,
        row_ids=["A_jev_V", "B_jev_T"],
        action_modes=["all_9", "legal_only"],
    )
    logger = RunLogger(run_dir)
    manifest = logger.load_manifest()
    assert manifest is not None
    assert manifest.status == "completed"
    # 2 rows * 2 action_modes * 2 opponents * 20 games = 160 episodes
    assert manifest.total_episodes == 160

    episodes = logger.load_episodes()
    assert len(episodes) == 160

    # Verify balanced (agent_mark, goes_first) in every 20-game slice
    for rid in ("A_jev_V", "B_jev_T"):
        for mode in ("all_9", "legal_only"):
            for opp in ("random", "minimax"):
                slice_eps = [
                    e
                    for e in episodes
                    if e.row_id == rid and e.action_mode == mode and e.opponent_type == opp
                ]
                assert len(slice_eps) == 20
                combos = [(e.agent_mark, e.goes_first) for e in slice_eps]
                assert combos.count(("X", True)) == 5
                assert combos.count(("O", False)) == 5
                assert combos.count(("O", True)) == 5
                assert combos.count(("X", False)) == 5

    # Verify illegal forfeit behavior on t6_all_9_random_001
    forfeit_eps = [e for e in episodes if e.episode_id == "t6_all_9_random_001"]
    assert len(forfeit_eps) == 2
    for ep in forfeit_eps:
        assert ep.outcome == "loss_by_illegal_move"
        assert ep.termination_reason == "illegal_forfeit"
        assert ep.winner == other_mark(ep.agent_mark)
        assert ep.agent_reward == -1.0
        assert ep.illegal_moves == 1
        assert ep.num_agent_turns == 1

    # Verify invalid response also terminates as illegal_forfeit
    invalid_eps = [e for e in episodes if e.episode_id == "t6_all_9_minimax_000"]
    assert len(invalid_eps) == 2
    for ep in invalid_eps:
        assert ep.outcome == "loss_by_illegal_move"
        assert ep.termination_reason == "illegal_forfeit"
        assert ep.invalid_responses == 1

    # Verify deterministic episode_seed shared across rows
    by_ep: dict[tuple[str, str, str], set[int]] = {}
    for ep in episodes:
        by_ep.setdefault((ep.action_mode, ep.opponent_type, ep.episode_id), set()).add(
            ep.episode_seed
        )
    assert all(len(seeds) == 1 for seeds in by_ep.values())

    # Test T6 budget halt + resume
    monkeypatch.setattr(
        "dm_bench.runners.matches.build_row_adapters",
        lambda repo_root=None, *, mock=False, budget_tracker=None, row_ids=None: _build_stub_adapters(
            repo_root,
            mock=mock,
            budget_tracker=budget_tracker,
            row_ids=row_ids,
            cost_per_call=0.05,
        ),
    )
    halted_dir = await run_matches_experiment(
        repo_root=tmp_repo,
        pilot=True,
        concurrency=1,
        max_cost_usd=0.25,
        row_ids=["B_jev_T"],
        action_modes=["all_9"],
    )
    halted_logger = RunLogger(halted_dir)
    halted_manifest = halted_logger.load_manifest()
    assert halted_manifest is not None
    assert halted_manifest.status == "halted_budget_cap"
    partial_eps = halted_logger.load_episodes()
    assert 0 < len(partial_eps) < 40

    # Resume with sufficient budget
    resumed_calls: dict[str, _StubAdapter] = {}
    monkeypatch.setattr(
        "dm_bench.runners.matches.build_row_adapters",
        lambda repo_root=None, *, mock=False, budget_tracker=None, row_ids=None: _build_stub_adapters(
            repo_root,
            mock=mock,
            budget_tracker=budget_tracker,
            row_ids=row_ids,
            cost_per_call=0.001,
            created_log=resumed_calls,
        ),
    )
    resumed_dir = await run_matches_experiment(
        repo_root=tmp_repo,
        pilot=True,
        concurrency=2,
        max_cost_usd=10.0,
        resume_run_id=halted_dir.name,
        row_ids=["B_jev_T"],
        action_modes=["all_9"],
    )
    assert resumed_dir == halted_dir
    final_manifest = halted_logger.load_manifest()
    assert final_manifest is not None
    assert final_manifest.status == "completed"
    final_eps = halted_logger.load_episodes()
    assert len(final_eps) == 40
    assert len({e.resume_key() for e in final_eps}) == 40


def test_cli_check_env_eval_and_run_discovery(tmp_path: Path) -> None:
    tmp_repo = _make_temp_repo(tmp_path)

    rc = cli.main(["--repo-root", str(tmp_repo), "check-env"])
    assert rc == 0

    rc_smoke = cli.main(
        [
            "--repo-root",
            str(tmp_repo),
            "smoke",
            "--mock",
            "--rows",
            "B_jev_T",
            "--latency-trials",
            "2",
        ]
    )
    assert rc_smoke == 0

    rc_eval = cli.main(
        [
            "--repo-root",
            str(tmp_repo),
            "eval",
            "--exp",
            "all",
            "--pilot",
            "--mock",
            "--rows",
            "A_jev_V",
            "B_jev_T",
            "D_jev_VT",
            "C_ghyd_V",
            "F_ghyd_T",
        ]
    )
    assert rc_eval == 0

    discovered = cli._discover_latest_runs(tmp_repo)
    assert {p.parent.name for p in discovered} == {"T1_T2", "T3", "T4", "T5", "T6"}

    rc_analyze = cli.main(["--repo-root", str(tmp_repo), "analyze", "--tag", "pilot_e2e"])
    assert rc_analyze == 0
    assert (tmp_repo / "analysis" / "reports" / "pilot_e2e_summary.md").is_file()

