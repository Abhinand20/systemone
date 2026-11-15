"""Unit and integration tests for the dm-bench FastAPI trace viewer."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from dm_bench.contracts import DecisionLogRow, EpisodeLogRow, H2HMatchLogRow, ImageFrame, RunManifest
from dm_bench.games.tic_tac_toe.oracle import analyze_position
from dm_bench.games.tic_tac_toe.renderer import render_ttt_png
from dm_bench.logging_io import MediaStore, RunLogger
from dm_bench.viewer import create_app


def _make_decision_row(
    *,
    experiment_id: str,
    run_id: str,
    row_id: str,
    condition_code: str,
    model_key: str,
    model_id: str,
    family: str,
    episode_id: str,
    board: list[str],
    player_to_move: str,
    requested_action: str | None,
    executed_action: str | None,
    response_valid: bool = True,
    action_is_legal: bool = True,
    action_is_optimal: bool = True,
    legal_regret: int | None = 0,
    effective_regret: int = 0,
    render_variant: str | None = "clean_default",
    text_format: str | None = "json_canonical",
    action_mode: str = "all_9",
    frame_paths: list[str] | None = None,
    frame_hashes: list[str] | None = None,
    state_text: str | None = None,
    action_probabilities: dict[str, float] | None = None,
    confidence: float | None = None,
    error_type: str | None = None,
    step_idx: int = 0,
    ply: int = 0,
    opponent_type: str | None = None,
    termination_reason: str | None = None,
    include_oracle_fields: bool = True,
) -> DecisionLogRow:
    analysis = analyze_position(board, player_to_move)  # type: ignore[arg-type]
    return DecisionLogRow(
        experiment_id=experiment_id,
        run_id=run_id,
        row_id=row_id,
        condition_code=condition_code,
        model_key=model_key,
        model_id=model_id,
        family=family,
        episode_id=episode_id,
        step_idx=step_idx,
        timestamp_utc=f"2026-11-15T00:00:{step_idx:02d}Z",
        render_variant=render_variant,
        text_format=text_format,
        action_mode=action_mode,  # type: ignore[arg-type]
        frame_paths=frame_paths or [],
        frame_hashes=frame_hashes or [],
        state_text=state_text,
        board=board,
        player_to_move=player_to_move,
        starting_player="X",
        ply=ply,
        legal_actions=list(analysis.action_values.keys()),
        state_value=analysis.state_value if include_oracle_fields else None,
        optimal_actions=list(analysis.optimal_actions) if include_oracle_fields else [],
        action_values=dict(analysis.action_values) if include_oracle_fields else {},
        tactical_category=analysis.tactical_category if include_oracle_fields else None,
        requested_action=requested_action,
        executed_action=executed_action,
        response_valid=response_valid,
        action_is_legal=action_is_legal,
        action_is_optimal=action_is_optimal,
        legal_regret=legal_regret,
        effective_regret=effective_regret,
        action_probabilities=action_probabilities,
        confidence=confidence,
        error_type=error_type,
        model_latency_ms=45.0,
        end_to_end_latency_ms=52.0,
        input_tokens=120,
        output_tokens=4,
        api_cost_usd=0.0001,
        opponent_type=opponent_type,
        termination_reason=termination_reason,
    )


def _seed_synthetic_repo(tmp_path: Path) -> tuple[str, str]:
    store = MediaStore(repo_root=tmp_path)

    # Board 1: X has immediate win at R1C3 (row 0: X, X, .)
    board1 = ["X", "X", ".", "O", "O", ".", ".", ".", "."]
    png1 = render_ttt_png(board1, "X", variant="clean_default", position_id="pos_0001")
    saved1 = store.save_frame(ImageFrame(png_bytes=png1))

    # Board 2: forced_block for O
    board2 = ["X", "X", ".", ".", "O", ".", ".", ".", "."]
    png2 = render_ttt_png(board2, "O", variant="tilt_blur", position_id="pos_0002")
    saved2 = store.save_frame(ImageFrame(png_bytes=png2))

    # --- Offline Run (T1_T2) ---
    t2_dir = tmp_path / "experiments" / "results" / "T1_T2" / "run_001"
    t2_manifest = RunManifest(
        experiment_id="T1_T2",
        run_id="run_001",
        status="completed",
        created_at_utc="2026-11-15T00:00:00Z",
        updated_at_utc="2026-11-15T00:05:00Z",
        completed_at_utc="2026-11-15T00:05:00Z",
        rows=["A_jev_V", "B_jev_T", "C_ghyd_V", "F_ghyd_T"],
        total_decisions=5,
        valid_decisions=4,
        illegal_decisions=1,
        invalid_decisions=1,
        total_spent_usd=0.0005,
    )
    t2_logger = RunLogger(t2_dir, manifest=t2_manifest)

    probs_opt = {
        "R1C1": 0.01,
        "R1C2": 0.01,
        "R1C3": 0.88,
        "R2C1": 0.01,
        "R2C2": 0.01,
        "R2C3": 0.02,
        "R3C1": 0.02,
        "R3C2": 0.02,
        "R3C3": 0.02,
    }
    # 1) Optimal DM-V
    t2_logger.log_decision(
        _make_decision_row(
            experiment_id="T1_T2",
            run_id="run_001",
            row_id="A_jev_V",
            condition_code="A-DM-V",
            model_key="jev_1_14_0",
            model_id="jev-1.14.0",
            family="decision_model",
            episode_id="pos_0001",
            board=board1,
            player_to_move="X",
            requested_action="R1C3",
            executed_action="R1C3",
            frame_paths=[saved1.path or ""],
            frame_hashes=[saved1.sha256 or ""],
            action_probabilities=probs_opt,
            confidence=0.88,
        )
    )
    # 2) Suboptimal legal DM-T on pos_0001
    t2_logger.log_decision(
        _make_decision_row(
            experiment_id="T1_T2",
            run_id="run_001",
            row_id="B_jev_T",
            condition_code="B-DM-T",
            model_key="jev_1_14_0",
            model_id="jev-1.14.0",
            family="decision_model",
            episode_id="pos_0001",
            board=board1,
            player_to_move="X",
            requested_action="R2C3",
            executed_action="R2C3",
            action_is_legal=True,
            action_is_optimal=False,
            legal_regret=1,
            effective_regret=1,
            render_variant=None,
            text_format="json_canonical",
            state_text='{"game":"tic_tac_toe","player_to_move":"X"}',
            action_probabilities=probs_opt,
            confidence=0.51,
        )
    )
    # 3) Illegal move on pos_0001 (C_ghyd_V)
    t2_logger.log_decision(
        _make_decision_row(
            experiment_id="T1_T2",
            run_id="run_001",
            row_id="C_ghyd_V",
            condition_code="C-VLM-V",
            model_key="gemini_4_hydrogen",
            model_id="gemini-4-hydrogen",
            family="generative",
            episode_id="pos_0001",
            board=board1,
            player_to_move="X",
            requested_action="R1C1",
            executed_action=None,
            response_valid=True,
            action_is_legal=False,
            action_is_optimal=False,
            legal_regret=None,
            effective_regret=2,
            render_variant="clean_default",
            text_format=None,
            frame_paths=[saved1.path or ""],
            frame_hashes=[saved1.sha256 or ""],
        )
    )
    # 4) Invalid API/schema response on pos_0001 (F_ghyd_T, legal_only variant)
    t2_logger.log_decision(
        _make_decision_row(
            experiment_id="T1_T2",
            run_id="run_001",
            row_id="F_ghyd_T",
            condition_code="F-LLM-T",
            model_key="gemini_4_hydrogen",
            model_id="gemini-4-hydrogen",
            family="generative",
            episode_id="pos_0001",
            board=board1,
            player_to_move="X",
            requested_action=None,
            executed_action=None,
            response_valid=False,
            action_is_legal=False,
            action_is_optimal=False,
            legal_regret=None,
            effective_regret=2,
            render_variant=None,
            text_format="ascii_grid",
            action_mode="legal_only",
            state_text="ASCII GRID",
            error_type="SchemaParseError",
        )
    )
    # 5) pos_0002 with omitted oracle fields so endpoint fallback computes them
    t2_logger.log_decision(
        _make_decision_row(
            experiment_id="T1_T2",
            run_id="run_001",
            row_id="A_jev_V",
            condition_code="A-DM-V",
            model_key="jev_1_14_0",
            model_id="jev-1.14.0",
            family="decision_model",
            episode_id="pos_0002",
            board=board2,
            player_to_move="O",
            requested_action="R1C3",
            executed_action="R1C3",
            render_variant="tilt_blur",
            text_format=None,
            frame_paths=[saved2.path or ""],
            frame_hashes=[saved2.sha256 or ""],
            include_oracle_fields=False,
        )
    )

    # --- Closed-Loop Run (T6) ---
    t6_dir = tmp_path / "experiments" / "results" / "T6" / "run_t6"
    t6_manifest = RunManifest(
        experiment_id="T6",
        run_id="run_t6",
        status="completed",
        created_at_utc="2026-11-15T01:00:00Z",
        updated_at_utc="2026-11-15T01:05:00Z",
        completed_at_utc="2026-11-15T01:05:00Z",
        rows=["A_jev_V", "B_jev_T"],
        total_decisions=3,
        total_episodes=2,
        total_spent_usd=0.0003,
    )
    t6_logger = RunLogger(t6_dir, manifest=t6_manifest)

    # Shared episode_id across two rows: A_jev_V and B_jev_T
    t6_logger.log_episode(
        EpisodeLogRow(
            experiment_id="T6",
            run_id="run_t6",
            episode_id="ep_0001",
            row_id="A_jev_V",
            condition_code="A-DM-V",
            model_key="jev_1_14_0",
            model_id="jev-1.14.0",
            family="decision_model",
            action_mode="all_9",
            opponent_type="minimax",
            episode_seed=101,
            agent_mark="X",
            starting_player="X",
            goes_first=True,
            outcome="draw",
            winner=None,
            termination_reason="draw",
            agent_reward=0.0,
            num_plies=9,
            num_agent_turns=2,
            timestamp_utc="2026-11-15T01:01:00Z",
        )
    )
    t6_logger.log_episode(
        EpisodeLogRow(
            experiment_id="T6",
            run_id="run_t6",
            episode_id="ep_0001",
            row_id="B_jev_T",
            condition_code="B-DM-T",
            model_key="jev_1_14_0",
            model_id="jev-1.14.0",
            family="decision_model",
            action_mode="all_9",
            opponent_type="random",
            episode_seed=101,
            agent_mark="O",
            starting_player="X",
            goes_first=False,
            outcome="loss_by_illegal_move",
            winner="X",
            termination_reason="illegal_forfeit",
            agent_reward=-1.0,
            num_plies=2,
            num_agent_turns=1,
            illegal_moves=1,
            timestamp_utc="2026-11-15T01:02:00Z",
        )
    )

    # Log step_idx=1 BEFORE step_idx=0 to verify replay sorting!
    t6_logger.log_decision(
        _make_decision_row(
            experiment_id="T6",
            run_id="run_t6",
            row_id="A_jev_V",
            condition_code="A-DM-V",
            model_key="jev_1_14_0",
            model_id="jev-1.14.0",
            family="decision_model",
            episode_id="ep_0001",
            board=board1,
            player_to_move="X",
            requested_action="R1C3",
            executed_action="R1C3",
            step_idx=1,
            ply=4,
            frame_paths=[saved1.path or ""],
            opponent_type="minimax",
            termination_reason="draw",
        )
    )
    t6_logger.log_decision(
        _make_decision_row(
            experiment_id="T6",
            run_id="run_t6",
            row_id="A_jev_V",
            condition_code="A-DM-V",
            model_key="jev_1_14_0",
            model_id="jev-1.14.0",
            family="decision_model",
            episode_id="ep_0001",
            board=[".", ".", ".", ".", ".", ".", ".", ".", "."],
            player_to_move="X",
            requested_action="R2C2",
            executed_action="R2C2",
            step_idx=0,
            ply=0,
            frame_paths=[saved1.path or ""],
            opponent_type="minimax",
        )
    )
    t6_logger.log_decision(
        _make_decision_row(
            experiment_id="T6",
            run_id="run_t6",
            row_id="B_jev_T",
            condition_code="B-DM-T",
            model_key="jev_1_14_0",
            model_id="jev-1.14.0",
            family="decision_model",
            episode_id="ep_0001",
            board=["X", ".", ".", ".", ".", ".", ".", ".", "."],
            player_to_move="O",
            requested_action="R1C1",
            executed_action=None,
            action_is_legal=False,
            action_is_optimal=False,
            legal_regret=None,
            effective_regret=1,
            step_idx=0,
            ply=1,
            opponent_type="random",
            termination_reason="illegal_forfeit",
        )
    )

    return Path(saved1.path or "").name, Path(saved2.path or "").name


def test_empty_results_handling(tmp_path: Path) -> None:
    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "dm-bench Trace Viewer" in res_index.text

    res_runs = client.get("/api/runs")
    assert res_runs.status_code == 200
    assert res_runs.json() == {"runs": []}

    res_missing = client.get("/api/runs/T1_T2/does_not_exist/decisions")
    assert res_missing.status_code == 404


def test_media_serving_and_path_traversal_protection(tmp_path: Path) -> None:
    frame_name, _ = _seed_synthetic_repo(tmp_path)
    secret_file = tmp_path / "secret.png"
    secret_file.write_bytes(b"NOT_FOR_PUBLIC")

    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    ok_res = client.get(f"/media/frames/{frame_name}")
    assert ok_res.status_code == 200
    assert ok_res.headers["content-type"] == "image/png"
    assert ok_res.content.startswith(b"\x89PNG")

    missing_res = client.get("/media/frames/nonexistent.png")
    assert missing_res.status_code == 404

    traversal_paths = [
        "/media/frames/..%2F..%2Fsecret.png",
        "/media/frames/%2e%2e%2fsecret.png",
        "/media/frames/subdir/foo.png",
        "/media/frames/..%5Csecret.png",
        "/media/frames/not_a_png.txt",
    ]
    for path in traversal_paths:
        res = client.get(path)
        assert res.status_code == 400, f"Expected 400 for {path}, got {res.status_code}"


def test_runs_decisions_filters_and_pagination(tmp_path: Path) -> None:
    _seed_synthetic_repo(tmp_path)
    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    runs_res = client.get("/api/runs")
    assert runs_res.status_code == 200
    runs = runs_res.json()["runs"]
    assert len(runs) == 2
    assert {r["experiment_id"] for r in runs} == {"T1_T2", "T6"}

    # All decisions + facets
    all_res = client.get("/api/runs/T1_T2/run_001/decisions")
    assert all_res.status_code == 200
    body = all_res.json()
    assert body["total"] == 5
    assert len(body["decisions"]) == 5
    assert set(body["facets"]["row_ids"]) == {"A_jev_V", "B_jev_T", "C_ghyd_V", "F_ghyd_T"}
    assert set(body["facets"]["tactical_categories"]) == {"immediate_win"}

    # Failure filter variants
    assert client.get("/api/runs/T1_T2/run_001/decisions?failure_filter=failures").json()["total"] == 3
    illegal = client.get("/api/runs/T1_T2/run_001/decisions?failure_filter=illegal").json()
    assert illegal["total"] == 1
    assert illegal["decisions"][0]["row_id"] == "C_ghyd_V"

    suboptimal = client.get("/api/runs/T1_T2/run_001/decisions?failure_filter=suboptimal").json()
    assert suboptimal["total"] == 1
    assert suboptimal["decisions"][0]["row_id"] == "B_jev_T"

    invalid = client.get("/api/runs/T1_T2/run_001/decisions?failure_filter=invalid").json()
    assert invalid["total"] == 1
    assert invalid["decisions"][0]["row_id"] == "F_ghyd_T"

    # Parameter filters
    assert client.get("/api/runs/T1_T2/run_001/decisions?condition=A-DM-V").json()["total"] == 2
    assert client.get("/api/runs/T1_T2/run_001/decisions?render_variant=tilt_blur").json()["total"] == 1
    assert client.get("/api/runs/T1_T2/run_001/decisions?text_format=ascii_grid").json()["total"] == 1
    assert client.get("/api/runs/T1_T2/run_001/decisions?action_mode=legal_only").json()["total"] == 1

    # Pagination
    p1 = client.get("/api/runs/T1_T2/run_001/decisions?offset=1&limit=2").json()
    assert p1["total"] == 5
    assert len(p1["decisions"]) == 2
    assert p1["decisions"][0]["row_id"] == "B_jev_T"


def test_matrix_grouping_and_oracle_values(tmp_path: Path) -> None:
    _seed_synthetic_repo(tmp_path)
    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    res = client.get("/api/runs/T1_T2/run_001/matrix/pos_0001")
    assert res.status_code == 200
    data = res.json()
    assert data["episode_id"] == "pos_0001"
    assert data["optimal_actions"] == ["R1C3"]
    assert data["action_values"]["R1C3"] == 1
    assert data["tactical_category"] == "immediate_win"
    assert [r["row_id"] for r in data["rows"]] == ["A_jev_V", "B_jev_T", "C_ghyd_V", "F_ghyd_T"]
    assert data["frame_urls"][0].startswith("/media/frames/")

    # Variant filter on matrix
    filtered = client.get("/api/runs/T1_T2/run_001/matrix/pos_0001?action_mode=legal_only")
    assert filtered.status_code == 200
    assert [r["row_id"] for r in filtered.json()["rows"]] == ["F_ghyd_T"]

    # Oracle fallback computation when action_values/optimal_actions omitted in JSONL
    fallback = client.get("/api/runs/T1_T2/run_001/matrix/pos_0002")
    assert fallback.status_code == 200
    fb_data = fallback.json()
    assert fb_data["optimal_actions"] == ["R1C3"]
    assert fb_data["action_values"]["R1C3"] == 0
    assert fb_data["tactical_category"] == "forced_block"

    assert client.get("/api/runs/T1_T2/run_001/matrix/unknown_pos").status_code == 404


def test_episodes_list_and_ordered_replay(tmp_path: Path) -> None:
    _seed_synthetic_repo(tmp_path)
    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    # Offline run has empty episodes list (200)
    empty_ep = client.get("/api/runs/T1_T2/run_001/episodes")
    assert empty_ep.status_code == 200
    assert empty_ep.json()["episodes"] == []

    # T6 episodes filtering
    all_ep = client.get("/api/runs/T6/run_t6/episodes").json()
    assert all_ep["total"] == 2
    assert set(all_ep["facets"]["opponent_types"]) == {"minimax", "random"}

    filtered_ep = client.get(
        "/api/runs/T6/run_t6/episodes?opponent_type=random&outcome=loss_by_illegal_move"
    ).json()
    assert filtered_ep["total"] == 1
    assert filtered_ep["episodes"][0]["row_id"] == "B_jev_T"

    # Episode replay ordering & row_id disambiguation
    replay_a = client.get("/api/runs/T6/run_t6/episodes/ep_0001?row_id=A_jev_V")
    assert replay_a.status_code == 200
    ra = replay_a.json()
    assert ra["episode"]["outcome"] == "draw"
    assert [d["step_idx"] for d in ra["decisions"]] == [0, 1]
    assert ra["decisions"][0]["requested_action"] == "R2C2"
    assert ra["decisions"][1]["requested_action"] == "R1C3"
    assert set(ra["available_rows"]) == {"A_jev_V", "B_jev_T"}

    replay_b = client.get("/api/runs/T6/run_t6/episodes/ep_0001?row_id=B_jev_T")
    assert replay_b.status_code == 200
    rb = replay_b.json()
    assert rb["episode"]["outcome"] == "loss_by_illegal_move"
    assert len(rb["decisions"]) == 1
    assert rb["decisions"][0]["termination_reason"] == "illegal_forfeit"

    assert client.get("/api/runs/T6/run_t6/episodes/ep_missing").status_code == 404


def test_index_ui_plain_english_and_no_legacy_jargon(tmp_path: Path) -> None:
    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    res_index = client.get("/")
    assert res_index.status_code == 200
    html = res_index.text

    assert "dm-bench Trace Viewer" in html
    for expected in (
        "Overall Results",
        "Board Puzzles",
        "Full Games",
        "Best Move Guide",
        "Model Responses on This Board",
    ):
        assert expected in html

    for legacy in (
        "T1–T6 Inspector",
        "9-Way Matrix",
        "Closed-Loop Episode Scrubber",
    ):
        assert legacy not in html


def test_summary_endpoint_run_and_suite_scopes(tmp_path: Path) -> None:
    _seed_synthetic_repo(tmp_path)

    # Seed a run with opposite mock_mode (mock_mode=True) so suite aggregation stays isolated.
    t3_dir = tmp_path / "experiments" / "results" / "T3" / "run_mock"
    t3_manifest = RunManifest(
        experiment_id="T3",
        run_id="run_mock",
        run_name="Blur Baseline",
        status="completed",
        mock_mode=True,
        created_at_utc="2026-11-15T02:00:00Z",
        updated_at_utc="2026-11-15T02:01:00Z",
        completed_at_utc="2026-11-15T02:01:00Z",
        rows=["A_jev_V"],
        total_decisions=2,
        valid_decisions=2,
        total_spent_usd=0.0,
    )
    t3_logger = RunLogger(t3_dir, manifest=t3_manifest)
    t3_logger.log_decision(
        _make_decision_row(
            experiment_id="T3",
            run_id="run_mock",
            row_id="A_jev_V",
            condition_code="A-DM-V",
            model_key="jev_1_14_0",
            model_id="jev-1.14.0",
            family="decision_model",
            episode_id="pos_0099",
            board=["X", "X", ".", "O", "O", ".", ".", ".", "."],
            player_to_move="X",
            requested_action="R1C3",
            executed_action="R1C3",
            render_variant="tilt_blur",
        )
    )
    t3_logger.log_decision(
        _make_decision_row(
            experiment_id="T3",
            run_id="run_mock",
            row_id="A_jev_V",
            condition_code="A-DM-V",
            model_key="jev_1_14_0",
            model_id="jev-1.14.0",
            family="decision_model",
            episode_id="pos_0100",
            board=["X", "X", ".", ".", "O", ".", ".", ".", "."],
            player_to_move="O",
            requested_action="R1C3",
            executed_action="R1C3",
            render_variant="clean_default",
        )
    )

    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    # 1) Run scope on T1_T2/run_001
    run_res = client.get("/api/runs/T1_T2/run_001/summary?scope=run&n_bootstrap=200")
    assert run_res.status_code == 200
    run_data = run_res.json()
    assert run_data["scope"] == "run"
    assert run_data["mock_mode"] is False
    assert [(r["experiment_id"], r["run_id"]) for r in run_data["included_runs"]] == [
        ("T1_T2", "run_001")
    ]
    lb_rows = run_data["leaderboard"]["rows"]
    assert {r["row_id"] for r in lb_rows} == {"A_jev_V", "B_jev_T", "C_ghyd_V", "F_ghyd_T"}
    for r in lb_rows:
        assert r["optimal_ci_low"] <= r["optimal_accuracy"] <= r["optimal_ci_high"]

    tactical_cats = set(run_data["tactical_breakdown"]["categories"])
    assert "immediate_win" in tactical_cats
    assert "jev_1_14_0" in run_data["calibration"]["by_model"]
    assert run_data["full_games"]["rows"] == []

    # 2) Suite scope on T1_T2/run_001 combines T1_T2 + T6 (both mock_mode=False) and excludes T3/run_mock
    suite_res = client.get("/api/runs/T1_T2/run_001/summary?scope=suite&n_bootstrap=200")
    assert suite_res.status_code == 200
    suite_data = suite_res.json()
    assert suite_data["scope"] == "suite"
    included_pairs = {(r["experiment_id"], r["run_id"]) for r in suite_data["included_runs"]}
    assert included_pairs == {("T1_T2", "run_001"), ("T6", "run_t6")}
    assert len(suite_data["full_games"]["rows"]) == 2

    # 3) Suite scope on mock run T3/run_mock stays isolated to mock runs only
    mock_res = client.get("/api/runs/T3/run_mock/summary?scope=suite&n_bootstrap=200")
    assert mock_res.status_code == 200
    mock_data = mock_res.json()
    assert mock_data["mock_mode"] is True
    assert [(r["experiment_id"], r["run_id"], r["run_name"]) for r in mock_data["included_runs"]] == [
        ("T3", "run_mock", "Blur Baseline")
    ]
    runs_list = client.get("/api/runs").json()["runs"]
    t3_entry = next(r for r in runs_list if r["run_id"] == "run_mock")
    assert t3_entry["run_name"] == "Blur Baseline"
    html = client.get("/").text
    assert "run.run_name" in html
    assert "r.run_name" in html
    assert {"immediate_win", "forced_block"}.issubset(
        set(mock_data["tactical_breakdown"]["categories"])
    )
    assert mock_data["full_games"]["rows"] == []

    # 4) Missing run and path validation
    assert client.get("/api/runs/T1_T2/does_not_exist/summary").status_code == 404
    assert client.get("/api/runs/T1_T2/..%5Cetc/summary").status_code == 400
    assert client.get("/api/runs/bad..exp/run_001/summary").status_code == 400


def test_viewer_h2h_summary_episodes_and_interleaved_replay(tmp_path: Path) -> None:
    results = tmp_path / "experiments" / "results"
    h2h_dir = results / "H2H" / "run_h2h_test"
    h2h_logger = RunLogger(
        h2h_dir,
        manifest=RunManifest(
            experiment_id="H2H",
            run_id="run_h2h_test",
            run_name="Jev vs Gdec",
            created_at_utc="2026-11-15T00:00:00Z",
            updated_at_utc="2026-11-15T00:00:00Z",
            rows=["A_jev_V", "A_gdec_V"],
            action_modes=["all_9"],
        ),
    )

    # Seed an H2H match between A_jev_V and A_gdec_V with interleaved turns at ply 0 & ply 1
    match_row = H2HMatchLogRow(
        run_id="run_h2h_test",
        matchup_id="A_jev_V_vs_A_gdec_V",
        episode_id="h2h_all_9_000",
        game_idx=0,
        action_mode="all_9",
        opening_mode="standard",
        opening_moves=[],
        row_a="A_jev_V",
        condition_code_a="A-DM-V",
        model_key_a="jev_1_14_0",
        model_id_a="jev-1.14.0",
        modality_a="image",
        mark_a="X",
        row_b="A_gdec_V",
        condition_code_b="A-DM-V",
        model_key_b="gemini_decision_1",
        model_id_b="gemini-decision-1",
        modality_b="image",
        mark_b="O",
        starting_mark="X",
        a_goes_first=True,
        outcome="player_a_win",
        outcome_a="win",
        outcome_b="loss",
        winner_row_id="A_jev_V",
        winner_mark="X",
        termination_reason="win",
        num_plies=2,
        num_model_turns=2,
        turns_a=1,
        turns_b=1,
        optimal_turns_a=1,
        optimal_turns_b=0,
        mean_effective_regret_a=0.0,
        mean_effective_regret_b=1.0,
        cost_usd_a=0.001,
        cost_usd_b=0.002,
        total_cost_usd=0.003,
        latency_ms_a=50.0,
        latency_ms_b=80.0,
        total_latency_ms=130.0,
        final_board="X...O....",
        move_history=["R1C1", "R2C2"],
        timestamp_utc="2026-11-15T00:00:02Z",
    )
    h2h_logger.log_h2h_match(match_row)

    turn_a = _make_decision_row(
        experiment_id="H2H",
        run_id="run_h2h_test",
        row_id="A_jev_V",
        condition_code="A-DM-V",
        model_key="jev_1_14_0",
        model_id="jev-1.14.0",
        family="decision_model",
        episode_id="h2h_all_9_000",
        board=[".", ".", ".", ".", ".", ".", ".", ".", "."],
        player_to_move="X",
        requested_action="R1C1",
        executed_action="R1C1",
        step_idx=0,
        ply=0,
        opponent_type="h2h:A_gdec_V",
    ).model_copy(
        update={
            "matchup_id": "A_jev_V_vs_A_gdec_V",
            "opponent_row_id": "A_gdec_V",
            "opponent_model_key": "gemini_decision_1",
        }
    )
    turn_b = _make_decision_row(
        experiment_id="H2H",
        run_id="run_h2h_test",
        row_id="A_gdec_V",
        condition_code="A-DM-V",
        model_key="gemini_decision_1",
        model_id="gemini-decision-1",
        family="decision_model",
        episode_id="h2h_all_9_000",
        board=["X", ".", ".", ".", ".", ".", ".", ".", "."],
        player_to_move="O",
        requested_action="R1C2",
        executed_action="R1C2",
        action_is_optimal=False,
        legal_regret=1,
        effective_regret=1,
        step_idx=1,
        ply=1,
        opponent_type="h2h:A_jev_V",
        termination_reason="win",
    ).model_copy(
        update={
            "matchup_id": "A_jev_V_vs_A_gdec_V",
            "opponent_row_id": "A_jev_V",
            "opponent_model_key": "jev_1_14_0",
        }
    )
    h2h_logger.log_decision(turn_a)
    h2h_logger.log_decision(turn_b)
    h2h_logger.finalize(status="completed")
    manifest = h2h_logger.load_manifest()
    assert manifest is not None
    assert manifest.run_name == "Jev vs Gdec"

    client = TestClient(create_app(repo_root=tmp_path))

    runs_data = client.get("/api/runs").json()["runs"]
    assert len(runs_data) == 1
    assert runs_data[0]["experiment_id"] == "H2H"
    assert runs_data[0]["has_h2h_matches"] is True
    assert runs_data[0]["has_episodes"] is True
    assert runs_data[0]["run_name"] == "Jev vs Gdec"

    sum_res = client.get("/api/runs/H2H/run_h2h_test/summary?scope=suite&n_bootstrap=200")
    assert sum_res.status_code == 200
    sdata = sum_res.json()
    assert sdata["totals"]["total_episodes"] == 1
    assert sdata["h2h"]["total_matches"] == 1
    assert len(sdata["h2h"]["matchups"]) == 1
    assert sdata["h2h"]["matchups"][0]["wins_a"] == 1
    assert [s["row_id"] for s in sdata["h2h"]["standings"]] == ["A_jev_V", "A_gdec_V"]

    eps_res = client.get("/api/runs/H2H/run_h2h_test/episodes?row_id=A_gdec_V")
    assert eps_res.status_code == 200
    ep_payload = eps_res.json()
    assert ep_payload["is_h2h"] is True
    assert ep_payload["total"] == 1
    assert ep_payload["facets"]["matchup_ids"] == ["A_jev_V_vs_A_gdec_V"]

    replay_res = client.get(
        "/api/runs/H2H/run_h2h_test/episodes/h2h_all_9_000?matchup_id=A_jev_V_vs_A_gdec_V&action_mode=all_9"
    )
    assert replay_res.status_code == 200
    replay = replay_res.json()
    assert replay["is_h2h"] is True
    assert replay["available_rows"] == ["A_jev_V", "A_gdec_V"]
    assert [(d["row_id"], d["player_side"], d["ply"]) for d in replay["decisions"]] == [
        ("A_jev_V", "A", 0),
        ("A_gdec_V", "B", 1),
    ]


def test_viewer_reconciles_legacy_rate_limit_episodes_and_labels(tmp_path: Path) -> None:
    t6_dir = tmp_path / "experiments" / "results" / "T6" / "run_legacy_rl"
    t6_logger = RunLogger(
        t6_dir,
        manifest=RunManifest(
            experiment_id="T6",
            run_id="run_legacy_rl",
            status="completed",
            created_at_utc="2026-11-15T01:00:00Z",
            updated_at_utc="2026-11-15T01:05:00Z",
            rows=["A_gdec_V"],
            total_decisions=2,
            total_episodes=2,
        ),
    )
    t6_logger.log_episode(
        EpisodeLogRow(
            experiment_id="T6",
            run_id="run_legacy_rl",
            episode_id="ep_win",
            row_id="A_gdec_V",
            condition_code="A-DM-V",
            model_key="gemini_decision_1",
            model_id="gemini-decision-1",
            family="decision_model",
            action_mode="legal_only",
            opponent_type="random",
            episode_seed=1,
            agent_mark="X",
            starting_player="X",
            goes_first=True,
            outcome="win",
            winner="X",
            termination_reason="win",
            agent_reward=1.0,
            num_plies=5,
            num_agent_turns=3,
            timestamp_utc="2026-11-15T01:01:00Z",
        )
    )
    # Legacy polluted RateLimitError episode stored as loss_by_illegal_move
    t6_logger.log_episode(
        EpisodeLogRow(
            experiment_id="T6",
            run_id="run_legacy_rl",
            episode_id="ep_rl",
            row_id="A_gdec_V",
            condition_code="A-DM-V",
            model_key="gemini_decision_1",
            model_id="gemini-decision-1",
            family="decision_model",
            action_mode="legal_only",
            opponent_type="random",
            episode_seed=2,
            agent_mark="O",
            starting_player="X",
            goes_first=False,
            outcome="loss_by_illegal_move",
            winner="X",
            termination_reason="illegal_forfeit",
            agent_reward=-1.0,
            num_plies=1,
            num_agent_turns=1,
            illegal_moves=1,
            invalid_responses=1,
            timestamp_utc="2026-11-15T01:02:00Z",
        )
    )
    t6_logger.log_decision(
        _make_decision_row(
            experiment_id="T6",
            run_id="run_legacy_rl",
            row_id="A_gdec_V",
            condition_code="A-DM-V",
            model_key="gemini_decision_1",
            model_id="gemini-decision-1",
            family="decision_model",
            episode_id="ep_rl",
            board=["X", ".", ".", ".", ".", ".", ".", ".", "."],
            player_to_move="O",
            requested_action=None,
            executed_action=None,
            response_valid=False,
            action_is_legal=False,
            action_is_optimal=False,
            legal_regret=None,
            effective_regret=2,
            action_mode="legal_only",
            opponent_type="random",
            error_type="RateLimitError",
            termination_reason=None,
        ).model_copy(update={"error_message": "429 RESOURCE_EXHAUSTED"})
    )

    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    eps_body = client.get("/api/runs/T6/run_legacy_rl/episodes").json()
    assert eps_body["total"] == 2
    by_id = {e["episode_id"]: e for e in eps_body["episodes"]}
    assert by_id["ep_rl"]["outcome"] == "aborted_error"
    assert by_id["ep_rl"]["termination_reason"] == "rate_limit"
    assert by_id["ep_rl"]["illegal_moves"] == 0
    assert by_id["ep_rl"]["agent_reward"] is None

    replay_body = client.get("/api/runs/T6/run_legacy_rl/episodes/ep_rl").json()
    assert replay_body["episode"]["outcome"] == "aborted_error"
    assert replay_body["decisions"][0]["error_category"] == "rate_limit"
    assert replay_body["decisions"][0]["effective_regret"] is None

    summary_body = client.get("/api/runs/T6/run_legacy_rl/summary?scope=run&n_bootstrap=200").json()
    fg_row = summary_body["full_games"]["rows"][0]
    assert fg_row["games"] == 1
    assert fg_row["wins"] == 1
    assert fg_row["losses_by_illegal_move"] == 0
    assert fg_row["aborted_errors"] == 1
    assert fg_row["rate_limit_errors"] == 1
    assert fg_row["win_rate"] == 1.0

    html = client.get("/").text
    assert "aborted_error" in html
    assert "Aborted (API Error)" in html

