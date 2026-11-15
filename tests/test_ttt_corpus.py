"""Unit tests for deterministic Tic-Tac-Toe corpus enumeration and stratified splits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from dm_bench.games.tic_tac_toe.corpus import (
    ABLATION_COUNTS,
    CORPUS_SEED,
    DATASET_RELATIVE_PATHS,
    DEV_COUNTS,
    EXPECTED_CANONICAL_COUNT,
    PILOT_COUNTS,
    TEST_COUNTS,
    build_all_corpora,
    enumerate_canonical_states,
    generate_corpora,
    load_corpus,
)
from dm_bench.games.tic_tac_toe.engine import (
    INDEX_TO_KEY,
    canonical_d4_key,
    new_ttt_state,
    step_ttt,
)
from dm_bench.games.tic_tac_toe.oracle import analyze_position

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_RECORD_KEYS = {
    "position_id",
    "canonical_hash",
    "board",
    "player_to_move",
    "starting_player",
    "move_history",
    "ply",
    "d4_sym_id",
    "state_value",
    "optimal_actions",
    "action_values",
    "tactical_category",
}


def test_enumerate_canonical_593_states_and_category_counts() -> None:
    entries = enumerate_canonical_states()
    assert len(entries) == EXPECTED_CANONICAL_COUNT == 593
    counts = Counter(e.tactical_category for e in entries)
    assert dict(counts) == {
        "immediate_win": 298,
        "losing_position": 89,
        "forced_block": 111,
        "fork_create": 37,
        "fork_defend": 32,
        "forced_draw": 26,
    }
    for entry in entries:
        assert entry.state.board.count(".") >= 2


def test_generate_corpora_stratification_zero_overlap_and_balance() -> None:
    splits_a = generate_corpora(CORPUS_SEED)
    splits_b = generate_corpora(CORPUS_SEED)
    assert splits_a == splits_b

    dev = splits_a["dev_50"]
    test = splits_a["test_240"]
    ablation = splits_a["ablation_80"]
    pilot = splits_a["pilot_30"]

    assert len(dev) == 50
    assert len(test) == 240
    assert len(ablation) == 80
    assert len(pilot) == 30

    assert dict(Counter(r["tactical_category"] for r in dev)) == DEV_COUNTS
    assert dict(Counter(r["tactical_category"] for r in test)) == TEST_COUNTS
    assert dict(Counter(r["tactical_category"] for r in ablation)) == ABLATION_COUNTS
    assert dict(Counter(r["tactical_category"] for r in pilot)) == PILOT_COUNTS

    # Zero canonical-key and position_id overlap between dev_50 and test_240.
    dev_canon = {canonical_d4_key(r["board"], r["player_to_move"]) for r in dev}
    test_canon = {canonical_d4_key(r["board"], r["player_to_move"]) for r in test}
    assert dev_canon.isdisjoint(test_canon)
    assert {r["canonical_hash"] for r in dev}.isdisjoint({r["canonical_hash"] for r in test})
    assert {r["position_id"] for r in dev}.isdisjoint({r["position_id"] for r in test})

    # ablation_80 and pilot_30 are exact subsets of test_240.
    test_by_id = {r["position_id"]: r for r in test}
    ablation_by_id = {r["position_id"]: r for r in ablation}
    for r in ablation:
        assert test_by_id[r["position_id"]] == r
    for r in pilot:
        assert test_by_id[r["position_id"]] == r
        assert ablation_by_id[r["position_id"]] == r

    # 50/50 X/O balance across splits and within every test/ablation category.
    assert Counter(r["player_to_move"] for r in dev) == {"X": 25, "O": 25}
    assert Counter(r["player_to_move"] for r in test) == {"X": 120, "O": 120}
    assert Counter(r["player_to_move"] for r in ablation) == {"X": 40, "O": 40}
    assert Counter(r["player_to_move"] for r in pilot) == {"X": 15, "O": 15}

    for split_records in (test, ablation):
        by_cat_turn = Counter((r["tactical_category"], r["player_to_move"]) for r in split_records)
        for cat in TEST_COUNTS:
            assert by_cat_turn[(cat, "X")] == by_cat_turn[(cat, "O")]


def test_committed_jsonl_datasets_match_builder_and_replay_cleanly(tmp_path: Path) -> None:
    tmp_paths = build_all_corpora(tmp_path)
    for split_name, rel_path in DATASET_RELATIVE_PATHS.items():
        disk_path = REPO_ROOT / rel_path
        assert disk_path.exists(), f"Missing generated dataset: {disk_path}"
        assert disk_path.read_text(encoding="utf-8") == tmp_paths[split_name].read_text(encoding="utf-8")

        records = load_corpus(disk_path)
        for rec in records:
            assert set(rec.keys()) == REQUIRED_RECORD_KEYS
            assert rec["board"].count(".") >= 2
            assert rec["ply"] == len(rec["move_history"])
            assert 0 <= rec["d4_sym_id"] < 8

            # Verify move_history replays cleanly from starting_player to board & player_to_move.
            state = new_ttt_state(rec["starting_player"])
            for idx in rec["move_history"]:
                outcome = step_ttt(state, INDEX_TO_KEY[idx])
                assert outcome.action_is_legal is True
                assert outcome.done is False
                state = outcome.next_state
            assert list(state.board) == rec["board"]
            assert state.player_to_move == rec["player_to_move"]

            # Verify oracle annotations match the oriented & mark-balanced board.
            analysis = analyze_position(rec["board"], rec["player_to_move"])
            assert rec["state_value"] == analysis.state_value
            assert rec["optimal_actions"] == list(analysis.optimal_actions)
            assert rec["action_values"] == analysis.action_values
            assert rec["tactical_category"] == analysis.tactical_category
