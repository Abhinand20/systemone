"""Deterministic Tic-Tac-Toe corpus enumeration, D4/mark balancing, and stratified splitting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from dm_bench.games.tic_tac_toe.engine import (
    TTTState,
    apply_d4_state,
    canonical_d4_key,
    check_winner,
    new_ttt_state,
    other_mark,
    swap_marks_state,
)
from dm_bench.games.tic_tac_toe.oracle import TACTICAL_CATEGORIES, analyze_position

CORPUS_SEED = 20261114
EXPECTED_CANONICAL_COUNT = 593

DEV_COUNTS: dict[str, int] = {
    "immediate_win": 18,
    "forced_block": 10,
    "losing_position": 7,
    "fork_create": 5,
    "fork_defend": 5,
    "forced_draw": 5,
}

TEST_COUNTS: dict[str, int] = {
    "immediate_win": 90,
    "forced_block": 50,
    "losing_position": 30,
    "fork_create": 28,
    "fork_defend": 24,
    "forced_draw": 18,
}

ABLATION_COUNTS: dict[str, int] = {
    "immediate_win": 28,
    "forced_block": 18,
    "losing_position": 10,
    "fork_create": 10,
    "fork_defend": 8,
    "forced_draw": 6,
}

PILOT_COUNTS: dict[str, int] = {cat: 5 for cat in TACTICAL_CATEGORIES}

DATASET_RELATIVE_PATHS: dict[str, Path] = {
    "dev_50": Path("experiments/datasets/dev/ttt_dev_50.jsonl"),
    "test_240": Path("experiments/datasets/test/ttt_test_240.jsonl"),
    "ablation_80": Path("experiments/datasets/test/ttt_ablation_80.jsonl"),
    "pilot_30": Path("experiments/datasets/test/ttt_pilot_30.jsonl"),
}


@dataclass(frozen=True, slots=True)
class _CanonicalEntry:
    canonical_key: str
    canonical_hash: str
    state: TTTState
    tactical_category: str


def _canonical_hash(canonical_key: str) -> str:
    return hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:16]


def enumerate_canonical_states() -> list[_CanonicalEntry]:
    """BFS-enumerate all reachable non-terminal states (>=2 empty cells) modulo D4 + mark swap."""
    root = new_ttt_state("X")
    queue: deque[TTTState] = deque([root])
    visited_exact: set[tuple[tuple[str, ...], str]] = {(root.board, root.player_to_move)}
    canonical_map: dict[str, _CanonicalEntry] = {}

    while queue:
        state = queue.popleft()
        if check_winner(state.board) is not None:
            continue

        empty_count = state.board.count(".")
        if empty_count >= 2:
            ckey = canonical_d4_key(state.board, state.player_to_move)
            if ckey not in canonical_map:
                analysis = analyze_position(state.board, state.player_to_move)
                canonical_map[ckey] = _CanonicalEntry(
                    canonical_key=ckey,
                    canonical_hash=_canonical_hash(ckey),
                    state=state,
                    tactical_category=analysis.tactical_category,
                )

        if empty_count == 0:
            continue

        next_player = other_mark(state.player_to_move)
        for idx, cell in enumerate(state.board):
            if cell == ".":
                next_board = state.board[:idx] + (state.player_to_move,) + state.board[idx + 1 :]
                exact_key = (next_board, next_player)
                if exact_key not in visited_exact:
                    visited_exact.add(exact_key)
                    queue.append(
                        TTTState(
                            board=next_board,
                            player_to_move=next_player,
                            move_history=state.move_history + (idx,),
                            starting_player=state.starting_player,
                        )
                    )

    entries = sorted(canonical_map.values(), key=lambda e: e.canonical_key)
    if len(entries) != EXPECTED_CANONICAL_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CANONICAL_COUNT} canonical states, found {len(entries)}"
        )
    return entries


def _build_record(
    position_id: str,
    canonical_hash: str,
    state: TTTState,
    d4_sym_id: int,
) -> dict[str, Any]:
    analysis = analyze_position(state.board, state.player_to_move)
    return {
        "position_id": position_id,
        "canonical_hash": canonical_hash,
        "board": list(state.board),
        "player_to_move": state.player_to_move,
        "starting_player": state.starting_player,
        "move_history": list(state.move_history),
        "ply": len(state.move_history),
        "d4_sym_id": d4_sym_id,
        "state_value": analysis.state_value,
        "optimal_actions": list(analysis.optimal_actions),
        "action_values": dict(analysis.action_values),
        "tactical_category": analysis.tactical_category,
    }


def generate_corpora(seed: int = CORPUS_SEED) -> dict[str, list[dict[str, Any]]]:
    """Generate deterministic stratified splits (dev_50, test_240, ablation_80, pilot_30)."""
    rng = random.Random(seed)
    canonical_entries = enumerate_canonical_states()

    # Group canonical entries by tactical category in fixed canonical_key order.
    by_category: dict[str, list[_CanonicalEntry]] = {cat: [] for cat in TACTICAL_CATEGORIES}
    for entry in canonical_entries:
        by_category[entry.tactical_category].append(entry)

    # Apply random D4 orientation and balance X/O 50/50 within each category.
    odd_flip_toggle = False
    oriented_pool: dict[
        str,
        tuple[
            list[tuple[_CanonicalEntry, TTTState, int]],
            list[tuple[_CanonicalEntry, TTTState, int]],
        ],
    ] = {}
    for cat in TACTICAL_CATEGORIES:
        items = list(by_category[cat])
        oriented_items: list[tuple[_CanonicalEntry, TTTState, int]] = []
        for entry in items:
            sym_id = rng.randrange(8)
            oriented_state = apply_d4_state(entry.state, sym_id)
            oriented_items.append((entry, oriented_state, sym_id))

        rng.shuffle(oriented_items)

        # Target half of states in this category as player_to_move == "O" (rest "X").
        n_total = len(oriented_items)
        if n_total % 2 == 0:
            n_o_target = n_total // 2
        else:
            n_o_target = (n_total // 2) + (1 if odd_flip_toggle else 0)
            odd_flip_toggle = not odd_flip_toggle

        x_list: list[tuple[_CanonicalEntry, TTTState, int]] = []
        o_list: list[tuple[_CanonicalEntry, TTTState, int]] = []

        # Normalize to X-to-move first, then flip marks on the first n_o_target entries
        # so X and O are balanced 50/50 within every tactical category.
        normalized_items: list[tuple[_CanonicalEntry, TTTState, int]] = []
        for entry, st, sym_id in oriented_items:
            if st.player_to_move == "O":
                st = swap_marks_state(st)
            normalized_items.append((entry, st, sym_id))

        for pos_idx, (entry, st, sym_id) in enumerate(normalized_items):
            if pos_idx < n_o_target:
                st = swap_marks_state(st)
                o_list.append((entry, st, sym_id))
            else:
                x_list.append((entry, st, sym_id))

        rng.shuffle(x_list)
        rng.shuffle(o_list)
        oriented_pool[cat] = (x_list, o_list)

    # Materialize all 593 records with stable position_id sorted by canonical_key.
    raw_items: list[tuple[_CanonicalEntry, TTTState, int]] = []
    for cat in TACTICAL_CATEGORIES:
        x_list, o_list = oriented_pool[cat]
        raw_items.extend(x_list)
        raw_items.extend(o_list)

    raw_items.sort(key=lambda item: item[0].canonical_key)
    record_by_hash: dict[str, dict[str, Any]] = {}
    for idx, (entry, st, sym_id) in enumerate(raw_items, start=1):
        pid = f"pos_{idx:04d}"
        rec = _build_record(
            position_id=pid,
            canonical_hash=entry.canonical_hash,
            state=st,
            d4_sym_id=sym_id,
        )
        record_by_hash[entry.canonical_hash] = rec

    # Build stratified zero-overlap splits preserving 50/50 X/O balance.
    dev_50: list[dict[str, Any]] = []
    test_240: list[dict[str, Any]] = []
    ablation_80: list[dict[str, Any]] = []
    pilot_30: list[dict[str, Any]] = []

    dev_odd_toggle = False
    pilot_odd_toggle = False

    for cat in TACTICAL_CATEGORIES:
        x_list, o_list = oriented_pool[cat]
        x_recs = [record_by_hash[e.canonical_hash] for e, _, _ in x_list]
        o_recs = [record_by_hash[e.canonical_hash] for e, _, _ in o_list]

        # 1. dev_50
        dev_n = DEV_COUNTS[cat]
        if dev_n % 2 == 0:
            dev_nx = dev_n // 2
        else:
            dev_nx = (dev_n // 2) + (1 if dev_odd_toggle else 0)
            dev_odd_toggle = not dev_odd_toggle
        dev_no = dev_n - dev_nx

        dev_50.extend(x_recs[:dev_nx])
        dev_50.extend(o_recs[:dev_no])

        rem_x = x_recs[dev_nx:]
        rem_o = o_recs[dev_no:]

        # 2. test_240 (zero overlap with dev_50, exact 50/50 X/O in every category)
        test_n = TEST_COUNTS[cat]
        test_nx = test_n // 2
        test_no = test_n - test_nx
        test_cat_x = rem_x[:test_nx]
        test_cat_o = rem_o[:test_no]
        test_240.extend(test_cat_x)
        test_240.extend(test_cat_o)

        # 3. ablation_80 (subset of test_240, exact 50/50 X/O in every category)
        abl_n = ABLATION_COUNTS[cat]
        abl_nx = abl_n // 2
        abl_no = abl_n - abl_nx
        abl_cat_x = test_cat_x[:abl_nx]
        abl_cat_o = test_cat_o[:abl_no]
        ablation_80.extend(abl_cat_x)
        ablation_80.extend(abl_cat_o)

        # 4. pilot_30 (5 per category subset of ablation_80 and test_240; 15 X / 15 O total)
        pilot_n = PILOT_COUNTS[cat]
        pilot_nx = (pilot_n // 2) + (1 if pilot_odd_toggle else 0)
        pilot_odd_toggle = not pilot_odd_toggle
        pilot_no = pilot_n - pilot_nx
        pilot_30.extend(abl_cat_x[:pilot_nx])
        pilot_30.extend(abl_cat_o[:pilot_no])

    # Sort each split by position_id for clean deterministic ordering.
    for split_list in (dev_50, test_240, ablation_80, pilot_30):
        split_list.sort(key=lambda r: r["position_id"])

    return {
        "dev_50": dev_50,
        "test_240": test_240,
        "ablation_80": ablation_80,
        "pilot_30": pilot_30,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a list of corpus records as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, separators=(",", ":")))
            fh.write("\n")


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Load a Tic-Tac-Toe corpus JSONL file."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def build_all_corpora(repo_root: Path) -> dict[str, Path]:
    """Generate and write all four Tic-Tac-Toe JSONL splits under repo_root/experiments/datasets/."""
    splits = generate_corpora(seed=CORPUS_SEED)
    written: dict[str, Path] = {}
    for split_name, rel_path in DATASET_RELATIVE_PATHS.items():
        out_path = repo_root / rel_path
        write_jsonl(out_path, splits[split_name])
        written[split_name] = out_path
    return written


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[4]
    paths = build_all_corpora(root)
    for name, p in paths.items():
        print(f"Wrote {name}: {p}")
