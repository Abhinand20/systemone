"""Unit tests for policy adapters, transient retry logic, BudgetTracker gating, and M0 smoke."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from dm_bench.adapters import (
    BudgetedPolicyAdapter,
    GeminiDecisionAdapter,
    GeminiGenerativeAdapter,
    MalformedResponseError,
    MockDecisionAdapter,
    MockGenerativeAdapter,
    SafetyBlockError,
    SchemaMismatchError,
    TypeSafeJevAdapter,
    build_row_adapters,
    call_with_retry,
    classify_error,
    close_adapters,
    extract_retry_after_seconds,
    infer_error_category,
    is_infra_error,
    is_rate_limit_error,
    is_transient_error,
)
from dm_bench.adapters.base import _resolve_optimal_actions
from dm_bench.config import (
    BudgetExceededError,
    BudgetTracker,
    PricingConfig,
    PromptConfig,
    find_repo_root,
    load_all_model_configs,
    load_prompt_config,
)
from dm_bench.contracts import DecisionLogRow, ImageFrame, PolicyRequest, RunManifest
from dm_bench.games.tic_tac_toe.engine import ACTION_CRITERIA, CELL_KEYS, legal_action_keys
from dm_bench.games.tic_tac_toe.formats import format_ttt_state
from dm_bench.games.tic_tac_toe.oracle import analyze_position
from dm_bench.games.tic_tac_toe.renderer import render_ttt_png
from dm_bench.logging_io import read_jsonl
from dm_bench.smoke import run_smoke


class FakeHTTPStatusError(Exception):
    def __init__(self, status_code: int, message: str = "http error") -> None:
        self.status_code = status_code
        super().__init__(message)


class FakeTypeSafeStatusError(Exception):
    def __init__(self, status: int, message: str = "typesafe error") -> None:
        self.status = status
        super().__init__(message)


def _sample_board_and_request(
    *,
    row_id: str = "D_jev_VT",
    condition_code: str = "D-DM-VT",
    model_key: str = "jev_1_14_0",
    use_vision: bool = True,
    use_text: bool = True,
    render_variant: str | None = "clean_default",
    text_format: str | None = "json_canonical",
    action_mode: str = "all_9",
    episode_id: str = "pos_test_001",
    step_idx: int = 0,
) -> PolicyRequest:
    board = ["X", "X", ".", ".", "O", ".", ".", ".", "."]
    player = "O"
    legal = list(legal_action_keys(board))
    analysis = analyze_position(board, player)
    criteria = (
        dict(ACTION_CRITERIA)
        if action_mode == "all_9"
        else {k: v for k, v in ACTION_CRITERIA.items() if k in set(legal)}
    )
    frames = (
        [
            ImageFrame(
                frame_id="board_image",
                png_bytes=render_ttt_png(board, player, variant=render_variant or "clean_default"),
                width=512,
                height=512,
            )
        ]
        if use_vision
        else []
    )
    state_text = (
        format_ttt_state(board, player, text_format=text_format or "json_canonical", move_history=[0, 4, 1])
        if use_text
        else None
    )
    return PolicyRequest(
        episode_id=episode_id,
        step_idx=step_idx,
        row_id=row_id,
        condition_code=condition_code,
        model_key=model_key,
        objective_instructions="Choose the single best Tic-Tac-Toe move.",
        action_criteria=criteria,
        frames=frames,
        state_text=state_text,
        render_variant=render_variant if use_vision else None,
        text_format=text_format if use_text else None,
        action_mode=action_mode,  # type: ignore[arg-type]
        player_to_move=player,
        legal_actions=legal,
        metadata={"board": board, "optimal_actions": list(analysis.optimal_actions)},
    )


async def test_mock_adapters_deterministic_and_modality_aware() -> None:
    root = find_repo_root()
    cfgs = load_all_model_configs(root)

    dm_adapter = MockDecisionAdapter(cfgs["jev_1_14_0"])
    gen_adapter = MockGenerativeAdapter(cfgs["gemini_4_hydrogen"])

    req_dm = _sample_board_and_request(row_id="D_jev_VT", model_key="jev_1_14_0")
    r1 = await dm_adapter.decide(req_dm)
    r2 = await dm_adapter.decide(req_dm)
    assert r1 == r2
    assert r1.response_valid is True
    assert r1.requested_action in CELL_KEYS
    assert r1.action_probabilities is not None
    assert set(r1.action_probabilities.keys()) == set(CELL_KEYS)
    assert abs(sum(r1.action_probabilities.values()) - 1.0) < 1e-5
    assert r1.confidence == pytest.approx(r1.action_probabilities[r1.requested_action])
    assert r1.input_tokens > 0
    assert r1.api_cost_usd > 0

    req_gen = _sample_board_and_request(
        row_id="E_ghyd_VT",
        condition_code="E-VLM-VT",
        model_key="gemini_4_hydrogen",
    )
    g1 = await gen_adapter.decide(req_gen)
    g2 = await gen_adapter.decide(req_gen)
    assert g1 == g2
    assert g1.action_probabilities is None
    assert g1.confidence is None
    assert g1.thought_tokens > 0
    assert g1.api_cost_usd > r1.api_cost_usd

    # Verify modality sensitivity across 100 synthetic episodes:
    # Text-present outperforms vision-only hard perturbations and legal_only has 0 illegal moves.
    legal_set = set(req_dm.legal_actions)
    opt_set = set(req_dm.metadata["optimal_actions"])
    text_opt = 0
    hard_vis_opt = 0
    hard_vis_illegal = 0
    legal_only_illegal = 0

    for i in range(100):
        ep = f"ep_{i:03d}"
        res_text = await dm_adapter.decide(
            _sample_board_and_request(
                row_id="B_jev_T",
                condition_code="B-DM-T",
                use_vision=False,
                use_text=True,
                render_variant=None,
                episode_id=ep,
            )
        )
        if res_text.requested_action in opt_set:
            text_opt += 1

        res_hard = await dm_adapter.decide(
            _sample_board_and_request(
                row_id="A_jev_V",
                condition_code="A-DM-V",
                use_vision=True,
                use_text=False,
                render_variant="tilt_blur",
                text_format=None,
                episode_id=ep,
            )
        )
        if res_hard.requested_action in opt_set:
            hard_vis_opt += 1
        if res_hard.requested_action not in legal_set:
            hard_vis_illegal += 1

        res_lo = await dm_adapter.decide(
            _sample_board_and_request(
                row_id="A_jev_V",
                condition_code="A-DM-V",
                use_vision=True,
                use_text=False,
                render_variant="dark_low_contrast",
                text_format=None,
                action_mode="legal_only",
                episode_id=ep,
            )
        )
        if res_lo.requested_action not in legal_set:
            legal_only_illegal += 1

    assert text_opt > hard_vis_opt
    assert hard_vis_illegal > 0
    assert legal_only_illegal == 0


async def test_retry_helper_exponential_backoff_and_zero_retry_on_malformed() -> None:
    assert is_transient_error(FakeHTTPStatusError(429)) is True
    assert is_transient_error(FakeHTTPStatusError(503)) is True
    assert is_transient_error(TimeoutError("timed out")) is True
    assert is_transient_error(FakeHTTPStatusError(400)) is False
    assert is_transient_error(json.JSONDecodeError("bad", "{", 0)) is False
    assert is_transient_error(BudgetExceededError(1.0, 1.0)) is False

    # 3 transient 429s followed by success on 4th attempt -> retries=3, backoff 1, 2, 4 * base.
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    attempts = 0

    async def flaky_call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            raise FakeHTTPStatusError(429, "rate limited")
        return "ok"

    outcome = await call_with_retry(
        flaky_call,
        max_retries=3,
        base_delay=1.0,
        jitter_fraction=0.0,
        sleep_fn=fake_sleep,
    )
    assert outcome.value == "ok"
    assert outcome.retries == 3
    assert sleeps == [1.0, 2.0, 4.0]
    assert outcome.end_to_end_latency_ms >= outcome.model_latency_ms >= 0.0

    # 4th failure exhausts max_retries=3 and re-raises.
    bad_attempts = 0

    async def always_503() -> str:
        nonlocal bad_attempts
        bad_attempts += 1
        raise FakeTypeSafeStatusError(503, "unavailable")

    with pytest.raises(FakeTypeSafeStatusError):
        await call_with_retry(
            always_503,
            max_retries=3,
            base_delay=0.001,
            jitter_fraction=0.0,
            sleep_fn=fake_sleep,
        )
    assert bad_attempts == 4

    # Malformed output / non-429 4xx -> zero retries.
    malformed_attempts = 0

    async def bad_json() -> str:
        nonlocal malformed_attempts
        malformed_attempts += 1
        raise json.JSONDecodeError("bad", "{", 1)

    with pytest.raises(json.JSONDecodeError):
        await call_with_retry(bad_json, max_retries=3, base_delay=0.001, sleep_fn=fake_sleep)
    assert malformed_attempts == 1


async def test_typesafe_jev_adapter_danger_zone_header_and_retry() -> None:
    root = find_repo_root()
    model_cfg = load_all_model_configs(root)["jev_1_14_0"]
    calls: list[dict[str, Any]] = []

    class FakeTypeSafeClient:
        def __init__(self) -> None:
            self.n = 0
            self.closed = False

        async def system_one(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            self.n += 1
            if self.n == 1:
                raise FakeHTTPStatusError(429, "too many requests")
            probs = {k: (0.84 if k == "R1C3" else 0.02) for k in CELL_KEYS}
            return SimpleNamespace(
                model="jev-1.14.0",
                answers={
                    "action": SimpleNamespace(
                        choice="R1C3",
                        probabilities=probs,
                        confidence=0.84,
                    )
                },
                usage=SimpleNamespace(input_tokens=400, output_tokens=0),
            )

        async def aclose(self) -> None:
            self.closed = True

    client = FakeTypeSafeClient()
    adapter = TypeSafeJevAdapter(
        model_cfg,
        client=client,
        max_retries=3,
        retry_base_delay=0.0,
    )

    # Vision + Text request MUST include TypeSafe-Danger-Zone-Images: "true" and both state keys.
    req_vt = _sample_board_and_request(row_id="D_jev_VT", use_vision=True, use_text=True)
    res_vt = await adapter.decide(req_vt)
    assert res_vt.response_valid is True
    assert res_vt.requested_action == "R1C3"
    assert res_vt.retries == 1
    assert res_vt.confidence == pytest.approx(0.84)
    assert res_vt.input_tokens == 400
    assert res_vt.api_cost_usd == pytest.approx((400 / 1_000_000) * 0.042)
    assert calls[-1]["extra_headers"] == {"TypeSafe-Danger-Zone-Images": "true"}
    assert "board_image" in calls[-1]["state"]
    assert isinstance(calls[-1]["state"]["game_state"], dict)
    assert calls[-1]["state"]["game_state"] == json.loads(req_vt.state_text or "")
    assert "R1" in calls[-1]["state"]["game_state"]["board"]
    assert "R1C1" in calls[-1]["state"]["game_state"]["board"]["R1"]
    assert calls[-1]["questions"]["action"].instructions == req_vt.objective_instructions

    # Text-only request MUST NOT send Danger-Zone-Images header and parses JSON into native dict.
    req_t = _sample_board_and_request(
        row_id="B_jev_T",
        condition_code="B-DM-T",
        use_vision=False,
        use_text=True,
        render_variant=None,
    )
    res_t = await adapter.decide(req_t)
    assert res_t.response_valid is True
    assert res_t.retries == 0
    assert calls[-1]["extra_headers"] is None
    assert list(calls[-1]["state"].keys()) == ["game_state"]
    assert isinstance(calls[-1]["state"]["game_state"], dict)
    assert calls[-1]["state"]["game_state"] == json.loads(req_t.state_text or "")

    # Non-JSON text formats (e.g., ascii_grid) pass through unchanged as str.
    req_ascii = _sample_board_and_request(
        row_id="B_jev_T",
        condition_code="B-DM-T",
        use_vision=False,
        use_text=True,
        render_variant=None,
        text_format="ascii_grid",
    )
    res_ascii = await adapter.decide(req_ascii)
    assert res_ascii.response_valid is True
    assert isinstance(calls[-1]["state"]["game_state"], str)
    assert calls[-1]["state"]["game_state"] == req_ascii.state_text

    await adapter.close()
    assert client.closed is True


def test_resolve_optimal_actions_from_structured_json_state() -> None:
    board = ["X", "X", ".", ".", "O", ".", ".", ".", "."]
    state_text = format_ttt_state(board, "O", text_format="json_canonical")
    req = PolicyRequest(
        episode_id="pos_struct_001",
        step_idx=0,
        row_id="B_jev_T",
        condition_code="B-DM-T",
        model_key="jev_1_14_0",
        objective_instructions="Choose the single best Tic-Tac-Toe move.",
        action_criteria=dict(ACTION_CRITERIA),
        frames=[],
        state_text=state_text,
        render_variant=None,
        text_format="json_canonical",
        action_mode="all_9",
        player_to_move=None,
        legal_actions=list(legal_action_keys(board)),
        metadata={},
    )
    assert _resolve_optimal_actions(req) == ["R1C3"]


async def test_gemini_decision_adapter_v1beta_and_safety_no_retry() -> None:
    root = find_repo_root()
    model_cfg = load_all_model_configs(root)["gemini_decision_1"]
    recorded_calls: list[dict[str, Any]] = []

    class FakeDecisionsAPI:
        def __init__(self, mode: str = "ok") -> None:
            self.mode = mode
            self.count = 0

        async def create(self, **kwargs: Any) -> Any:
            self.count += 1
            recorded_calls.append(kwargs)
            if self.mode == "safety":
                return SimpleNamespace(
                    answers=None,
                    prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
                    usage=None,
                )
            probs = {k: (0.76 if k == "R1C3" else 0.03) for k in CELL_KEYS}
            return SimpleNamespace(
                answers={
                    "action": SimpleNamespace(
                        choice="R1C3",
                        probabilities=probs,
                        confidence=0.76,
                    )
                },
                prompt_feedback=None,
                usage=SimpleNamespace(
                    total_input_tokens=500,
                    total_output_tokens=0,
                    total_thought_tokens=0,
                    total_tokens=500,
                ),
            )

    init_args: list[dict[str, Any]] = []

    def fake_genai_client(**kwargs: Any) -> Any:
        init_args.append(kwargs)
        return SimpleNamespace(
            aio=SimpleNamespace(decisions=FakeDecisionsAPI("ok"), aclose=None),
            close=None,
        )

    from google.genai import types as genai_types

    with patch("dm_bench.adapters.gemini_decision.genai.Client", side_effect=fake_genai_client):
        adapter = GeminiDecisionAdapter(model_cfg, api_key="test-key", retry_base_delay=0.0)

    http_opts = init_args[0]["http_options"]
    assert isinstance(http_opts, genai_types.HttpOptions)
    assert http_opts.api_version == "v1beta"
    assert http_opts.retry_options is not None
    assert http_opts.retry_options.attempts == 1
    req = _sample_board_and_request(
        row_id="D_gdec_VT",
        condition_code="D-DM-VT",
        model_key="gemini_decision_1",
        use_vision=True,
        use_text=True,
    )
    res = await adapter.decide(req)
    assert res.response_valid is True
    assert res.requested_action == "R1C3"
    assert res.confidence == pytest.approx(0.76)
    assert res.input_tokens == 500
    assert res.api_cost_usd == pytest.approx((500 / 1_000_000) * 0.10)

    blk = recorded_calls[0]["input"]
    assert blk[0]["type"] == "image"
    assert blk[0]["mime_type"] == "image/png"
    assert base64.b64decode(blk[0]["data"]) == req.frames[0].png_bytes
    assert blk[1] == {"type": "text", "text": req.state_text}
    assert recorded_calls[0]["questions"]["action"]["type"] == "choice"

    # Safety block must incur 0 retries.
    safety_api = FakeDecisionsAPI("safety")
    safety_client = SimpleNamespace(aio=SimpleNamespace(decisions=safety_api))
    safety_adapter = GeminiDecisionAdapter(model_cfg, client=safety_client, retry_base_delay=0.0)
    res_safety = await safety_adapter.decide(req)
    assert res_safety.response_valid is False
    assert res_safety.retries == 0
    assert res_safety.error_type == "SafetyBlockError"
    assert safety_api.count == 1


async def test_gemini_adapters_disable_sdk_internal_retries_and_track_429() -> None:
    import httpx
    from google.genai._interactions import APIError as GenAIInteractionsAPIError

    root = find_repo_root()
    cfgs = load_all_model_configs(root)

    # Real genai.Client instances must configure SDK-internal retries to max_retries == 0.
    dec_adapter = GeminiDecisionAdapter(cfgs["gemini_decision_1"], api_key="test-key")
    try:
        assert dec_adapter._client.aio.decisions.sdk_configuration.retry_config.max_retries == 0
    finally:
        await dec_adapter.close()

    gen_adapter = GeminiGenerativeAdapter(cfgs["gemini_4_hydrogen"], api_key="test-key")
    try:
        assert gen_adapter._client.aio.interactions.sdk_configuration.retry_config.max_retries == 0
    finally:
        await gen_adapter.close()

    # Transient 429 from google-genai must be retried by call_with_retry and logged in PolicyResult.retries.
    probs = {k: (0.76 if k == "R1C3" else 0.03) for k in CELL_KEYS}

    class FlakyDecisionsAPI:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                req_obj = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/decisions")
                resp_obj = httpx.Response(429, request=req_obj)
                raise GenAIInteractionsAPIError.generate(
                    status_code=429,
                    body={"error": {"message": "Quota exceeded"}},
                    message="Quota exceeded",
                    response=resp_obj,
                )
            return SimpleNamespace(
                answers={"action": SimpleNamespace(choice="R1C3", probabilities=probs, confidence=0.76)},
                prompt_feedback=None,
                usage=SimpleNamespace(
                    total_input_tokens=500,
                    total_output_tokens=0,
                    total_thought_tokens=0,
                    total_tokens=500,
                ),
            )

    flaky_api = FlakyDecisionsAPI()
    flaky_adapter = GeminiDecisionAdapter(
        cfgs["gemini_decision_1"],
        client=SimpleNamespace(aio=SimpleNamespace(decisions=flaky_api)),
        max_retries=2,
        retry_base_delay=0.0,
    )
    req = _sample_board_and_request(
        row_id="A_gdec_V",
        condition_code="A-DM-V",
        model_key="gemini_decision_1",
        use_vision=True,
        use_text=False,
        text_format=None,
    )
    res = await flaky_adapter.decide(req)
    assert res.response_valid is True
    assert res.retries == 1
    assert flaky_api.calls == 2


async def test_gemini_generative_adapter_v1_schema_billing_and_zero_retry_malformed() -> None:
    root = find_repo_root()
    model_cfg = load_all_model_configs(root)["gemini_4_hydrogen"]
    prompt_cfg = load_prompt_config(repo_root=root)
    captured: list[dict[str, Any]] = []

    class FakeInteractionsAPI:
        def __init__(self, output_text: str) -> None:
            self.output_text = output_text
            self.count = 0

        async def create(self, **kwargs: Any) -> Any:
            self.count += 1
            captured.append(kwargs)
            return SimpleNamespace(
                status="completed",
                output_text=self.output_text,
                errors=None,
                usage=SimpleNamespace(
                    total_input_tokens=600,
                    total_output_tokens=15,
                    total_thought_tokens=85,
                    total_tokens=700,  # 700 - 600 = 100 non-input tokens (no double-count)
                ),
            )

    init_args: list[dict[str, Any]] = []

    def fake_genai_client(**kwargs: Any) -> Any:
        init_args.append(kwargs)
        return SimpleNamespace(
            aio=SimpleNamespace(interactions=FakeInteractionsAPI('{"action": "R1C3"}')),
        )

    from google.genai import types as genai_types

    with patch("dm_bench.adapters.gemini_generative.genai.Client", side_effect=fake_genai_client):
        adapter = GeminiGenerativeAdapter(model_cfg, api_key="test-key", retry_base_delay=0.0)

    http_opts = init_args[0]["http_options"]
    assert isinstance(http_opts, genai_types.HttpOptions)
    assert http_opts.api_version == "v1"
    assert http_opts.retry_options is not None
    assert http_opts.retry_options.attempts == 1
    req = _sample_board_and_request(
        row_id="E_ghyd_VT",
        condition_code="E-VLM-VT",
        model_key="gemini_4_hydrogen",
        action_mode="legal_only",
    )
    req = req.model_copy(update={"objective_instructions": prompt_cfg.objective_instructions})
    res = await adapter.decide(req)

    assert res.response_valid is True
    assert res.requested_action == "R1C3"
    assert res.action_probabilities is None
    assert res.confidence is None
    assert res.input_tokens == 600
    assert res.output_tokens == 15
    assert res.thought_tokens == 85
    assert res.total_tokens == 700
    expected_cost = (600 / 1_000_000) * 1.50 + (100 / 1_000_000) * 7.50
    assert res.api_cost_usd == pytest.approx(expected_cost)

    call_kwargs = captured[0]
    assert call_kwargs["store"] is False
    assert call_kwargs["generation_config"] == {"thinking_level": "low"}
    for forbidden in ("temperature", "top_p", "top_k", "response_mime_type", "max_output_tokens"):
        assert forbidden not in call_kwargs
        assert forbidden not in call_kwargs["generation_config"]
    assert call_kwargs["system_instruction"] == prompt_cfg.build_generative_system_instruction(
        req.action_criteria
    )
    assert call_kwargs["response_format"]["schema"]["properties"]["action"]["enum"] == list(
        req.action_criteria.keys()
    )

    # Malformed JSON -> 0 retries, response_valid=False, still accounts for token cost.
    bad_api = FakeInteractionsAPI("{not valid json")
    bad_adapter = GeminiGenerativeAdapter(
        model_cfg,
        client=SimpleNamespace(aio=SimpleNamespace(interactions=bad_api)),
        retry_base_delay=0.0,
    )
    res_bad = await bad_adapter.decide(req)
    assert res_bad.response_valid is False
    assert res_bad.retries == 0
    assert res_bad.error_type == "MalformedResponseError"
    assert res_bad.api_cost_usd == pytest.approx(expected_cost)
    assert bad_api.count == 1

    # Action outside legal_only enum -> 0 retries, SchemaMismatchError.
    oob_api = FakeInteractionsAPI('{"action": "R1C1"}')
    oob_adapter = GeminiGenerativeAdapter(
        model_cfg,
        client=SimpleNamespace(aio=SimpleNamespace(interactions=oob_api)),
        retry_base_delay=0.0,
    )
    res_oob = await oob_adapter.decide(req)
    assert res_oob.response_valid is False
    assert res_oob.retries == 0
    assert res_oob.error_type == "SchemaMismatchError"
    assert oob_api.count == 1


async def test_factory_and_budget_tracker_enforcement() -> None:
    root = find_repo_root()
    tracker = BudgetTracker(max_cost_usd=0.0005)
    adapters = build_row_adapters(repo_root=root, mock=True, budget_tracker=tracker)
    assert len(adapters) == 9
    assert all(isinstance(entry[2], BudgetedPolicyAdapter) for entry in adapters.values())

    req = _sample_board_and_request(
        row_id="E_ghyd_VT",
        condition_code="E-VLM-VT",
        model_key="gemini_4_hydrogen",
    )
    _, _, ghyd_adapter = adapters["E_ghyd_VT"]
    first_res = await ghyd_adapter.decide(req)
    assert first_res.response_valid is True
    assert tracker.spent_usd == pytest.approx(first_res.api_cost_usd)
    assert tracker.spent_usd >= 0.0005

    with pytest.raises(BudgetExceededError):
        await ghyd_adapter.decide(req)

    await close_adapters(adapters)


def _prepare_temp_repo(tmp_path: Path) -> Path:
    src_root = find_repo_root()
    (tmp_path / "pyproject.toml").write_text((src_root / "pyproject.toml").read_text(), encoding="utf-8")
    shutil.copytree(src_root / "experiments" / "configs", tmp_path / "experiments" / "configs")
    shutil.copytree(src_root / "experiments" / "prompts", tmp_path / "experiments" / "prompts")
    dev_dir = tmp_path / "experiments" / "datasets" / "dev"
    dev_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        src_root / "experiments" / "datasets" / "dev" / "ttt_dev_50.jsonl",
        dev_dir / "ttt_dev_50.jsonl",
    )
    return tmp_path


async def test_run_smoke_mock_and_budget_halt(tmp_path: Path) -> None:
    temp_root = _prepare_temp_repo(tmp_path)

    summary = await run_smoke(
        repo_root=temp_root,
        mock=True,
        latency_trials=10,
        max_cost_usd=1.0,
    )
    assert summary["status"] == "completed"
    assert summary["all_contracts_passed"] is True
    assert summary["total_decisions"] == 9 + 3 * 10
    assert len(summary["contract_checks"]) == 9
    assert set(summary["latency_by_model"].keys()) == {
        "jev_1_14_0",
        "gemini_decision_1",
        "gemini_4_hydrogen",
    }

    run_dir = Path(summary["run_dir"])
    assert (run_dir / "decisions.jsonl").is_file()
    assert (run_dir / "latency_summary.json").is_file()
    assert (run_dir / "manifest.yaml").is_file()

    rows = [DecisionLogRow.model_validate(item) for item in read_jsonl(run_dir / "decisions.jsonl")]
    assert len(rows) == 39
    assert all(r.response_valid for r in rows)

    # Zero-budget run halts cleanly before completing all calls.
    halted_summary = await run_smoke(
        repo_root=temp_root,
        mock=True,
        latency_trials=2,
        max_cost_usd=0.0,
    )
    assert halted_summary["status"] == "halted_budget_cap"
    assert halted_summary["total_decisions"] == 0


def test_extract_retry_after_and_classify_error() -> None:
    class _RateLimitError(Exception):
        status_code = 429

        def __init__(self, msg: str, headers: dict[str, str] | None = None) -> None:
            super().__init__(msg)
            self.response = SimpleNamespace(status_code=429, headers=headers or {})

    class _BadRequestError(Exception):
        status_code = 400

    rl_msg = _RateLimitError("429 RESOURCE_EXHAUSTED. Please retry in 14.5s.")
    assert is_rate_limit_error(rl_msg) is True
    assert extract_retry_after_seconds(rl_msg) == pytest.approx(14.5)
    assert classify_error(rl_msg) == "rate_limit"

    rl_hdr = _RateLimitError("rate limited", headers={"Retry-After": "9"})
    assert extract_retry_after_seconds(rl_hdr) == pytest.approx(9.0)

    bad_req = _BadRequestError("INVALID_ARGUMENT")
    assert is_rate_limit_error(bad_req) is False
    assert classify_error(bad_req) == "api_error"
    assert classify_error(MalformedResponseError("bad json")) == "model_output"
    assert classify_error(SchemaMismatchError("wrong enum")) == "model_output"
    assert classify_error(SafetyBlockError("blocked")) == "model_output"

    assert infer_error_category("RateLimitError", "429 RESOURCE_EXHAUSTED") == "rate_limit"
    assert infer_error_category("BadRequestError", "400") == "api_error"
    assert infer_error_category("SchemaParseError", "bad schema") == "model_output"
    assert infer_error_category("MalformedOutputError", "bad") == "model_output"
    assert is_infra_error("RateLimitError", "429") is True
    assert is_infra_error("BadRequestError", "400") is True
    assert is_infra_error("MalformedOutputError", "bad") is False


async def test_retry_honors_retry_after_and_rate_limit_retries() -> None:
    class _RateLimit429(Exception):
        status_code = 429

    attempts = 0
    sleeps: list[float] = []
    rl_waits: list[float] = []

    async def _sleep(secs: float) -> None:
        sleeps.append(secs)

    async def _fn() -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise _RateLimit429("Resource exhausted. Please retry in 12s.")
        return "ok"

    outcome = await call_with_retry(
        _fn,
        max_retries=0,
        rate_limit_max_retries=2,
        base_delay=1.0,
        sleep_fn=_sleep,
        on_rate_limit=rl_waits.append,
    )

    assert outcome.value == "ok"
    assert outcome.retries == 2
    assert attempts == 3
    assert len(sleeps) == 2
    assert sleeps[0] >= 12.0
    assert len(rl_waits) == 2


async def test_gemini_decision_single_option_short_circuit() -> None:
    root = find_repo_root()
    models = load_all_model_configs(root)
    gdec_cfg = models["gemini_decision_1"]

    class _FailIfCalledClient:
        class _Aio:
            class _Decisions:
                async def create(self, **kwargs: Any) -> Any:
                    raise AssertionError("decisions.create should not be called for 1 option")

            decisions = _Decisions()

        aio = _Aio()

    adapter = GeminiDecisionAdapter(
        gdec_cfg,
        client=_FailIfCalledClient(),
        retry_base_delay=0.0,
    )
    req = _sample_board_and_request(
        row_id="A_gdec_V",
        condition_code="A-DM-V",
        model_key="gemini_decision_1",
    ).model_copy(
        update={
            "action_mode": "legal_only",
            "action_criteria": {"R2C2": "Center cell R2C2"},
            "legal_actions": ["R2C2"],
        }
    )
    res = await adapter.decide(req)
    assert res.response_valid is True
    assert res.requested_action == "R2C2"
    assert res.action_probabilities == {"R2C2": 1.0}
    assert res.confidence == pytest.approx(1.0)
    assert res.api_cost_usd == 0.0
    assert res.model_latency_ms == 0.0
    await adapter.close()
