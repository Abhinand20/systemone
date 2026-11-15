"""Base PolicyAdapter abstractions, shared response helpers, and deterministic mock adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
import random
from typing import Any

from dm_bench.config import ModelConfig, PromptConfig
from dm_bench.contracts import PolicyRequest, PolicyResult
from dm_bench.games.tic_tac_toe.engine import CELL_KEYS, is_terminal
from dm_bench.games.tic_tac_toe.oracle import analyze_position
from dm_bench.logging_io import sha256_bytes, sha256_json


class PolicyAdapter(ABC):
    """Abstract interface implemented by every live and mock model-condition adapter."""

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config

    @abstractmethod
    async def decide(self, request: PolicyRequest) -> PolicyResult:
        """Select an action for `request` and return a standardized `PolicyResult`."""

    async def close(self) -> None:
        """Release any underlying HTTP/SDK client resources."""
        return None


class DecisionModelAdapter(PolicyAdapter, ABC):
    """Base class for Decision Model adapters returning calibrated choice distributions."""


class GenerativeAdapter(PolicyAdapter, ABC):
    """Base class for generative VLM/LLM adapters returning structured JSON actions."""

    def build_system_instruction(self, request: PolicyRequest) -> str:
        """Build prompt-parity system instruction from request instructions & criteria."""
        prompt_cfg = PromptConfig(
            prompt_version="runtime",
            objective_instructions=request.objective_instructions,
            action_criteria=dict(request.action_criteria),
        )
        return prompt_cfg.build_generative_system_instruction(request.action_criteria)


def compute_request_payload_hash(
    model_id: str,
    request: PolicyRequest,
    extra: dict[str, Any] | None = None,
) -> str:
    """Compute a deterministic SHA-256 hash summarizing a PolicyRequest payload."""
    frame_digests = [
        frame.sha256 or sha256_bytes(frame.png_bytes)
        for frame in request.frames
    ]
    payload: dict[str, Any] = {
        "model_id": model_id,
        "row_id": request.row_id,
        "condition_code": request.condition_code,
        "objective_instructions": request.objective_instructions,
        "action_criteria": list(request.action_criteria.items()),
        "frame_hashes": frame_digests,
        "state_text": request.state_text,
        "action_mode": request.action_mode,
    }
    if extra:
        payload["extra"] = extra
    return sha256_json(payload)


def _deterministic_rng(request: PolicyRequest) -> random.Random:
    """Create a deterministic per-decision RNG seeded from key request attributes."""
    seed_str = (
        f"{request.row_id}|{request.episode_id}|{request.render_variant}|"
        f"{request.text_format}|{request.action_mode}|{request.step_idx}"
    )
    seed_int = int.from_bytes(hashlib.sha256(seed_str.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed_int)


def _normalize_board_cells(raw_board: Any) -> list[str] | None:
    """Normalize a 9-element list or nested/flat R1C1..R3C3 board dict into 9-char cells."""
    if isinstance(raw_board, (list, tuple)) and len(raw_board) == 9:
        return ["." if str(c) == "empty" else str(c) for c in raw_board]
    if isinstance(raw_board, dict):
        flat: dict[str, Any] = {}
        for k, v in raw_board.items():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[str(k)] = v
        if all(ck in flat for ck in CELL_KEYS):
            return ["." if str(flat[ck]) == "empty" else str(flat[ck]) for ck in CELL_KEYS]
    return None


def _resolve_optimal_actions(request: PolicyRequest) -> list[str]:
    """Resolve minimax-optimal actions from request metadata or JSON state_text if available."""
    meta_opt = request.metadata.get("optimal_actions")
    if isinstance(meta_opt, (list, tuple)) and meta_opt:
        return [str(x) for x in meta_opt]

    board = request.metadata.get("board")
    player = request.player_to_move or request.metadata.get("player_to_move")
    if (board is None or player is None) and request.state_text:
        text = request.state_text.strip()
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                board = board or parsed.get("board")
                player = player or parsed.get("player_to_move")

    norm_board = _normalize_board_cells(board)
    if (
        norm_board is not None
        and all(c in ("X", "O", ".") for c in norm_board)
        and player in ("X", "O")
        and not is_terminal(norm_board)
    ):
        try:
            analysis = analyze_position(norm_board, player)
            return list(analysis.optimal_actions)
        except Exception:
            pass

    if request.legal_actions:
        return list(request.legal_actions)
    return list(request.action_criteria.keys())


def _mock_rates(
    model_key: str,
    has_vision: bool,
    has_text: bool,
    render_variant: str | None,
    text_format: str | None,
    action_mode: str,
    is_generative: bool,
    is_smoke: bool,
) -> tuple[float, float, float]:
    """Return `(optimal_prob, illegal_prob, invalid_prob)` for deterministic mock synthesis."""
    if has_text:
        opt = 0.94 if has_vision else 0.91
        ill = 0.01 if has_vision else 0.02
        inv = 0.0
        if text_format == "ascii_grid":
            opt -= 0.02
        elif text_format == "coord_list":
            opt -= 0.04
            ill += 0.01
        elif text_format == "json_with_history":
            opt += 0.01

        if model_key == "gemini_4_hydrogen":
            opt += 0.02
        elif model_key == "gemini_decision_1":
            opt -= 0.01
    else:
        variant = render_variant or "clean_default"
        variant_profiles: dict[str, tuple[float, float, float]] = {
            "clean_default": (0.76, 0.06, 0.0),
            "labeled_margins": (0.81, 0.04, 0.0),
            "thin_alt_font": (0.66, 0.10, 0.01 if is_generative else 0.0),
            "dark_low_contrast": (0.50, 0.16, 0.03 if is_generative else 0.0),
            "tilt_blur": (0.46, 0.19, 0.04 if is_generative else 0.0),
        }
        opt, ill, inv = variant_profiles.get(variant, (0.74, 0.07, 0.0))
        if model_key == "jev_1_14_0":
            opt += 0.02
        elif model_key == "gemini_decision_1":
            opt -= 0.01
        elif model_key == "gemini_4_hydrogen":
            if variant in ("clean_default", "labeled_margins"):
                opt += 0.03
            else:
                opt -= 0.02

    if action_mode == "legal_only":
        ill = 0.0
        opt += 0.03

    if is_smoke:
        inv = 0.0

    opt = min(0.98, max(0.05, opt))
    ill = min(0.45, max(0.0, ill))
    inv = min(0.20, max(0.0, inv))
    return opt, ill, inv


def _estimate_mock_tokens(
    request: PolicyRequest,
    rng: random.Random,
    *,
    is_generative: bool,
) -> tuple[int, int, int, int]:
    """Return realistic `(input_tokens, output_tokens, thought_tokens, total_tokens)`."""
    criteria_tokens = 12 * len(request.action_criteria) + 45
    vision_tokens = 258 * len(request.frames)
    text_tokens = max(24, len(request.state_text) // 3) if request.state_text else 0
    jitter = rng.randint(0, 12)
    input_tokens = criteria_tokens + vision_tokens + text_tokens + jitter

    if is_generative:
        output_tokens = rng.randint(12, 18)
        thought_tokens = rng.randint(45, 115)
        total_tokens = input_tokens + output_tokens + thought_tokens
    else:
        output_tokens = 0
        thought_tokens = 0
        total_tokens = input_tokens

    return input_tokens, output_tokens, thought_tokens, total_tokens


class MockDecisionAdapter(DecisionModelAdapter):
    """Deterministic modality-aware mock adapter for Decision Models."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)

    async def decide(self, request: PolicyRequest) -> PolicyResult:
        if not request.frames and request.state_text is None:
            raise ValueError("PolicyRequest must include at least one frame or state_text")
        if not request.action_criteria:
            raise ValueError("PolicyRequest.action_criteria must be non-empty")

        rng = _deterministic_rng(request)
        criteria_keys = list(request.action_criteria.keys())
        legal_set = set(request.legal_actions) if request.legal_actions else set(criteria_keys)
        legal_keys = [k for k in criteria_keys if k in legal_set] or list(criteria_keys)
        illegal_keys = [k for k in criteria_keys if k not in legal_set]

        raw_optimal = _resolve_optimal_actions(request)
        optimal_keys = [k for k in raw_optimal if k in legal_keys] or list(legal_keys)
        optimal_set = set(optimal_keys)
        suboptimal_legal = [k for k in legal_keys if k not in optimal_set]

        has_vision = bool(request.frames)
        has_text = request.state_text is not None
        is_smoke = bool(request.metadata.get("smoke_test"))

        opt_p, ill_p, _ = _mock_rates(
            self.model_config.model_key,
            has_vision,
            has_text,
            request.render_variant,
            request.text_format,
            request.action_mode,
            is_generative=False,
            is_smoke=is_smoke,
        )

        draw = rng.random()
        if illegal_keys and draw < ill_p:
            chosen = rng.choice(illegal_keys)
            outcome_kind = "illegal"
        elif draw < ill_p + opt_p or not suboptimal_legal:
            chosen = rng.choice(optimal_keys)
            outcome_kind = "optimal"
        else:
            chosen = rng.choice(suboptimal_legal)
            outcome_kind = "suboptimal"

        # Build calibrated probability distribution summing to 1.0 with argmax == chosen.
        weights: dict[str, float] = {}
        if outcome_kind == "optimal":
            chosen_base = rng.uniform(0.58, 0.90)
        else:
            chosen_base = rng.uniform(0.38, 0.62)

        for key in criteria_keys:
            if key == chosen:
                weights[key] = chosen_base
            elif key in optimal_set:
                weights[key] = rng.uniform(0.06, 0.16)
            elif key in legal_set:
                weights[key] = rng.uniform(0.01, 0.05)
            else:
                # Slightly higher illegal probability leakage on hard vision perturbations.
                leak_hi = (
                    0.035
                    if (not has_text and request.render_variant in ("dark_low_contrast", "tilt_blur"))
                    else 0.012
                )
                weights[key] = rng.uniform(0.001, leak_hi)

        # Ensure `chosen` remains strictly the argmax before normalization.
        max_other = max((w for k, w in weights.items() if k != chosen), default=0.0)
        if weights[chosen] <= max_other:
            weights[chosen] = max_other + 0.15

        total_w = sum(weights.values())
        probs = {k: round(v / total_w, 6) for k, v in weights.items()}
        drift = round(1.0 - sum(probs.values()), 6)
        probs[chosen] = round(probs[chosen] + drift, 6)
        confidence = float(probs[chosen])

        in_tok, out_tok, th_tok, tot_tok = _estimate_mock_tokens(
            request, rng, is_generative=False
        )
        cost = self.model_config.pricing.estimate_cost_usd(
            input_tokens=in_tok,
            output_tokens=out_tok,
            thought_tokens=th_tok,
            total_tokens=tot_tok,
        )

        base_lat = 105.0 if self.model_config.model_key == "jev_1_14_0" else 165.0
        model_lat = round(base_lat + rng.uniform(8.0, 55.0), 3)
        e2e_lat = round(model_lat + rng.uniform(1.0, 4.5), 3)

        return PolicyResult(
            requested_action=chosen,
            action_probabilities=probs,
            confidence=confidence,
            response_valid=True,
            retries=0,
            model_latency_ms=model_lat,
            end_to_end_latency_ms=e2e_lat,
            input_tokens=in_tok,
            output_tokens=out_tok,
            thought_tokens=th_tok,
            total_tokens=tot_tok,
            api_cost_usd=cost,
            raw_response={"choice": chosen, "confidence": confidence, "probabilities": probs},
            request_payload_hash=compute_request_payload_hash(
                self.model_config.model_id, request, extra={"mock": True}
            ),
            model_id=self.model_config.model_id,
            provider=self.model_config.provider,
        )


class MockGenerativeAdapter(GenerativeAdapter):
    """Deterministic modality-aware mock adapter for Generative VLM/LLM models."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)

    async def decide(self, request: PolicyRequest) -> PolicyResult:
        if not request.frames and request.state_text is None:
            raise ValueError("PolicyRequest must include at least one frame or state_text")
        if not request.action_criteria:
            raise ValueError("PolicyRequest.action_criteria must be non-empty")

        rng = _deterministic_rng(request)
        criteria_keys = list(request.action_criteria.keys())
        legal_set = set(request.legal_actions) if request.legal_actions else set(criteria_keys)
        legal_keys = [k for k in criteria_keys if k in legal_set] or list(criteria_keys)
        illegal_keys = [k for k in criteria_keys if k not in legal_set]

        raw_optimal = _resolve_optimal_actions(request)
        optimal_keys = [k for k in raw_optimal if k in legal_keys] or list(legal_keys)
        optimal_set = set(optimal_keys)
        suboptimal_legal = [k for k in legal_keys if k not in optimal_set]

        has_vision = bool(request.frames)
        has_text = request.state_text is not None
        is_smoke = bool(request.metadata.get("smoke_test"))

        opt_p, ill_p, inv_p = _mock_rates(
            self.model_config.model_key,
            has_vision,
            has_text,
            request.render_variant,
            request.text_format,
            request.action_mode,
            is_generative=True,
            is_smoke=is_smoke,
        )

        in_tok, out_tok, th_tok, tot_tok = _estimate_mock_tokens(
            request, rng, is_generative=True
        )
        cost = self.model_config.pricing.estimate_cost_usd(
            input_tokens=in_tok,
            output_tokens=out_tok,
            thought_tokens=th_tok,
            total_tokens=tot_tok,
        )
        model_lat = round(310.0 + rng.uniform(25.0, 140.0), 3)
        e2e_lat = round(model_lat + rng.uniform(1.5, 5.5), 3)
        payload_hash = compute_request_payload_hash(
            self.model_config.model_id, request, extra={"mock": True}
        )

        draw = rng.random()
        if draw < inv_p:
            return PolicyResult(
                requested_action=None,
                action_probabilities=None,
                confidence=None,
                response_valid=False,
                error_type="MalformedResponseError",
                error_message="Simulated malformed JSON output on hard visual perturbation",
                error_category="model_output",
                retries=0,
                model_latency_ms=model_lat,
                end_to_end_latency_ms=e2e_lat,
                input_tokens=in_tok,
                output_tokens=out_tok,
                thought_tokens=th_tok,
                total_tokens=tot_tok,
                api_cost_usd=cost,
                raw_response="{invalid_json",
                request_payload_hash=payload_hash,
                model_id=self.model_config.model_id,
                provider=self.model_config.provider,
            )

        adj_draw = rng.random()
        if illegal_keys and adj_draw < ill_p:
            chosen = rng.choice(illegal_keys)
        elif adj_draw < ill_p + opt_p or not suboptimal_legal:
            chosen = rng.choice(optimal_keys)
        else:
            chosen = rng.choice(suboptimal_legal)

        return PolicyResult(
            requested_action=chosen,
            action_probabilities=None,
            confidence=None,
            response_valid=True,
            retries=0,
            model_latency_ms=model_lat,
            end_to_end_latency_ms=e2e_lat,
            input_tokens=in_tok,
            output_tokens=out_tok,
            thought_tokens=th_tok,
            total_tokens=tot_tok,
            api_cost_usd=cost,
            raw_response={"action": chosen},
            request_payload_hash=payload_hash,
            model_id=self.model_config.model_id,
            provider=self.model_config.provider,
        )
