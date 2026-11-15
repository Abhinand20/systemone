"""TypeSafe Jev Decision Model adapter (`jev-1.14.0` via `typesafe_sdk`)."""

from __future__ import annotations

import inspect
import json
from typing import Any

from typesafe_sdk import AsyncTypeSafeClient, Choice, Image, RetryPolicy

from dm_bench.adapters.base import DecisionModelAdapter, compute_request_payload_hash
from dm_bench.adapters.retry import (
    MalformedResponseError,
    RetryStats,
    SchemaMismatchError,
    call_with_retry,
)
from dm_bench.config import ModelConfig
from dm_bench.contracts import PolicyRequest, PolicyResult


def _to_serializable(obj: Any) -> dict[str, Any] | str | None:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return str(obj)


def _parse_state_text(state_text: str) -> Any:
    """Parse JSON object/array state_text into native Content for TypeSafe; keep plain text as str."""
    stripped = state_text.strip()
    if stripped and stripped[0] in ("{", "["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return state_text
        if isinstance(parsed, (dict, list)):
            return parsed
    return state_text


class TypeSafeJevAdapter(DecisionModelAdapter):
    """Adapter for TypeSafe `jev-1.14.0` using `AsyncTypeSafeClient.system_one`."""

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        super().__init__(model_config)
        self._owns_client = client is None
        self._closed = False
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        if client is not None:
            self._client = client
        else:
            if not api_key:
                raise ValueError(
                    f"API key is required for {model_config.model_key} ({model_config.api_env_var})"
                )
            self._client = AsyncTypeSafeClient(
                api_key=api_key,
                retry=RetryPolicy(max_retries=0),
            )

    async def decide(self, request: PolicyRequest) -> PolicyResult:
        if not request.frames and request.state_text is None:
            raise ValueError("PolicyRequest must include at least one frame or state_text")
        if not request.action_criteria:
            raise ValueError("PolicyRequest.action_criteria must be non-empty")

        state: dict[str, Any] = {}
        if request.frames:
            state["board_image"] = Image(request.frames[0].png_bytes)
            for idx, extra_frame in enumerate(request.frames[1:], start=2):
                key = extra_frame.frame_id if extra_frame.frame_id != "board_image" else f"board_image_{idx}"
                state[key] = Image(extra_frame.png_bytes)
        if request.state_text is not None:
            state["game_state"] = _parse_state_text(request.state_text)

        extra_headers: dict[str, str] | None = None
        if request.frames:
            extra_headers = dict(self.model_config.extra_headers_on_images)
            extra_headers["TypeSafe-Danger-Zone-Images"] = "true"

        questions = {
            "action": Choice(
                instructions=request.objective_instructions,
                criteria=dict(request.action_criteria),
            )
        }
        payload_hash = compute_request_payload_hash(
            self.model_config.model_id,
            request,
            extra={
                "danger_zone_images": bool(request.frames),
                "structured_game_state": isinstance(state.get("game_state"), (dict, list)),
            },
        )

        stats = RetryStats()
        response: Any = None
        try:
            outcome = await call_with_retry(
                lambda: self._client.system_one(
                    state=state,
                    questions=questions,
                    model=self.model_config.model_id,
                    extra_headers=extra_headers,
                ),
                max_retries=self._max_retries,
                base_delay=self._retry_base_delay,
                stats=stats,
            )
            response = outcome.value
            return self._parse_response(request, response, outcome.retries, outcome.model_latency_ms, outcome.end_to_end_latency_ms, payload_hash)
        except Exception as exc:
            return PolicyResult(
                requested_action=None,
                action_probabilities=None,
                confidence=None,
                response_valid=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
                retries=stats.retries,
                model_latency_ms=stats.model_latency_ms,
                end_to_end_latency_ms=stats.end_to_end_latency_ms,
                raw_response=_to_serializable(response),
                request_payload_hash=payload_hash,
                model_id=self.model_config.model_id,
                provider=self.model_config.provider,
            )

    def _parse_response(
        self,
        request: PolicyRequest,
        response: Any,
        retries: int,
        model_latency_ms: float,
        end_to_end_latency_ms: float,
        payload_hash: str,
    ) -> PolicyResult:
        answers = getattr(response, "answers", None)
        if not isinstance(answers, dict) or "action" not in answers:
            raise SchemaMismatchError("TypeSafe response missing answers['action']")

        ans = answers["action"]
        choice = ans.get("choice") if isinstance(ans, dict) else getattr(ans, "choice", None)
        raw_probs = ans.get("probabilities") if isinstance(ans, dict) else getattr(ans, "probabilities", None)
        raw_conf = ans.get("confidence") if isinstance(ans, dict) else getattr(ans, "confidence", None)

        if not isinstance(choice, str) or not choice:
            raise MalformedResponseError(f"Invalid choice value in TypeSafe response: {choice!r}")
        if not isinstance(raw_probs, dict) or not raw_probs:
            raise SchemaMismatchError("TypeSafe ChoiceAnswer missing probabilities mapping")
        if raw_conf is None:
            raise SchemaMismatchError("TypeSafe ChoiceAnswer missing confidence")

        probabilities = {str(k): float(v) for k, v in raw_probs.items()}
        confidence = float(raw_conf)

        usage = getattr(response, "usage", None)
        input_tokens = int((usage.get("input_tokens") if isinstance(usage, dict) else getattr(usage, "input_tokens", 0)) or 0) if usage is not None else 0
        output_tokens = int((usage.get("output_tokens") if isinstance(usage, dict) else getattr(usage, "output_tokens", 0)) or 0) if usage is not None else 0
        total_tokens = input_tokens + output_tokens

        cost = self.model_config.pricing.estimate_cost_usd(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thought_tokens=0,
            total_tokens=total_tokens,
        )

        # Validate choice is inside requested action criteria (if not, contract mismatch).
        if choice not in request.action_criteria:
            raise SchemaMismatchError(
                f"TypeSafe choice {choice!r} not in requested criteria {list(request.action_criteria.keys())}"
            )

        return PolicyResult(
            requested_action=choice,
            action_probabilities=probabilities,
            confidence=confidence,
            response_valid=True,
            retries=retries,
            model_latency_ms=model_latency_ms,
            end_to_end_latency_ms=end_to_end_latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thought_tokens=0,
            total_tokens=total_tokens,
            api_cost_usd=cost,
            raw_response=_to_serializable(response),
            request_payload_hash=payload_hash,
            model_id=self.model_config.model_id,
            provider=self.model_config.provider,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for method_name in ("aclose", "close"):
            fn = getattr(self._client, method_name, None)
            if callable(fn):
                res = fn()
                if inspect.isawaitable(res):
                    await res
                break
