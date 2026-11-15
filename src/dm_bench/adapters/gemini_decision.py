"""Google Gemini Decision 1 adapter (`gemini-decision-1` via `google-genai` v1beta)."""

from __future__ import annotations

import asyncio
import base64
import inspect
import time
from typing import Any

from google import genai
from google.genai import types

from dm_bench.adapters.base import DecisionModelAdapter, compute_request_payload_hash
from dm_bench.adapters.retry import (
    MalformedResponseError,
    RetryStats,
    SafetyBlockError,
    SchemaMismatchError,
    call_with_retry,
    classify_error,
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


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class GeminiDecisionAdapter(DecisionModelAdapter):
    """Adapter for `gemini-decision-1` using `client.aio.decisions.create` on `v1beta`."""

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        max_retries: int = 3,
        rate_limit_max_retries: int | None = None,
        retry_base_delay: float = 1.0,
    ) -> None:
        super().__init__(model_config)
        self._closed = False
        self._max_retries = max_retries
        self._rate_limit_max_retries = (
            rate_limit_max_retries
            if rate_limit_max_retries is not None
            else (max(max_retries, 6) if retry_base_delay > 0 else max_retries)
        )
        self._retry_base_delay = retry_base_delay
        self._rate_limit_until = 0.0
        if client is not None:
            self._client = client
        else:
            if not api_key:
                raise ValueError(
                    f"API key is required for {model_config.model_key} ({model_config.api_env_var})"
                )
            api_version = model_config.api_version or "v1beta"
            # Disable SDK-internal HTTP retries so call_with_retry tracks retries and separates backoff from model_latency_ms.
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    api_version=api_version,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )

    def _build_input_blocks(self, request: PolicyRequest) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for frame in request.frames:
            blocks.append(
                {
                    "type": "image",
                    "data": base64.b64encode(frame.png_bytes).decode("ascii"),
                    "mime_type": frame.mime_type or "image/png",
                }
            )
        if request.state_text is not None:
            blocks.append({"type": "text", "text": request.state_text})
        return blocks

    async def decide(self, request: PolicyRequest) -> PolicyResult:
        if not request.frames and request.state_text is None:
            raise ValueError("PolicyRequest must include at least one frame or state_text")
        if not request.action_criteria:
            raise ValueError("PolicyRequest.action_criteria must be non-empty")

        input_blocks = self._build_input_blocks(request)
        questions = {
            "action": {
                "type": "choice",
                "instructions": request.objective_instructions,
                "criteria": dict(request.action_criteria),
            }
        }
        payload_hash = compute_request_payload_hash(
            self.model_config.model_id,
            request,
            extra={"api_version": self.model_config.api_version or "v1beta"},
        )

        # Gemini CreateDecisionRequest requires 2..255 choice criteria; a 1-option legal_only
        # position (e.g. final ply of Tic-Tac-Toe) is a forced single move.
        if len(request.action_criteria) == 1:
            only_key = next(iter(request.action_criteria.keys()))
            return PolicyResult(
                requested_action=only_key,
                action_probabilities={only_key: 1.0},
                confidence=1.0,
                response_valid=True,
                retries=0,
                model_latency_ms=0.0,
                end_to_end_latency_ms=0.0,
                raw_response={"forced_single_choice": only_key},
                request_payload_hash=payload_hash,
                model_id=self.model_config.model_id,
                provider=self.model_config.provider,
            )

        def _note_rate_limit(wait_seconds: float) -> None:
            if self._retry_base_delay > 0:
                self._rate_limit_until = max(self._rate_limit_until, time.monotonic() + wait_seconds)

        if self._retry_base_delay > 0:
            cooldown_left = self._rate_limit_until - time.monotonic()
            if cooldown_left > 0:
                await asyncio.sleep(cooldown_left)

        stats = RetryStats()
        response: Any = None
        try:
            outcome = await call_with_retry(
                lambda: self._client.aio.decisions.create(
                    model=self.model_config.model_id,
                    input=input_blocks,
                    questions=questions,
                ),
                max_retries=self._max_retries,
                rate_limit_max_retries=self._rate_limit_max_retries,
                base_delay=self._retry_base_delay,
                stats=stats,
                on_rate_limit=_note_rate_limit,
            )
            response = outcome.value
            return self._parse_response(
                request,
                response,
                outcome.retries,
                outcome.model_latency_ms,
                outcome.end_to_end_latency_ms,
                payload_hash,
            )
        except Exception as exc:
            return PolicyResult(
                requested_action=None,
                action_probabilities=None,
                confidence=None,
                response_valid=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
                error_category=classify_error(exc),
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
        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = _get_field(prompt_feedback, "block_reason") if prompt_feedback is not None else None

        answers = getattr(response, "answers", None)
        if not isinstance(answers, dict) or "action" not in answers:
            if block_reason is not None:
                raise SafetyBlockError(f"Gemini Decision blocked by prompt feedback: {block_reason}")
            raise SchemaMismatchError("Gemini Decision response missing answers['action']")

        ans = answers["action"]
        choice = _get_field(ans, "choice")
        raw_probs = _get_field(ans, "probabilities")
        raw_conf = _get_field(ans, "confidence")

        if not isinstance(choice, str) or not choice:
            raise MalformedResponseError(f"Invalid choice value in Gemini Decision response: {choice!r}")
        if not isinstance(raw_probs, dict) or not raw_probs:
            raise SchemaMismatchError("Gemini Decision ChoiceAnswer missing probabilities mapping")
        if raw_conf is None:
            raise SchemaMismatchError("Gemini Decision ChoiceAnswer missing confidence")

        probabilities = {str(k): float(v) for k, v in raw_probs.items()}
        confidence = float(raw_conf)

        usage = getattr(response, "usage", None)
        input_tokens = int((_get_field(usage, "total_input_tokens", 0) or 0)) if usage is not None else 0
        output_tokens = int((_get_field(usage, "total_output_tokens", 0) or 0)) if usage is not None else 0
        thought_tokens = int((_get_field(usage, "total_thought_tokens", 0) or 0)) if usage is not None else 0
        total_tokens = int((_get_field(usage, "total_tokens", 0) or 0)) if usage is not None else 0
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens + thought_tokens

        cost = self.model_config.pricing.estimate_cost_usd(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thought_tokens=thought_tokens,
            total_tokens=total_tokens,
        )

        if choice not in request.action_criteria:
            raise SchemaMismatchError(
                f"Gemini Decision choice {choice!r} not in requested criteria {list(request.action_criteria.keys())}"
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
            thought_tokens=thought_tokens,
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
        aio = getattr(self._client, "aio", None)
        if aio is not None:
            aclose_fn = getattr(aio, "aclose", None)
            if callable(aclose_fn):
                res = aclose_fn()
                if inspect.isawaitable(res):
                    await res
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()
