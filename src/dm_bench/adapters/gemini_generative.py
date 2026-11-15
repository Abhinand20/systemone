"""Google Gemini 4 Hydrogen adapter (`gemini-4-hydrogen` via `google-genai` v1 Interactions API)."""

from __future__ import annotations

import base64
import inspect
import json
from typing import Any

from google import genai
from google.genai import types

from dm_bench.adapters.base import GenerativeAdapter, compute_request_payload_hash
from dm_bench.adapters.retry import (
    MalformedResponseError,
    RetryStats,
    SafetyBlockError,
    SchemaMismatchError,
    call_with_retry,
)
from dm_bench.config import ModelConfig
from dm_bench.contracts import PolicyRequest, PolicyResult

_FORBIDDEN_GEN_CONFIG_KEYS: frozenset[str] = frozenset(
    {"temperature", "top_p", "top_k", "response_mime_type", "max_output_tokens", "store"}
)


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


def _extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    steps = getattr(response, "steps", None)
    if isinstance(steps, list):
        parts: list[str] = []
        for step in reversed(steps):
            step_type = _get_field(step, "type")
            if step_type == "user_input":
                break
            if step_type == "thought":
                continue
            if step_type != "model_output":
                continue
            content = _get_field(step, "content")
            if isinstance(content, list):
                for item in reversed(content):
                    if _get_field(item, "type") == "text":
                        item_text = _get_field(item, "text")
                        if isinstance(item_text, str):
                            parts.append(item_text)
        if parts:
            return "".join(reversed(parts))

    return text if isinstance(text, str) else ""


class GeminiGenerativeAdapter(GenerativeAdapter):
    """Adapter for `gemini-4-hydrogen` via `client.aio.interactions.create` on `v1`."""

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
            api_version = model_config.api_version or "v1"
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

    def _build_generation_config(self) -> dict[str, Any]:
        raw_cfg = dict(self.model_config.generation_config or {})
        clean_cfg = {k: v for k, v in raw_cfg.items() if k not in _FORBIDDEN_GEN_CONFIG_KEYS}
        thinking_level = clean_cfg.get("thinking_level", "low")
        if thinking_level == "minimal":
            thinking_level = "low"
        clean_cfg["thinking_level"] = thinking_level
        return clean_cfg

    def _build_response_format(self, request: PolicyRequest) -> dict[str, Any]:
        return {
            "type": "text",
            "mime_type": "application/json",
            "schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": list(request.action_criteria.keys()),
                    }
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        }

    def _extract_usage_and_cost(self, response: Any) -> tuple[int, int, int, int, float]:
        usage = getattr(response, "usage", None) if response is not None else None
        if usage is None:
            return 0, 0, 0, 0, 0.0
        input_tokens = int(_get_field(usage, "total_input_tokens", 0) or 0)
        output_tokens = int(_get_field(usage, "total_output_tokens", 0) or 0)
        thought_tokens = int(_get_field(usage, "total_thought_tokens", 0) or 0)
        total_tokens = int(_get_field(usage, "total_tokens", 0) or 0)
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens + thought_tokens

        cost = self.model_config.pricing.estimate_cost_usd(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thought_tokens=thought_tokens,
            total_tokens=total_tokens,
        )
        return input_tokens, output_tokens, thought_tokens, total_tokens, cost

    async def decide(self, request: PolicyRequest) -> PolicyResult:
        if not request.frames and request.state_text is None:
            raise ValueError("PolicyRequest must include at least one frame or state_text")
        if not request.action_criteria:
            raise ValueError("PolicyRequest.action_criteria must be non-empty")

        input_blocks = self._build_input_blocks(request)
        system_instruction = self.build_system_instruction(request)
        generation_config = self._build_generation_config()
        response_format = self._build_response_format(request)
        payload_hash = compute_request_payload_hash(
            self.model_config.model_id,
            request,
            extra={
                "api_version": self.model_config.api_version or "v1",
                "generation_config": generation_config,
                "system_instruction": system_instruction,
            },
        )

        stats = RetryStats()
        response: Any = None
        try:
            outcome = await call_with_retry(
                lambda: self._client.aio.interactions.create(
                    model=self.model_config.model_id,
                    input=input_blocks,
                    system_instruction=system_instruction,
                    generation_config=generation_config,
                    response_format=response_format,
                    store=False,
                ),
                max_retries=self._max_retries,
                base_delay=self._retry_base_delay,
                stats=stats,
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
            in_tok, out_tok, th_tok, tot_tok, cost = self._extract_usage_and_cost(response)
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
                input_tokens=in_tok,
                output_tokens=out_tok,
                thought_tokens=th_tok,
                total_tokens=tot_tok,
                api_cost_usd=cost,
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
        status = getattr(response, "status", None)
        errors = getattr(response, "errors", None)
        if status in ("failed", "cancelled", "incomplete"):
            detail = f"status={status!r}"
            if errors:
                detail = f"{detail}, errors={errors!r}"
                err_text = str(errors).lower()
                if "safety" in err_text or "block" in err_text or "policy" in err_text:
                    raise SafetyBlockError(f"Gemini Interaction blocked: {detail}")
            raise MalformedResponseError(f"Gemini Interaction did not complete: {detail}")

        output_text = _extract_output_text(response).strip()
        if not output_text:
            raise MalformedResponseError("Gemini Interaction returned empty output_text")

        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise MalformedResponseError(f"Invalid JSON from Gemini Interaction: {exc}") from exc

        if not isinstance(parsed, dict) or "action" not in parsed:
            raise SchemaMismatchError(f"JSON response missing 'action' field: {parsed!r}")

        action = parsed["action"]
        if not isinstance(action, str) or action not in request.action_criteria:
            raise SchemaMismatchError(
                f"Gemini Interaction action {action!r} not in allowed enum {list(request.action_criteria.keys())}"
            )

        in_tok, out_tok, th_tok, tot_tok, cost = self._extract_usage_and_cost(response)

        return PolicyResult(
            requested_action=action,
            action_probabilities=None,
            confidence=None,
            response_valid=True,
            retries=retries,
            model_latency_ms=model_latency_ms,
            end_to_end_latency_ms=end_to_end_latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            thought_tokens=th_tok,
            total_tokens=tot_tok,
            api_cost_usd=cost,
            raw_response={"output_text": output_text, "parsed": parsed},
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
