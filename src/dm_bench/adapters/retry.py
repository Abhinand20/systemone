"""Async transient-error retry helper with exponential backoff and latency accounting."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import random
import re
import time
from typing import Any, Awaitable, Callable, Generic, Literal, TypeVar

import httpx
from pydantic import ValidationError

from dm_bench.config import BudgetExceededError

try:
    import httpx2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    httpx2 = None

try:
    from typesafe_sdk import (
        TypeSafeAPIConnectionError,
        TypeSafeAPITimeoutError,
        TypeSafeInternalServerError,
        TypeSafeRateLimitError,
    )

    _TYPESAFE_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
        TypeSafeRateLimitError,
        TypeSafeInternalServerError,
        TypeSafeAPIConnectionError,
        TypeSafeAPITimeoutError,
    )
except ImportError:  # pragma: no cover
    _TYPESAFE_TRANSIENT_TYPES = ()

try:
    from google.genai import _interactions as _genai_interactions
    from google.genai import errors as _genai_errors

    _GENAI_TRANSIENT_TYPES: tuple[type[BaseException], ...] = tuple(
        cls
        for cls in (
            getattr(_genai_errors, "ServerError", None),
            getattr(_genai_interactions, "RateLimitError", None),
            getattr(_genai_interactions, "InternalServerError", None),
            getattr(_genai_interactions, "APIConnectionError", None),
            getattr(_genai_interactions, "APITimeoutError", None),
        )
        if isinstance(cls, type)
    )
except ImportError:  # pragma: no cover
    _GENAI_TRANSIENT_TYPES = ()

logger = logging.getLogger(__name__)

T = TypeVar("T")

TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

ErrorCategory = Literal["rate_limit", "api_error", "model_output"]

MODEL_OUTPUT_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "MalformedResponseError",
        "SchemaMismatchError",
        "SchemaParseError",
        "SafetyBlockError",
        "MalformedOutputError",
        "MalformedJSON",
        "ParseError",
        "ValidationError",
        "JSONDecodeError",
    }
)

_RETRY_IN_SECONDS_RE = re.compile(r"retry\s+in\s+(\d+(?:\.\d+)?)\s*(ms|s)\b", re.IGNORECASE)
_RETRY_DELAY_RE = re.compile(r"retry_?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_MAX_RETRY_AFTER_SECONDS: float = 120.0


class NonRetryableAdapterError(RuntimeError):
    """Base class for response/contract failures that must never be retried."""


class MalformedResponseError(NonRetryableAdapterError):
    """Raised when model output is empty, unparseable, or malformed JSON."""


class SchemaMismatchError(NonRetryableAdapterError):
    """Raised when model output violates the expected choice/JSON schema."""


class SafetyBlockError(NonRetryableAdapterError):
    """Raised when a provider blocks or halts a response due to safety/policy."""


@dataclass(slots=True)
class RetryStats:
    """Mutable latency and retry counter populated even if all retries fail."""

    retries: int = 0
    model_latency_ms: float | None = None
    end_to_end_latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class RetryOutcome(Generic[T]):
    """Successful invocation result paired with retry and latency telemetry."""

    value: T
    retries: int
    model_latency_ms: float
    end_to_end_latency_ms: float


def _extract_status_code(exc: BaseException) -> int | None:
    """Extract an integer HTTP status code from SDK or HTTP client exceptions."""
    for attr in ("status_code", "status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            return val
    response = getattr(exc, "response", None)
    if response is not None:
        resp_code = getattr(response, "status_code", None)
        if isinstance(resp_code, int) and not isinstance(resp_code, bool) and resp_code > 0:
            return resp_code
    return None


def is_transient_error(exc: BaseException) -> bool:
    """Return True iff `exc` represents a transient 429/5xx/network/timeout error."""
    if isinstance(
        exc,
        (
            NonRetryableAdapterError,
            BudgetExceededError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
            TypeError,
            KeyError,
        ),
    ):
        return False

    status = _extract_status_code(exc)
    if status is not None:
        return status in TRANSIENT_STATUS_CODES or (500 <= status < 600)

    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return True

    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True

    if httpx2 is not None:
        h2_types = tuple(
            cls
            for cls in (
                getattr(httpx2, "TimeoutException", None),
                getattr(httpx2, "NetworkError", None),
                getattr(httpx2, "RemoteProtocolError", None),
            )
            if isinstance(cls, type)
        )
        if h2_types and isinstance(exc, h2_types):
            return True

    if _TYPESAFE_TRANSIENT_TYPES and isinstance(exc, _TYPESAFE_TRANSIENT_TYPES):
        return True

    if _GENAI_TRANSIENT_TYPES and isinstance(exc, _GENAI_TRANSIENT_TYPES):
        return True

    return False


def is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if `exc` represents a 429 / rate-limit response."""
    if _extract_status_code(exc) == 429:
        return True
    name = type(exc).__name__.lower()
    if "ratelimit" in name:
        return True
    msg = str(exc).lower()
    return "too_many_requests" in msg or "rate limit exceeded" in msg


def extract_retry_after_seconds(exc: BaseException) -> float | None:
    """Extract server-suggested retry delay in seconds from HTTP headers or message."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is not None:
        raw_header = headers.get("retry-after") or headers.get("Retry-After")
        if raw_header is not None:
            try:
                val = float(str(raw_header).strip())
                if val > 0:
                    return min(val, _MAX_RETRY_AFTER_SECONDS)
            except ValueError:
                pass

    msg = str(exc)
    match = _RETRY_IN_SECONDS_RE.search(msg)
    if match:
        val = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "ms":
            val /= 1000.0
        if val > 0:
            return min(val, _MAX_RETRY_AFTER_SECONDS)

    match_delay = _RETRY_DELAY_RE.search(msg)
    if match_delay:
        val = float(match_delay.group(1))
        if val > 0:
            return min(val, _MAX_RETRY_AFTER_SECONDS)

    return None


def classify_error(exc: BaseException) -> ErrorCategory:
    """Classify an adapter exception as `rate_limit`, `api_error`, or `model_output`."""
    if isinstance(
        exc,
        (
            NonRetryableAdapterError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
            TypeError,
            KeyError,
        ),
    ):
        return "model_output"
    if is_rate_limit_error(exc):
        return "rate_limit"
    return "api_error"


def infer_error_category(
    error_type: str | None,
    error_message: str | None = None,
    explicit_category: str | None = None,
) -> ErrorCategory | None:
    """Infer error category from explicit category or logged error_type/error_message."""
    if explicit_category in ("rate_limit", "api_error", "model_output"):
        return explicit_category  # type: ignore[return-value]
    if not error_type:
        return None
    if error_type in MODEL_OUTPUT_ERROR_TYPES:
        return "model_output"
    et_lower = error_type.lower()
    if any(tok in et_lower for tok in ("malformed", "schema", "parse", "json", "safety", "validation")):
        return "model_output"
    msg_lower = (error_message or "").lower()
    if (
        "ratelimit" in et_lower
        or "too_many_requests" in msg_lower
        or "rate limit" in msg_lower
        or "error code: 429" in msg_lower
    ):
        return "rate_limit"
    return "api_error"


def is_infra_error(
    error_type: str | None,
    error_message: str | None = None,
    error_category: str | None = None,
) -> bool:
    """Return True if a failure was caused by rate limits or API/infra errors, not model output."""
    cat = infer_error_category(error_type, error_message, error_category)
    return cat in ("rate_limit", "api_error")


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    rate_limit_max_retries: int | None = None,
    base_delay: float = 1.0,
    jitter_fraction: float = 0.25,
    rng: random.Random | None = None,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    stats: RetryStats | None = None,
    on_rate_limit: Callable[[float], None] | None = None,
) -> RetryOutcome[T]:
    """Execute async `fn` with exponential backoff ONLY on transient errors.

    Backoff schedule defaults to `1s, 2s, 4s` (+ up to `25%` jitter) for `max_retries=3`.
    For HTTP 429 rate-limit errors, server-provided `Retry-After` / `"retry in Xs"` hints are
    honored when `base_delay > 0`, and `rate_limit_max_retries` (if provided) may allow
    additional rate-limit retries. Malformed outputs, schema mismatches, safety blocks, and
    non-transient 4xx errors trigger zero retries and re-raise immediately.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")
    if rate_limit_max_retries is not None and rate_limit_max_retries < 0:
        raise ValueError(f"rate_limit_max_retries must be >= 0, got {rate_limit_max_retries}")
    if base_delay < 0:
        raise ValueError(f"base_delay must be >= 0, got {base_delay}")

    e2e_start = time.perf_counter()
    retries = 0
    last_model_latency_ms = 0.0

    while True:
        call_start = time.perf_counter()
        try:
            value = await fn()
        except Exception as exc:
            call_end = time.perf_counter()
            last_model_latency_ms = (call_end - call_start) * 1000.0
            rate_limited = is_rate_limit_error(exc)
            effective_max_retries = (
                rate_limit_max_retries
                if (rate_limited and rate_limit_max_retries is not None)
                else max_retries
            )
            if is_transient_error(exc) and retries < effective_max_retries:
                delay_base = base_delay * (2**retries)
                hint_seconds = (
                    extract_retry_after_seconds(exc)
                    if (rate_limited and base_delay > 0)
                    else None
                )
                if hint_seconds is not None:
                    delay_base = max(delay_base, hint_seconds)
                    max_jitter = min(delay_base * max(0.0, jitter_fraction), 4.0)
                else:
                    max_jitter = delay_base * max(0.0, jitter_fraction)
                jitter = (
                    rng.uniform(0.0, max_jitter)
                    if rng is not None
                    else random.uniform(0.0, max_jitter)
                )
                wait_seconds = delay_base + jitter
                if rate_limited and on_rate_limit is not None and wait_seconds > 0:
                    on_rate_limit(wait_seconds)
                retries += 1
                logger.warning(
                    "Transient error (%s: %s); retry %d/%d in %.3fs",
                    type(exc).__name__,
                    exc,
                    retries,
                    effective_max_retries,
                    wait_seconds,
                )
                if wait_seconds > 0:
                    await sleep_fn(wait_seconds)
                continue

            e2e_latency_ms = (time.perf_counter() - e2e_start) * 1000.0
            if stats is not None:
                stats.retries = retries
                stats.model_latency_ms = last_model_latency_ms
                stats.end_to_end_latency_ms = e2e_latency_ms
            raise

        call_end = time.perf_counter()
        model_latency_ms = (call_end - call_start) * 1000.0
        e2e_latency_ms = (call_end - e2e_start) * 1000.0
        if stats is not None:
            stats.retries = retries
            stats.model_latency_ms = model_latency_ms
            stats.end_to_end_latency_ms = e2e_latency_ms
        return RetryOutcome(
            value=value,
            retries=retries,
            model_latency_ms=model_latency_ms,
            end_to_end_latency_ms=e2e_latency_ms,
        )
