"""Async transient-error retry helper with exponential backoff and latency accounting."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import random
import time
from typing import Any, Awaitable, Callable, Generic, TypeVar

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


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    jitter_fraction: float = 0.25,
    rng: random.Random | None = None,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    stats: RetryStats | None = None,
) -> RetryOutcome[T]:
    """Execute async `fn` with exponential backoff ONLY on transient errors.

    Backoff schedule defaults to `1s, 2s, 4s` (+ up to `25%` jitter) for `max_retries=3`.
    Malformed outputs, schema mismatches, safety blocks, and non-transient 4xx errors
    trigger zero retries and re-raise immediately.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")
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
            if is_transient_error(exc) and retries < max_retries:
                delay_base = base_delay * (2**retries)
                max_jitter = delay_base * max(0.0, jitter_fraction)
                jitter = (
                    rng.uniform(0.0, max_jitter)
                    if rng is not None
                    else random.uniform(0.0, max_jitter)
                )
                wait_seconds = delay_base + jitter
                retries += 1
                logger.warning(
                    "Transient error (%s: %s); retry %d/%d in %.3fs",
                    type(exc).__name__,
                    exc,
                    retries,
                    max_retries,
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
