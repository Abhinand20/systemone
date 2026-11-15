"""Unified policy adapters, transient retry helpers, and row factory."""

from dm_bench.adapters.base import (
    DecisionModelAdapter,
    GenerativeAdapter,
    MockDecisionAdapter,
    MockGenerativeAdapter,
    PolicyAdapter,
)
from dm_bench.adapters.factory import (
    BudgetedPolicyAdapter,
    build_row_adapters,
    close_adapters,
)
from dm_bench.adapters.gemini_decision import GeminiDecisionAdapter
from dm_bench.adapters.gemini_generative import GeminiGenerativeAdapter
from dm_bench.adapters.retry import (
    ErrorCategory,
    MalformedResponseError,
    NonRetryableAdapterError,
    RetryOutcome,
    RetryStats,
    SafetyBlockError,
    SchemaMismatchError,
    call_with_retry,
    classify_error,
    extract_retry_after_seconds,
    infer_error_category,
    is_infra_error,
    is_rate_limit_error,
    is_transient_error,
)
from dm_bench.adapters.typesafe_jev import TypeSafeJevAdapter

__all__ = [
    "BudgetedPolicyAdapter",
    "DecisionModelAdapter",
    "ErrorCategory",
    "GeminiDecisionAdapter",
    "GeminiGenerativeAdapter",
    "GenerativeAdapter",
    "MalformedResponseError",
    "MockDecisionAdapter",
    "MockGenerativeAdapter",
    "NonRetryableAdapterError",
    "PolicyAdapter",
    "RetryOutcome",
    "RetryStats",
    "SafetyBlockError",
    "SchemaMismatchError",
    "TypeSafeJevAdapter",
    "build_row_adapters",
    "call_with_retry",
    "classify_error",
    "close_adapters",
    "extract_retry_after_seconds",
    "infer_error_category",
    "is_infra_error",
    "is_rate_limit_error",
    "is_transient_error",
]
