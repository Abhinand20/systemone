"""Adapter factory wiring the 9 experimental condition rows with optional USD budget gating."""

from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path

from dm_bench.adapters.base import (
    MockDecisionAdapter,
    MockGenerativeAdapter,
    PolicyAdapter,
)
from dm_bench.adapters.gemini_decision import GeminiDecisionAdapter
from dm_bench.adapters.gemini_generative import GeminiGenerativeAdapter
from dm_bench.adapters.typesafe_jev import TypeSafeJevAdapter
from dm_bench.config import (
    BudgetTracker,
    ConditionRowConfig,
    ModelConfig,
    load_all_model_configs,
    load_core_matrix,
    load_env,
)
from dm_bench.contracts import PolicyRequest, PolicyResult


class BudgetedPolicyAdapter(PolicyAdapter):
    """PolicyAdapter wrapper enforcing hard USD budget checks before and after every call."""

    def __init__(self, inner: PolicyAdapter, budget_tracker: BudgetTracker) -> None:
        super().__init__(inner.model_config)
        self.inner = inner
        self.budget_tracker = budget_tracker

    async def decide(self, request: PolicyRequest) -> PolicyResult:
        self.budget_tracker.check_can_proceed()
        result = await self.inner.decide(request)
        self.budget_tracker.record(result.api_cost_usd)
        return result

    async def close(self) -> None:
        await self.inner.close()


def _create_live_adapter(model_cfg: ModelConfig) -> PolicyAdapter:
    api_key = os.environ.get(model_cfg.api_env_var, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing required environment variable {model_cfg.api_env_var} "
            f"for model {model_cfg.model_key}"
        )
    if model_cfg.provider == "typesafe":
        return TypeSafeJevAdapter(model_cfg, api_key=api_key)
    if model_cfg.provider == "google_decision":
        return GeminiDecisionAdapter(model_cfg, api_key=api_key)
    if model_cfg.provider == "google_generative":
        return GeminiGenerativeAdapter(model_cfg, api_key=api_key)
    raise ValueError(f"Unsupported provider: {model_cfg.provider!r}")


def _create_mock_adapter(model_cfg: ModelConfig) -> PolicyAdapter:
    if model_cfg.family == "decision_model":
        return MockDecisionAdapter(model_cfg)
    if model_cfg.family == "generative":
        return MockGenerativeAdapter(model_cfg)
    raise ValueError(f"Unsupported model family: {model_cfg.family!r}")


def build_row_adapters(
    repo_root: Path | None = None,
    *,
    mock: bool = False,
    budget_tracker: BudgetTracker | None = None,
    row_ids: list[str] | None = None,
) -> dict[str, tuple[ConditionRowConfig, ModelConfig, PolicyAdapter]]:
    """Build `(ConditionRowConfig, ModelConfig, PolicyAdapter)` tuples keyed by `row_id`."""
    if not mock:
        load_env(repo_root)

    matrix = load_core_matrix(repo_root)
    model_configs = load_all_model_configs(repo_root)

    if row_ids is not None:
        selected_rows = [matrix.get_row(rid) for rid in dict.fromkeys(row_ids)]
    else:
        selected_rows = list(matrix.rows)

    adapter_by_model: dict[str, PolicyAdapter] = {}
    result: dict[str, tuple[ConditionRowConfig, ModelConfig, PolicyAdapter]] = {}

    try:
        for row_cfg in selected_rows:
            if row_cfg.model_key not in model_configs:
                raise KeyError(
                    f"Condition row {row_cfg.row_id!r} references unknown model_key {row_cfg.model_key!r}"
                )
            model_cfg = model_configs[row_cfg.model_key]
            if row_cfg.model_key not in adapter_by_model:
                base_adapter = (
                    _create_mock_adapter(model_cfg)
                    if mock
                    else _create_live_adapter(model_cfg)
                )
                adapter_by_model[row_cfg.model_key] = base_adapter
            else:
                base_adapter = adapter_by_model[row_cfg.model_key]

            wrapped: PolicyAdapter = (
                BudgetedPolicyAdapter(base_adapter, budget_tracker)
                if budget_tracker is not None
                else base_adapter
            )
            result[row_cfg.row_id] = (row_cfg, model_cfg, wrapped)
    except Exception:
        # Best-effort sync cleanup if creation fails mid-way (e.g., missing second API key).
        for adapter in adapter_by_model.values():
            close_fn = getattr(getattr(adapter, "_client", None), "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        raise

    return result


async def close_adapters(
    adapters: dict[str, tuple[ConditionRowConfig, ModelConfig, PolicyAdapter]]
    | Iterable[PolicyAdapter],
) -> None:
    """Close all unique underlying adapters cleanly."""
    items: Iterable[PolicyAdapter]
    if isinstance(adapters, dict):
        items = (entry[2] for entry in adapters.values())
    else:
        items = adapters

    seen_ids: set[int] = set()
    for adapter in items:
        inner = getattr(adapter, "inner", adapter)
        key = id(inner)
        if key in seen_ids:
            continue
        seen_ids.add(key)
        await adapter.close()
