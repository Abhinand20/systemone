"""YAML configuration loaders, pricing calculator, and thread-safe USD budget gate."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class BudgetExceededError(RuntimeError):
    """Raised when accumulated API spend meets or exceeds the configured USD cap."""

    def __init__(self, spent_usd: float, max_cost_usd: float) -> None:
        self.spent_usd = spent_usd
        self.max_cost_usd = max_cost_usd
        super().__init__(
            f"Hard USD budget cap exceeded: spent=${spent_usd:.6f} >= max=${max_cost_usd:.6f}"
        )


class BudgetTracker:
    """Thread-safe hard USD spend cap checked before dispatch and updated after response."""

    def __init__(self, max_cost_usd: float = 10.0, initial_spent_usd: float = 0.0) -> None:
        if max_cost_usd < 0:
            raise ValueError(f"max_cost_usd must be non-negative, got {max_cost_usd}")
        if initial_spent_usd < 0:
            raise ValueError(f"initial_spent_usd must be non-negative, got {initial_spent_usd}")
        self._max_cost_usd = float(max_cost_usd)
        self._spent_usd = float(initial_spent_usd)
        self._call_count = 0
        self._lock = threading.Lock()

    @property
    def max_cost_usd(self) -> float:
        with self._lock:
            return self._max_cost_usd

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent_usd

    @property
    def remaining_usd(self) -> float:
        with self._lock:
            return max(0.0, self._max_cost_usd - self._spent_usd)

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    @property
    def is_exceeded(self) -> bool:
        with self._lock:
            return self._spent_usd >= self._max_cost_usd

    def check_can_proceed(self) -> None:
        """Raise BudgetExceededError if spent_usd >= max_cost_usd before dispatching a call."""
        with self._lock:
            if self._spent_usd >= self._max_cost_usd:
                raise BudgetExceededError(self._spent_usd, self._max_cost_usd)

    def record(self, cost_usd: float) -> float:
        """Record completed call cost in USD and return the updated total spent."""
        added = max(0.0, float(cost_usd))
        with self._lock:
            self._spent_usd += added
            self._call_count += 1
            return self._spent_usd


class PricingConfig(BaseModel):
    """Per-million-token pricing with non-double-counting billable token calculation."""

    input_per_1m_usd: float = 0.0
    output_per_1m_usd: float = 0.0
    thought_per_1m_usd: float = 0.0

    def estimate_cost_usd(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        thought_tokens: int = 0,
        total_tokens: int | None = None,
    ) -> float:
        """Compute USD cost without double-counting output and thought tokens.

        Per IMPLEMENTATION_PLAN.md §3.2 rule 3: when both `total_tokens` (> 0) and
        `input_tokens` are populated, non-input billable tokens equal
        `max(0, total_tokens - input_tokens)`, preventing double-counting when
        providers include thoughts in `output_tokens` or `total_tokens`. Otherwise
        fall back to `max(0, output_tokens) + max(0, thought_tokens)`.
        """
        in_tok = max(0, int(input_tokens or 0))
        out_tok = max(0, int(output_tokens or 0))
        th_tok = max(0, int(thought_tokens or 0))

        if total_tokens is not None and total_tokens > 0 and input_tokens is not None:
            non_input_tokens = max(0, int(total_tokens) - in_tok)
            if self.output_per_1m_usd == self.thought_per_1m_usd:
                non_input_cost = (non_input_tokens / 1_000_000.0) * self.output_per_1m_usd
            else:
                thought_billable = min(th_tok, non_input_tokens)
                output_billable = non_input_tokens - thought_billable
                non_input_cost = (
                    (output_billable / 1_000_000.0) * self.output_per_1m_usd
                    + (thought_billable / 1_000_000.0) * self.thought_per_1m_usd
                )
        else:
            non_input_cost = (
                (out_tok / 1_000_000.0) * self.output_per_1m_usd
                + (th_tok / 1_000_000.0) * self.thought_per_1m_usd
            )

        input_cost = (in_tok / 1_000_000.0) * self.input_per_1m_usd
        return float(input_cost + non_input_cost)


class ModelConfig(BaseModel):
    """Configuration for a pinned model backend."""

    model_key: str
    model_id: str
    provider: Literal["typesafe", "google_decision", "google_generative"]
    family: Literal["decision_model", "generative"]
    api_env_var: str
    api_version: str | None = None
    supports_vision: bool = True
    supports_text: bool = True
    image_limits: dict[str, int] = Field(default_factory=dict)
    extra_headers_on_images: dict[str, str] = Field(default_factory=dict)
    pricing: PricingConfig
    generation_config: dict[str, Any] = Field(default_factory=dict)


class ConditionRowConfig(BaseModel):
    """One row of the 9-row experimental condition matrix."""

    row_id: str
    condition_code: str
    model_key: str
    use_vision: bool
    use_text: bool
    family: Literal["decision_model", "generative"]


class CoreMatrixConfig(BaseModel):
    """Container for the 9 model-condition rows."""

    rows: list[ConditionRowConfig]

    def get_row(self, row_id: str) -> ConditionRowConfig:
        """Return the row matching `row_id`, raising KeyError if absent."""
        for row in self.rows:
            if row.row_id == row_id:
                return row
        raise KeyError(f"Unknown condition row_id: {row_id}")

    def row_ids(self) -> list[str]:
        """Return ordered list of all row IDs."""
        return [row.row_id for row in self.rows]


class GameConfig(BaseModel):
    """Game-level defaults, supported variants, and dataset paths."""

    game: str
    board_size: int = 3
    image_size: tuple[int, int] = (512, 512)
    default_render_variant: str = "clean_default"
    render_variants: list[str]
    default_text_format: str = "json_canonical"
    text_formats: list[str]
    default_action_mode: str = "all_9"
    action_modes: list[str]
    datasets: dict[str, str]
    prompt_template: str


class PromptConfig(BaseModel):
    """Shared objective instructions and R1C1..R3C3 action criteria."""

    prompt_version: str
    objective_instructions: str
    action_criteria: dict[str, str]

    def get_action_criteria(
        self,
        legal_actions: list[str] | None = None,
        action_mode: str = "all_9",
    ) -> dict[str, str]:
        """Return ordered action criteria dict filtered for `legal_only` when requested."""
        if action_mode == "all_9":
            return dict(self.action_criteria)
        if action_mode == "legal_only":
            if legal_actions is None:
                raise ValueError("legal_actions must be provided when action_mode='legal_only'")
            legal_set = set(legal_actions)
            return {k: v for k, v in self.action_criteria.items() if k in legal_set}
        raise ValueError(f"Unsupported action_mode: {action_mode}")

    def build_generative_system_instruction(
        self,
        action_criteria: dict[str, str] | None = None,
    ) -> str:
        """Build generative model system instruction using exact prompt parity."""
        criteria = action_criteria if action_criteria is not None else self.action_criteria
        criteria_lines = "\n".join(f"- {key}: {desc}" for key, desc in criteria.items())
        return (
            f"{self.objective_instructions}\n\n"
            f"Valid action keys:\n{criteria_lines}\n\n"
            'Respond ONLY with a JSON object matching {"action": "<KEY>"} with no commentary.'
        )


def find_repo_root(start: Path | None = None) -> Path:
    """Locate repository root by walking upward for pyproject.toml or .git."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return data


def load_env(repo_root: Path | None = None) -> Path | None:
    """Load environment variables from `.env` at repository root if present."""
    root = find_repo_root(repo_root)
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
        return env_path
    return None


def load_model_config(model_key: str, repo_root: Path | None = None) -> ModelConfig:
    """Load a single model config YAML from `experiments/configs/models/<model_key>.yaml`."""
    root = find_repo_root(repo_root)
    path = root / "experiments" / "configs" / "models" / f"{model_key}.yaml"
    return ModelConfig.model_validate(_load_yaml(path))


def load_all_model_configs(repo_root: Path | None = None) -> dict[str, ModelConfig]:
    """Load all model configs under `experiments/configs/models/` keyed by `model_key`."""
    root = find_repo_root(repo_root)
    models_dir = root / "experiments" / "configs" / "models"
    configs: dict[str, ModelConfig] = {}
    for path in sorted(models_dir.glob("*.yaml")):
        cfg = ModelConfig.model_validate(_load_yaml(path))
        configs[cfg.model_key] = cfg
    return configs


def load_core_matrix(repo_root: Path | None = None) -> CoreMatrixConfig:
    """Load the 9-row experimental condition matrix."""
    root = find_repo_root(repo_root)
    path = root / "experiments" / "configs" / "conditions" / "core_matrix.yaml"
    return CoreMatrixConfig.model_validate(_load_yaml(path))


def load_game_config(game: str = "tic_tac_toe", repo_root: Path | None = None) -> GameConfig:
    """Load game configuration YAML from `experiments/configs/games/<game>.yaml`."""
    root = find_repo_root(repo_root)
    path = root / "experiments" / "configs" / "games" / f"{game}.yaml"
    return GameConfig.model_validate(_load_yaml(path))


def load_prompt_config(
    path: Path | str | None = None,
    repo_root: Path | None = None,
) -> PromptConfig:
    """Load prompt template YAML (defaults to `experiments/prompts/tic_tac_toe_v1.yaml`)."""
    root = find_repo_root(repo_root)
    if path is None:
        resolved = root / "experiments" / "prompts" / "tic_tac_toe_v1.yaml"
    else:
        candidate = Path(path)
        resolved = candidate if candidate.is_absolute() else root / candidate
    return PromptConfig.model_validate(_load_yaml(resolved))
