"""dm_bench: Multimodal Decision Model vs Generative VLM/LLM benchmark harness."""

from dm_bench.config import (
    BudgetExceededError,
    BudgetTracker,
    ConditionRowConfig,
    CoreMatrixConfig,
    GameConfig,
    ModelConfig,
    PricingConfig,
    PromptConfig,
)
from dm_bench.contracts import (
    DecisionLogRow,
    EpisodeLogRow,
    ImageFrame,
    PolicyRequest,
    PolicyResult,
    RunManifest,
)
from dm_bench.logging_io import (
    MediaStore,
    RunLogger,
    sha256_bytes,
    sha256_json,
    sha256_text,
)

__version__ = "0.1.0"

__all__ = [
    "BudgetExceededError",
    "BudgetTracker",
    "ConditionRowConfig",
    "CoreMatrixConfig",
    "DecisionLogRow",
    "EpisodeLogRow",
    "GameConfig",
    "ImageFrame",
    "MediaStore",
    "ModelConfig",
    "PolicyRequest",
    "PolicyResult",
    "PricingConfig",
    "PromptConfig",
    "RunLogger",
    "RunManifest",
    "__version__",
    "sha256_bytes",
    "sha256_json",
    "sha256_text",
]
