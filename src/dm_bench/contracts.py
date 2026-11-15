"""Core Pydantic contracts for policies, decision logs, episodes, and manifests."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DecisionResumeKey = tuple[str, str, str | None, str | None, str]
EpisodeResumeKey = tuple[str, str, str, str]


class ImageFrame(BaseModel):
    """Single rendered observation frame attached to a PolicyRequest."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_id: str = "board_image"
    png_bytes: bytes = Field(..., repr=False)
    sha256: str | None = None
    width: int = 512
    height: int = 512
    mime_type: str = "image/png"
    path: str | None = None


class PolicyRequest(BaseModel):
    """Unified observation and action-choice request passed to every PolicyAdapter."""

    request_id: str = ""
    game: str = "tic_tac_toe"
    episode_id: str
    step_idx: int = 0
    row_id: str
    condition_code: str
    model_key: str
    objective_instructions: str
    action_criteria: dict[str, str]
    frames: list[ImageFrame] = Field(default_factory=list)
    state_text: str | None = None
    render_variant: str | None = None
    text_format: str | None = None
    action_mode: Literal["all_9", "legal_only"] = "all_9"
    player_to_move: str | None = None
    legal_actions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyResult(BaseModel):
    """Standardized response returned by every PolicyAdapter."""

    requested_action: str | None = None
    action_probabilities: dict[str, float] | None = None
    confidence: float | None = None
    response_valid: bool = True
    error_type: str | None = None
    error_message: str | None = None
    retries: int = 0
    model_latency_ms: float | None = None
    end_to_end_latency_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    total_tokens: int = 0
    api_cost_usd: float = 0.0
    raw_response: dict[str, Any] | str | None = None
    request_payload_hash: str | None = None
    model_id: str | None = None
    provider: str | None = None


class DecisionLogRow(BaseModel):
    """One JSONL record per model decision across M0 smoke, T1-T5 offline, and T6 turns."""

    experiment_id: str
    run_id: str
    row_id: str
    condition_code: str
    model_key: str
    model_id: str
    family: str
    episode_id: str
    step_idx: int = 0
    timestamp_utc: str

    game: str = "tic_tac_toe"
    render_variant: str | None = None
    text_format: str | None = None
    action_mode: Literal["all_9", "legal_only"] = "all_9"
    frame_paths: list[str] = Field(default_factory=list)
    frame_hashes: list[str] = Field(default_factory=list)
    state_text: str | None = None
    request_payload_hash: str | None = None
    action_criteria_keys: list[str] = Field(default_factory=list)

    board: list[str] = Field(default_factory=list)
    player_to_move: str | None = None
    starting_player: str | None = None
    ply: int = 0
    move_history: list[str] = Field(default_factory=list)
    legal_actions: list[str] = Field(default_factory=list)
    state_value: int | None = None
    optimal_actions: list[str] = Field(default_factory=list)
    action_values: dict[str, int] = Field(default_factory=dict)
    tactical_category: str | None = None
    canonical_hash: str | None = None
    d4_sym_id: int | None = None

    requested_action: str | None = None
    executed_action: str | None = None
    response_valid: bool = True
    action_is_legal: bool = False
    action_is_optimal: bool = False
    legal_regret: int | float | None = None
    effective_regret: int | float | None = None
    minimax_regret: int | float | None = None
    action_probabilities: dict[str, float] | None = None
    confidence: float | None = None
    optimal_mass: float | None = None
    illegal_mass: float | None = None

    retries: int = 0
    error_type: str | None = None
    error_message: str | None = None
    model_latency_ms: float | None = None
    end_to_end_latency_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    total_tokens: int = 0
    api_cost_usd: float = 0.0

    opponent_type: str | None = None
    agent_mark: str | None = None
    goes_first: bool | None = None
    termination_reason: str | None = None
    raw_response: dict[str, Any] | str | None = None

    @model_validator(mode="after")
    def _sync_regret_aliases(self) -> "DecisionLogRow":
        """Keep effective_regret and minimax_regret synchronized if only one is supplied."""
        if self.effective_regret is None and self.minimax_regret is not None:
            self.effective_regret = self.minimax_regret
        elif self.minimax_regret is None and self.effective_regret is not None:
            self.minimax_regret = self.effective_regret
        return self

    def resume_key(self) -> DecisionResumeKey:
        """Return deterministic checkpoint-resume key for offline evaluations."""
        return (
            self.row_id,
            self.episode_id,
            self.render_variant,
            self.text_format,
            self.action_mode,
        )


class EpisodeLogRow(BaseModel):
    """One JSONL record per completed closed-loop match in T6."""

    experiment_id: str = "T6"
    run_id: str
    episode_id: str
    row_id: str
    condition_code: str
    model_key: str
    model_id: str
    family: str
    game: str = "tic_tac_toe"
    action_mode: Literal["all_9", "legal_only"] = "all_9"
    opponent_type: Literal["random", "minimax"]
    episode_seed: int
    agent_mark: Literal["X", "O"]
    starting_player: Literal["X", "O"]
    goes_first: bool
    outcome: Literal["win", "draw", "loss_by_board", "loss_by_illegal_move"]
    winner: str | None = None
    termination_reason: str
    agent_reward: float
    num_plies: int
    num_agent_turns: int
    illegal_moves: int = 0
    invalid_responses: int = 0
    optimal_agent_turns: int = 0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_thought_tokens: int = 0
    total_latency_ms: float = 0.0
    mean_effective_regret: float | None = None
    timestamp_utc: str

    def resume_key(self) -> EpisodeResumeKey:
        """Return deterministic checkpoint-resume key for closed-loop episodes."""
        return (self.row_id, self.episode_id, self.opponent_type, self.action_mode)


class RunManifest(BaseModel):
    """Run-level metadata saved as manifest.yaml inside each experiment run directory."""

    experiment_id: str
    run_id: str
    run_name: str | None = None
    game: str = "tic_tac_toe"
    status: Literal["running", "completed", "halted_budget_cap", "failed"] = "running"
    created_at_utc: str
    updated_at_utc: str
    completed_at_utc: str | None = None
    mock_mode: bool = False
    pilot_mode: bool = False
    dataset_path: str | None = None
    rows: list[str] = Field(default_factory=list)
    render_variants: list[str] = Field(default_factory=list)
    text_formats: list[str] = Field(default_factory=list)
    action_modes: list[str] = Field(default_factory=list)
    max_cost_usd: float = 10.0
    total_spent_usd: float = 0.0
    total_decisions: int = 0
    total_episodes: int = 0
    valid_decisions: int = 0
    illegal_decisions: int = 0
    invalid_decisions: int = 0
    model_configs: dict[str, Any] = Field(default_factory=dict)
    git_commit: str | None = None
    notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
