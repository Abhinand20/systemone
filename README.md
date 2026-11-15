# systemone (`dm-bench`) — Can Decision Models Play Games?

Multimodal agent benchmark comparing typed, non-generative **Decision Models** (`jev-1.14.0`, `gemini-decision-1`) against generative **VLM/LLMs** (`gemini-4-hydrogen`) across pixels, structured text, and hybrid inputs.

Games: **Tic-tac-toe → Breakout → ViZDoom**.

See [RESEARCH_PLAN.md](./RESEARCH_PLAN.md) for the research questions, hypotheses, experimental matrix, and Milestone 1 design decisions, and [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for architecture and SDK specifications.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
# Populate TYPESAFE_API_KEY and GEMINI_API_KEY in .env for live runs
```

## CLI Workflow

```bash
.venv/bin/bench check-env
.venv/bin/bench build-corpus
.venv/bin/bench smoke --mock
.venv/bin/bench smoke --max-cost-usd 1.0
.venv/bin/bench eval --exp all --pilot --max-cost-usd 3.0
.venv/bin/bench analyze
.venv/bin/bench view --port 8000
```

## Repository Layout

- `experiments/configs/` — Model, game, and 9-row experimental condition configs
- `experiments/prompts/` — Versioned prompt and action criteria templates
- `experiments/datasets/` — Deterministic stratified dev/test/ablation/pilot corpora
- `experiments/media/` — Content-addressed (`sha256`) 512×512 PNG frame store
- `experiments/results/` — Run logs (`decisions.jsonl`, `episodes.jsonl`, `manifest.yaml`)
- `src/dm_bench/` — Core benchmark package (config, contracts, logging, games, adapters, runners, analysis, viewer)
- `tests/` — Unit and integration test suite
