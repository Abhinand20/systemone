# systemone (`dm-bench`) — Can Decision Models Play Games?

Multimodal agent benchmark comparing typed, non-generative **Decision Models** (`jev-1.14.0`, `gemini-decision-1`) against generative **VLM/LLMs** (`gemini-4-hydrogen`) across pixels, structured text, and hybrid inputs.

Games: **Tic-tac-toe → Breakout → ViZDoom**.

See [RESEARCH_PLAN.md](./RESEARCH_PLAN.md) for the research questions, hypotheses, experimental matrix, and Milestone 1 design decisions, and [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for architecture and SDK specifications.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```

## Offline Mock Verification

Run the entire Milestone 0 + Milestone 1 pipeline deterministically without API credentials:

```bash
.venv/bin/bench build-corpus
.venv/bin/bench smoke --mock
.venv/bin/bench eval --exp all --pilot --mock --max-cost-usd 5.0
.venv/bin/bench analyze --tag mock-pilot
.venv/bin/bench view --port 8000
```

Outputs land in `experiments/results/`, `analysis/plots/mock-pilot/`, and `analysis/reports/mock-pilot_summary.md`. Open `http://127.0.0.1:8000` while `bench view` is running to inspect decision traces, 9-way comparisons, and closed-loop T6 episodes.

## Live API Testing (Milestone 0 → Milestone 1)

1. Configure credentials:
   ```bash
   cp .env.example .env
   # Edit .env and set TYPESAFE_API_KEY and GEMINI_API_KEY
   .venv/bin/bench check-env
   ```

2. Run Milestone 0 contract & latency smoke check (`$1.00` hard cap):
   ```bash
   .venv/bin/bench smoke --max-cost-usd 1.0
   ```

3. Run a live `--pilot` sweep across all experiments (`$3.00` hard cap) and inspect:
   ```bash
   .venv/bin/bench eval --exp all --pilot --max-cost-usd 3.0
   .venv/bin/bench analyze --tag live-pilot
   .venv/bin/bench view --port 8000
   ```

4. Run the full Milestone 1 held-out suite (`$10.00` default cap per experiment; `--resume-run-id <run_id>` supported on interruption):
   ```bash
   .venv/bin/bench eval --exp T1_T2 --max-cost-usd 10.0
   .venv/bin/bench eval --exp T3 --max-cost-usd 10.0
   .venv/bin/bench eval --exp T4 --max-cost-usd 10.0
   .venv/bin/bench eval --exp T5 --max-cost-usd 10.0
   .venv/bin/bench eval --exp T6 --max-cost-usd 10.0
   .venv/bin/bench analyze --tag m1-full
   ```

## Repository Layout

- `experiments/configs/` — Model, game, and 9-row experimental condition configs
- `experiments/prompts/` — Versioned prompt and action criteria templates
- `experiments/datasets/` — Deterministic stratified dev/test/ablation/pilot corpora
- `experiments/media/` — Content-addressed (`sha256`) 512×512 PNG frame store
- `experiments/results/` — Run logs (`decisions.jsonl`, `episodes.jsonl`, `manifest.yaml`)
- `analysis/` — Generated plots (`analysis/plots/`) and Markdown summary reports (`analysis/reports/`)
- `src/dm_bench/` — Core benchmark package (config, contracts, logging, games, adapters, runners, analysis, viewer)
- `tests/` — Unit and integration test suite
