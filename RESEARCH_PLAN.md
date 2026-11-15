# Can Decision Models Play Games? A Multimodal Agent Benchmark

**Research and learning plan · v0.2 · November 14, 2026**  
**Status:** Proposed experiments; no results have been collected. Milestone 1 decisions locked in §15.  
**Primary models of interest:** TypeSafe Jev (`jev-1.14.0`) and Google Gemini Decision 1 (`gemini-decision-1`).  
**Generative comparator:** Google Gemini 4 Hydrogen (`gemini-4-hydrogen`, `thinking_level="low"`).  
**Games:** Tic-tac-toe → Breakout → ViZDoom (increasing control, perception, and planning demands).

> **North star:** Learn when a typed, non-generative decision model is an effective game-playing controller, how much the observation representation matters, and when a conventional LLM/VLM is a better choice. Produce reproducible evidence, informative failure cases, and a clear technical write-up—not a leaderboard victory at any cost.

## 1. Research questions

1. **Representation:** How much does performance change when an agent receives only pixels, only structured state, or both?
2. **Architecture:** Given the same observation and allowed actions, when do typed decision models outperform, match, or trail generative models in quality, latency, and cost?
3. **Perception versus policy:** Are failures caused by misunderstanding the game state or by choosing a poor action from a correctly perceived state?
4. **Temporal reasoning:** How do frame history, explicit velocity, action duration, and memory affect performance on dynamic games?
5. **Compositional control:** Does decomposing perception into atomic judgments (e.g., object present → object location → action) help more than requesting a direct action?
6. **Limits:** How does performance degrade as the task moves from discrete perfect-information games to fast, partially observable 3D environments?
7. **Practicality:** What combination achieves the strongest *game performance per dollar, second, and model call*?

**Scope:** Primarily inference-time agents with frozen model weights. No training, reinforcement learning, or fine-tuning in the core benchmark. These can be extensions after the initial comparison.

## 2. Testable hypotheses (not assumed results)

- **H1 — Structured-state advantage:** Text-only agents will outperform screenshot-only agents on tic-tac-toe due to near-zero perception ambiguity.
- **H2 — Rich-input advantage with caveats:** Hybrid inputs will generally improve reliability where screenshots omit information, but may not help if the extra state is redundant, noisy, or confers privileged knowledge.
- **H3 — Reaction-cost trade-off:** Typed decision models may offer better decision latency and cost than generative VLMs, but that alone will not guarantee better closed-loop scores.
- **H4 — Temporal bottleneck:** In Breakout, a single frame will underperform a short frame history or an explicit ball-velocity estimate, across model families.
- **H5 — Modular perception:** A pipeline of visual localization plus symbolic decisions will outperform direct image-to-action when perception is the bottleneck, at the cost of additional implementation complexity and/or calls.
- **H6 — Difficulty transition:** Performance differences will grow in Doom when action horizons, spatial ambiguity, and recovery from errors become important.

For every hypothesis, report **support, mixed evidence, or non-support** and consider alternative explanations. Do not selectively report only the most entertaining runs.

## 3. Experimental matrix: six core conditions

Keep model versions fixed within the study. “DM” is a *typed decision model* (e.g., Jev, Gemini Decision 1); “VLM” means a *generative language model with image input*; “LLM” means a text-only generative model. C, E, and F use the **same multimodal model** (`gemini-4-hydrogen`) with different inputs.

| ID | Policy | Perception input | Additional structured input | Expected output |
|---|---|---|---|---|
| **A — DM-V** | Decision model | Screenshot or frame history | None | Choice from shared action set |
| **B — DM-T** | Decision model | None | Text/JSON game state | Choice from shared action set |
| **C — VLM-V** | Generative multimodal LLM | Same screenshots as A | None | One enum action, validated |
| **D — DM-VT** | Decision model | Same screenshots as A | Same state as B | Choice from shared action set |
| **E — VLM-VT** | Generative multimodal LLM | Same screenshots as C | Same state as B | One enum action, validated |
| **F — LLM-T** | Text-only generative LLM | None | Same state as B | One enum action, validated |

**Optional secondary conditions (for later milestones / extensions; omitted in Milestone 1):**

- **G — CV → DM:** Deterministic or independently trained perception extracts objects/velocity, then DM chooses the action.
- **H — DM perception → DM policy:** Jev `Noul`/`Box`/`Choice` produces an intermediate scene description, then a separate Jev `Choice` selects an action. Explicitly count both calls.
- **I — Classical / oracle:** A random policy; a simple heuristic; and, when available, minimax or a rule-based near-oracle. These are reference baselines, not language models.

### Important comparison boundaries

- **A vs C**, **B vs F**, **D vs E** compare *systems*, not the pure causal effect of decoding strategy. The models may differ in size, training data, vision encoders, and knowledge.
- **A vs B** or **C vs F** are not strictly equivalent observation conditions. Label these **representation comparisons**, not proofs that a model is “smarter.”
- **D/E vs A/C** quantify what adding structured information does; the added information must be clearly characterized as *observable* or *privileged*.
- If a model fails to support multiple images, retain the single-frame condition and mark the multi-frame comparison unsupported rather than silently changing its input.

## 4. Model and API configurations

### 4.1 Decision models (TypeSafe Jev & Google Gemini Decision 1)

- Use a **pinned model ID**, not an unversioned alias, and record the version and request payload hash with each run (`jev-1.14.0` for TypeSafe Jev; `gemini-decision-1` for Google Decisions API `v1beta`).
- For each action, send the state (image refs, text, or both) and one **`Choice`** question with an identical set of labeled candidate actions. Record `choice`, the full `probabilities` vector, and `confidence` when returned.
- Treat `Noul` and `Box` as **additional experimental primitives**, not mandatory machinery for the core six conditions.
- TypeSafe image support is explicitly **experimental**. Official documentation requires the header `TypeSafe-Danger-Zone-Images: true` for requests containing images or Box questions. The API supports still PNG/JPEG assets, not video clips as such; supply sampled frames.
- Official Jev image limits currently include **280–1568 px per side**, **16 MiB per image**, and **32 MiB for the request**. **Atari Breakout frames are 210 × 160**, so resize/pad to a compliant, fixed size (e.g., 420 × 320) and use exactly the same rendered pixels for the paired VLM condition. Maintain aspect ratio.
- Keep image assets once in the `state` and reference them cleanly rather than repeatedly embedding them in separate questions.
- The vendor describes decision probabilities as calibrated; **measure calibration empirically** instead of accepting the claim.

### 4.2 Generative comparators

- Pin **`gemini-4-hydrogen`** (`thinking_level="low"`, `store=False`) on the Gemini Interactions API (`v1`) across C, E, and F so modality comparisons hold model weights constant, keeping the generative adapter interface generic for future comparators.
- Require a strict **single-action enum** via the provider’s structured-output mechanism. Record reasoning-effort settings, token caps, and all retries.
- Avoid asking the VLM for long chain-of-thought or explanations on every action; the baseline has the same objective as DM: **choose the next action**.
- Model generation failures, malformed output, timeouts, and retries are part of system performance and must be logged, not hidden.

### 4.3 Shared controller contract

Each agent is a function of observation and optional short history:

```text
Policy(observation, valid_actions, memory?) -> { action, optional_confidence, raw_response }
Environment.step(action, action_repeat) -> { next_observation, reward, terminal, info }
```

- All conditions get the **same permitted action semantics**, same action duration, same objective text, and comparable move budget.
- For the headline policy comparison, **validate** actions against the shared set. Separately report invalid/illegal-action rate. Do not silently repair choices using an oracle.
- For tic-tac-toe, report two modes: **(i)** all nine board cells (`R1C1..R3C3`) supplied as default across T1–T6 so candidate keys never leak occupancy in vision-only conditions (illegal moves trigger immediate forfeit loss in T6); **(ii)** only legal cells provided (T4 + legal-only T6 slice, a pure strategy/decision test).
- Any memory mechanism must be explicit and matched across policies. Base condition = no cross-step narrative memory beyond the observation window; secondary condition = identical compact action/observation history.
- Never leak terminal state, precise object coordinates, enemy positions behind walls, future velocity, or simulator internals to “visible-state” text conditions unless testing an **oracle privileged-state upper bound** explicitly.

## 5. Two evaluation tracks (essential)

### Track 1 — Offline action quality (latency-neutral)

Construct a **fixed corpus of identical decision points** for all agents, with screenshots and paired text representations saved in advance. Evaluate the proposed action against an oracle, expert label, or short rollout from exactly the same saved state. The environment does not advance while a model answers.

### Track 2 — Closed-loop gameplay (system quality)

Agents repeatedly observe, decide, and act through full episodes. Two timing modes:

1. **Paused/step-based:** Stop simulation during inference; advance a fixed number of game frames after the action. Measures gameplay quality independent of wall-clock inference speed.
2. **Real-time/asynchronous** *(secondary, only when feasible)*: Simulation continues while inference happens; actions may become stale. Measures practical responsiveness and latency sensitivity.

**Do not blend results from the two modes.** Distinguish *simulation frames elapsed* from *wall-clock seconds elapsed*.

## 6. Game 1 — Tic-tac-toe: rule understanding and representation

**Environment:** Deterministic 3×3 engine that enforces the rules and renders a consistent **512 × 512 PNG** screenshot (`R1C1`..`R3C3` keys with spatial criteria `"Top-left cell (row 1, column 1)"`, etc.; clean unlabeled grid by default with `"Player to move: X"` indicator). Use a minimax solver to label optimal actions. Balance starting player (`X` / `O`), move number, and board orientation.

**Canonical observation:**

```json
{
  "game": "tic_tac_toe",
  "board": ["X", ".", "O", ".", "X", ".", ".", ".", "O"],
  "player_to_move": "X"
}
```

Begin with `board` + `player_to_move`; test `move_history` as an explicit T5 ablation.

**Concrete experiments**

| ID | Experiment | Control | Measurement / lesson |
|---|---|---|---|
| **T1** | Fixed-position best-move test | Stratified board states: immediate win, forced block, fork, anti-fork, multiple equally optimal moves, draw | Optimal-move-set accuracy, regret, invalid moves |
| **T2** | Visual vs symbolic | A/B/C/D/E/F on same 240 held-out positions, 1 frame | Pixel-reading penalty, hybrid gain, model-family differences |
| **T3** | Rendering robustness | 4 perturbations (`labeled_margins`, `dark_low_contrast`, `thin_alt_font`, `tilt_blur`) on 80-state stratified slice (flag for 240) | Accuracy degradation; visual brittleness |
| **T4** | Action-choice schema | All 9 cells vs legal-cell choices only | Rule-following vs constrained decision-making |
| **T5** | Prompt representation | `json_canonical` vs `ascii_grid` vs `coord_list` vs `json_with_history` on 80-state stratified slice (flag for 240) | Sensitivity to representational formatting |
| **T6** | Full matches | 100 games vs random + 100 vs minimax per condition (balanced X/O), `all-9` (strict illegal forfeit) + `legal-only` slice | Win/draw/loss, non-loss rate, illegal-action rate, cost/game |

**Sample sizes:** `50` dev positions + `240` stratified symmetry-deduplicated held-out positions (≥2 empty cells, random D4 orientation per state); `80`-state slice for T3/T5; `100` matched games vs random and `100` vs minimax per condition (`--pilot` preset: `30` states / `20` games).

**Primary metric:** Fraction of moves in the *minimax-optimal action set* and minimax value regret from the current player's perspective.

## 7. Game 2 — Breakout: perception, memory, and reactive control

*(Milestone 2 — `ALE/Breakout-v5`, 420 × 320 padded screenshots, visible-derived vs privileged state, B1–B7.)*

## 8. Game 3 — ViZDoom: spatial control and longer horizons

*(Milestone 3 — `Basic` → `Defend the Center` → `Health Gathering`, synchronous `DoomGame`, D1–D7.)*

## 9. Cross-cutting ablations and controls

Run the minimum viable matrix first, then choose ablations that answer specific questions. **Change one dimension at a time.**

## 10. Metrics and analysis plan

- **Tic-tac-toe:** Optimal-action-set accuracy on held-out positions; minimax value regret; non-loss rate vs minimax; illegal-action rate.
- **Every game:** End-to-end milliseconds per decision (p50, p95), total cost per episode, and successful outcomes per dollar.
- **Statistical discipline:** Paired 95% bootstrap confidence intervals on fixed offline state sets; preserve all runs including errors/forfeits; separate `dev` from `test`.

## 11. Logging, reproducibility, and artifacts

Record a JSONL row per decision (`decisions.jsonl`), episode summaries (`episodes.jsonl`), `manifest.yaml`, and content-addressed PNG frames (`sha256` hashes).

## 12. Recommended execution order and cost gates

1. **Milestone 0 — Feasibility:** Pin models, verify `.env` API contracts, image acceptance, probability vectors, token usage, and 10 warm/cold latency calls under a strict budget gate (`$10.00` default cap per run).
2. **Milestone 1 — Tic-tac-toe harness:** Engine, renderer, minimax oracle, six adapters, T1–T6 runners, bootstrap CI analysis, and lightweight FastAPI + HTML/JS trace viewer.
3. **Milestone 2 — Breakout harness.**
4. **Milestone 3 — ViZDoom.**
5. **Milestone 4 — Analysis and write-up.**

## 13. Failure taxonomy for the write-up

1. **Perception** · 2. **State/time** · 3. **Decision quality** · 4. **Tool/contract** · 5. **Control** · 6. **Planning** · 7. **Calibration**.

## 14. Proposed final article outline

*Can Decision Models Play Games? What Tic-tac-toe, Breakout, and Doom Reveal About Multimodal Agents*

## 15. Decision log (locked v0.2 choices)

| Question | Locked Decision (v0.2 — Nov 14, 2026) |
|---|---|
| **Milestone 1 scope** | Full Milestone 0 + Milestone 1 (`T1`–`T6` complete + FastAPI/HTML trace viewer). |
| **Decision models (A/B/D)** | Generic `DecisionModelAdapter` with TypeSafe `jev-1.14.0` and Google `gemini-decision-1` (`v1beta`). |
| **Generative comparator (C/E/F)** | Generic `GenerativeAdapter` starting with `gemini-4-hydrogen` (`thinking_level="low"`, `store=False`), same model across C/E/F. |
| **Secondary conditions** | Strictly the 6 core conditions (`A–F`, 9 model-condition rows) + minimax oracle scoring in M1. |
| **Action & visual cues** | `R1C1..R3C3` keys with spatial criteria; clean unlabeled 512×512 PNG grid by default; margin labels tested in T3. |
| **Legality & T6 forfeits** | `all-9` default across T1–T6; `legal-only` in T4 + T6 slice. Illegal/invalid moves in `all-9` T6 forfeit immediately. |
| **Corpus & match scale** | `50` dev + `240` held-out test states; `80`-state slice for T3/T5; `100` vs random + `100` vs minimax (`--pilot`: `30` / `20`). |
| **Spending cap** | Hard stop at `$10.00` USD per run by default (`--max-cost-usd`). |
| **Execution sequence** | Milestone 0 + core engine/oracle/adapters first → live `.env` smoke test → T1–T6 runners & trace viewer. |
