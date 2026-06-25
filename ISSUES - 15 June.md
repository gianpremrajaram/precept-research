# Precept Dissertation Experiments - Implementation Backlog

> Two-agent LLM coordination under a degraded channel, measured by conditional usable information (CPVI) at the handoff. Derived from `RESEARCH_ROADMAP.md`. The release gate is a distinction-grade, reproducible result set, not the calendar.

---

## Overview

This is the canonical engineering backlog for the dissertation experiments. Scope: a reproducible research codebase that (1) simulates the transposed piano-movers task in Pymunk, (2) runs two LLM agents negotiating over a degradable channel in LangGraph against locally served open-weight models, (3) scores each handoff offline (CPVI) and online (a target-free statistic), and (4) can gate low-information handoffs and replicate the finding on real multi-agent failure logs. Tickets are sized for Claude Code to pick up end to end, each 5-10 hours of dev effort; larger pieces of work are pulled together rather than fragmented into sub-2-hour tickets.

**Relationship to the roadmap.** Every ticket traces to a section of `RESEARCH_ROADMAP.md`. The roadmap is the why and the experimental design; this backlog is the what-to-build and the acceptance criteria. **Numbering note:** the roadmap refers to the open compute decision as "DSE-014"; in this backlog that work is **DSE-005** (the benchmark harness that informs it), and **DSE-014** here is the CPVI estimator. This is the only cross-reference drift; the roadmap was deliberately not edited.

**Timeline philosophy.** The phase windows in the roadmap are targets; if a ticket reveals deeper complexity (Pymunk contact instability, vLLM guided-decoding drift, probe underfit), extend the window rather than ship a degraded result. The two hard dates are a late-August dissertation submission and a 21 September paper. The pilot gates (DSE-019) front-load the central feasibility risk so a pivot happens in week two, not week eight.

---

## Conventions

### Priority

- **P0**: dissertation-critical. A frozen result or the core system depends on it.
- **P1**: important. Strengthens the dissertation but is not load-bearing for the minimal claim.
- **P2**: optional / deferred. Upside arms with explicit cut-lines (roadmap §3.6).

### Effort sizing

- **S**: <= 4h. (Avoided here; folded into M tickets.)
- **M**: 4-8h (the default for this backlog).
- **L**: 8-10h (kept at or under the 10h ceiling; re-split if a ticket grows past 10h).
- **XL**: >10h. Not used; any XL is re-split before pickup.

### Issue type

- `infra`: environment, serving, CI, config, data versioning.
- `feat`: a code component (simulator, agent loop, estimator, gate).
- `experiment`: a driver that executes a sweep and produces a dataset/result.
- `analysis`: statistics, figures, calibration, hypothesis tests.
- `research`: a design decision with an executable deliverable (a benchmark, a calibration report) that informs a human choice.
- `docs`: documentation, reproducibility appendix.

### Phase map (aligned to roadmap §4)

| Phase | Theme | Epics |
| --- | --- | --- |
| 0. Foundation | repo, serving, config, data schema, model benchmark | E1 |
| 1. Pilot | simulator, agent loop, conditions, gate harness | E2, E3, E6 |
| 2. Measurement | CPVI stack, twins, runtime statistics, calibration | E4, E5 (partial) |
| 3. RQ1 | information-gradient sweep + analysis | E7 (RQ1) |
| 4. RQ2 + gate | twin/proxy analysis, gate operating point | E7 (RQ2), E5 |
| 5. RQ3b | causal gate experiment | E5, E7 (RQ3b) |
| 6. RQ3a + optional | external validity; optional arms | E7 (RQ3a), E8 |
| 7-8. Assembly + paper | analysis infra, reproducibility, exports | E9 |

### Python / stack baseline

- Python >= 3.11. Primary deps (floor-pinned, e.g. `pkg>=x,<x+1`): `pymunk`, `langgraph`, `langchain-openai`, `pydantic>=2`, `sentence-transformers`, `scikit-learn`, `numpy`, `pandas`, `scipy`, `statsmodels`, `hydra-core` (or `omegaconf`), `opentelemetry-api`, `pyarrow`, `datasets` (HuggingFace), `openai` (vLLM client).
- Serving: `vllm` (run on Myriad GPU nodes, not a hard dep of the analysis code).
- Dev: `pytest`, `pytest-cov`, `hypothesis`, `mypy` (strict on `src/preceptx/`), `ruff` (lint + format), `pip-audit`, `pre-commit`.
- Tests: pytest, coverage target 80%+ on `sim/`, `measure/`, `gate/`. LLM-dependent paths tested against a mock client or the 8B tier.

---

## Epic Map

| Epic | ID range | Priority | Phase | Issues |
| --- | --- | --- | --- | --- |
| E1. Foundation & Infrastructure | DSE-001 to DSE-005 | P0 | 0 | 5 |
| E2. Simulator (Pymunk) | DSE-006 to DSE-009 | P0 | 1 | 4 |
| E3. Agent Loop (LangGraph) | DSE-010 to DSE-012 | P0 | 1 | 3 |
| E4. Measurement Stack (CPVI) | DSE-013 to DSE-015 | P0 | 2 | 3 |
| E5. Runtime Gate & Causal Arm | DSE-016 to DSE-018 | P0 | 2 / 5 | 3 |
| E6. Pilot | DSE-019 | P0 | 1 | 1 |
| E7. Core Experiments | DSE-020 to DSE-025 | P0/P1 | 3-6 | 6 |
| E8. Optional Arms | DSE-026 to DSE-027 | P2 | 6 | 2 |
| E9. Analysis Infra & Reproducibility | DSE-028 to DSE-030 | P0/P1 | 7-8 | 3 |

**Total: 30 issues. Minimal-claim core: DSE-001 to DSE-025, DSE-028 to DSE-030 (28). Optional: DSE-026, DSE-027.**

---

## Target Repository Structure

```
precept-experiments/
├── pyproject.toml
├── README.md
├── CITATION.cff
├── configs/                      # hydra / omegaconf
│   ├── experiment/  condition/  model/  serialisation/  difficulty/
├── src/preceptx/
│   ├── sim/                      # arena, load, actions, state, serialise, outcomes
│   ├── agents/                   # langgraph graph, prompts, channel
│   ├── serving/                  # vllm client wrapper, model registry
│   ├── data/                     # handoff schema, dataset writer, otel capture
│   ├── measure/                  # featuriser, pvi_cpvi, twin, divergence
│   ├── gate/                     # statistics, calibration, precept_integration, controls
│   ├── experiments/              # rq1, rq2, rq3a, rq3b, optional drivers
│   └── analysis/                 # stats, figures, reproducibility
├── scripts/
│   ├── myriad/                   # SGE jobscripts: serve.sh, sweep.sh
│   └── benchmark_models.py
├── tests/  {unit, integration, fixtures}
├── data/                         # versioned handoff datasets (gitignored, hash-stamped)
└── docs/                         # observatory demo trace, reproducibility appendix
```

---

# E1. Foundation & Infrastructure (Phase 0)

## DSE-001: Repository scaffolding, environment, dependency pinning, CI

**Epic:** E1 **Type:** infra **Priority:** P0 **Effort:** M (6-8h) **Phase:** 0 **Dependencies:** none

### Context
Everything builds on a clean, reproducible package. This ticket establishes the `src/preceptx/` layout, floor-pinned dependencies, strict typing and linting, and a CI pipeline that proves the codebase installs and tests on every push. Reproducibility is a dissertation examination criterion, so the foundation must be strict from day one.

### Acceptance Criteria
- [ ] `pyproject.toml` (PEP 621), `src/preceptx/` layout, `pip install -e .[dev]` succeeds on Linux x86_64 and macOS ARM64
- [ ] Dependency floors pinned per the stack baseline; `data`/`serving` extras separated from core analysis deps so the analysis code installs without `vllm`
- [ ] `.github/workflows/ci.yml`: matrix Python 3.11/3.12 on ubuntu-latest; jobs `lint` (`ruff check`, `ruff format --check`), `typecheck` (`mypy --strict src/preceptx`), `test` (`pytest --cov`)
- [ ] `.pre-commit-config.yaml` with ruff, ruff-format, mypy (scoped), `detect-private-key`, pinned revs
- [ ] `.gitignore` covers Python artefacts, `data/` (datasets are hash-stamped, not committed), model caches
- [ ] README stub with setup, and a one-paragraph map to `RESEARCH_ROADMAP.md`

### Technical Notes
- Package name `preceptx` to avoid clashing with the Precept OSS core; the experiments depend on Precept as an external dep (for the OTel exporter and the contract layer) but live in their own repo.
- Keep `mypy --strict` from the start; retrofitting is expensive.
- Do not add Codecov; coverage as a CI artefact is sufficient at this stage.

### Testing Requirements
- CI green on an empty PR touching only the workflow
- A deliberate lint and a deliberate type error are caught in throwaway branches

### Out of Scope
- Release/publish automation (this is a research repo, not a package release)

### Definition of Done
- Branch merged, CI green on `main`, local install verified on both platforms

---

## DSE-002: Myriad SGE job scripts and vLLM serving harness

**Epic:** E1 **Type:** infra **Priority:** P0 **Effort:** L (8-10h) **Phase:** 0 **Dependencies:** DSE-001

### Context
The experiments call locally served open-weight models behind a vLLM OpenAI-compatible endpoint on Myriad GPU nodes. This ticket provides the SGE job scripts to serve any tier of the model ladder, a thin client wrapper the rest of the code imports, and health-check/teardown utilities. Serving is decoupled from the analysis code so the latter runs anywhere.

### Acceptance Criteria
- [ ] `scripts/myriad/serve.sh`: parameterised SGE jobscript requesting `-l gpu=N`, wall time, and project (Free or priority); loads CUDA module, activates venv, launches `vllm serve` with greedy decoding, fixed seed, pinned revision, and `--guided-decoding-backend xgrammar`
- [ ] Supports each ladder tier: bf16 for 8B/14B; `--quantization awq --tensor-parallel-size 2` path for 70B-AWQ; documented memory/GPU mapping per roadmap §0 table
- [ ] `src/preceptx/serving/client.py`: `LLMClient` wrapping an OpenAI-compatible client pointed at the local endpoint; methods for chat and for structured (JSON-schema) calls; configurable base URL, model, temperature=0, seed
- [ ] Health check (`/v1/models` reachable, a smoke completion) and a graceful-shutdown helper
- [ ] `docs/serving.md` documenting Free-vs-priority queue expectations and how to switch tiers

### Technical Notes
- Myriad is single-node; TP is capped by GPUs-per-node (<= 4). 70B-AWQ on 2x A100-40GB or 1x A100-80GB.
- The client must be model-agnostic so swapping tiers is a config change, not code.
- Claude Code can author and unit-test the client against a mocked endpoint; running on Myriad requires cluster credentials (human step) - mark this in the README.

### Testing Requirements
- `LLMClient` unit-tested against a mock server (httpx mock / responses): chat and structured calls parse correctly, retries on transient errors
- Manual: `serve.sh` launches the 8B tier and the health check passes (human, on cluster)

### Out of Scope
- Multi-node serving; autoscaling; non-vLLM backends

### Definition of Done
- Client merged with passing mock tests; serve scripts documented; one tier verified live on Myriad

---

## DSE-003: Config system, seeding, run manifests, and determinism harness

**Epic:** E1 **Type:** infra **Priority:** P0 **Effort:** M (6-8h) **Phase:** 0 **Dependencies:** DSE-001

### Context
Every run must be reconstructable from a config plus a manifest. This ticket adds a Hydra/OmegaConf config tree (condition, serialisation, difficulty, model, seed), centralised seeding, and a run manifest capturing git SHA, config hash, model revision, and library versions. It also adds a determinism harness that quantifies LLM run-to-run variance, since batched inference is not bit-exact and the thesis must report seed sensitivity honestly.

### Acceptance Criteria
- [ ] `configs/` tree with composable groups; a single `experiment` config selects condition x serialisation x difficulty x model x seed
- [ ] `src/preceptx/config.py`: typed config dataclasses validated on load; invalid combinations raise with a clear message
- [ ] Centralised `set_global_seed(seed)` seeding Python, NumPy, and torch; documented limits for LLM determinism
- [ ] `RunManifest` written per run: git SHA, config hash, model name+revision, dep versions, timestamp; persisted alongside outputs
- [ ] `scripts/determinism_check.py`: runs the same fixed-seed episode K times against a served model and reports action-agreement rate and outcome variance

### Technical Notes
- Manifests are the reproducibility backbone for the examiner appendix (DSE-030); keep the schema stable.
- The determinism harness output feeds the thesis limitations section; expect non-zero variance and report it rather than suppress it.

### Testing Requirements
- Unit: config composition produces expected typed objects; invalid combos rejected; manifest round-trips through JSON
- Integration: determinism check runs on a mock client deterministically and on the 8B tier produces a variance report

### Out of Scope
- A bespoke experiment-tracking server (manifests + parquet are sufficient)

### Definition of Done
- Configs, seeding, manifest, and determinism harness merged with tests

---

## DSE-004: Handoff dataset schema, structured logging, OTel capture, and versioning

**Epic:** E1 **Type:** feat **Priority:** P0 **Effort:** L (8-10h) **Phase:** 0 **Dependencies:** DSE-001

### Context
The per-handoff record is the backbone the measurement, gate, and experiment tickets all consume, so its schema must be fixed early and treated as a stable contract (the way Precept fixed its IR). This ticket defines the schema, a writer that persists episodes to versioned Parquet/JSONL, and the OpenTelemetry capture that emits each handoff through Precept's exporter.

### Acceptance Criteria
- [ ] `src/preceptx/data/schema.py`: a Pydantic `HandoffRecord` with at least: `episode_id`, `step`, `condition`, `serialisation`, `difficulty`, `model`, `seed`, `state` (structured physics dict), `state_str` (serialised), `message_raw`, `message_delivered` (post-channel), `action`, `pre_state`/`post_state`, `progress`, `success`, `collision`, `stuck`, and placeholders for the four `Y` labels (filled by DSE-009)
- [ ] `src/preceptx/data/writer.py`: append-safe writer to hash-stamped Parquet under `data/<dataset_hash>/`; an index file maps dataset hash -> config + manifest
- [ ] `src/preceptx/data/otel_capture.py`: emits each `HandoffRecord` as an OTel span/event via the Precept exporter; fail-open if the exporter is absent
- [ ] A `load_dataset(hash)` helper returning a pandas/pyarrow frame with typed columns
- [ ] Schema documented in `docs/handoff_schema.md`; downstream tickets import this schema, never redefine fields

### Technical Notes
- Treat the schema as versioned (`schema_version` field); a breaking change bumps it and the loader handles both.
- Keep the structured `state` and the `state_str` both, so serialisation A/B (DSE-008) is recoverable from the dataset.
- Do not store full prompts in the dataset (size); store message and state only, with prompt templates versioned in the repo.

### Testing Requirements
- Unit: record validates; writer round-trips; loader returns typed columns; OTel capture no-ops cleanly without an exporter
- Property: arbitrary valid records persist and reload identically

### Out of Scope
- A database; a remote dataset store (local hash-stamped files are sufficient)

### Definition of Done
- Schema, writer, loader, and OTel capture merged with tests; `docs/handoff_schema.md` published

---

## DSE-005: Model-ladder benchmark harness (informs the compute decision)

**Epic:** E1 **Type:** research **Priority:** P0 **Effort:** M (6-8h) **Phase:** 0 **Dependencies:** DSE-002

### Context
The open decision (roadmap §0, referred to there as DSE-014) is which Myriad allocation and model size to commit to. This ticket builds the executable that informs it: a harness that, given a served model, measures throughput, memory, JSON-schema adherence, and a quick task-capability smoke, and emits a comparison table across the ladder. The allocation decision itself is a human step; the evidence is automated.

### Acceptance Criteria
- [ ] `scripts/benchmark_models.py`: for a given served endpoint, measures tokens/sec, time-to-first-token, peak GPU memory (via `nvidia-smi` parse or vLLM metrics), structured-output adherence rate over N schema-constrained calls, and a 10-episode C0 capability smoke success rate
- [ ] Runs across the ladder tiers by pointing at successive endpoints; outputs a single CSV/Markdown comparison table
- [ ] Produces a short auto-generated recommendation note (which tier clears a capability floor at acceptable throughput on which GPU)
- [ ] Documented so the human can run it once per available allocation and decide

### Technical Notes
- The capability smoke reuses the simulator + agent loop once those exist; until then it runs against a fixed scripted scenario stub so the harness is buildable in Phase 0 and enriched later.
- This ticket is the bridge that lets the rest of the plan default to the 14B workhorse without blocking on the allocation decision.

### Testing Requirements
- Unit: metric parsing and table generation tested against captured fixture outputs
- Integration: full run against the 8B tier produces a populated table (human, on cluster)

### Out of Scope
- The allocation decision (human); fine-tuning any model

### Definition of Done
- Harness merged with tests; one comparison table generated; recommendation note produced

---

# E2. Simulator (Pymunk) (Phase 1)

## DSE-006: Arena and T-shaped load construction

**Epic:** E2 **Type:** feat **Priority:** P0 **Effort:** M (6-8h) **Phase:** 1 **Dependencies:** DSE-001

### Context
The physical substrate: a top-down, damped three-chamber arena joined by two configurable slits, and a T-shaped dynamic load whose rotation through the slits is the cognitive core of the task (roadmap §2.1).

### Acceptance Criteria
- [ ] `src/preceptx/sim/arena.py::build_arena(slit_width, geometry)`: `pymunk.Space` with `gravity=(0,0)`, `damping<1`, static `Segment` walls forming three chambers and two slit gaps; goal region defined in chamber three
- [ ] `src/preceptx/sim/load.py::add_T_load(space, pos, mass)`: one dynamic `Body` carrying two box `Poly` shapes forming a T; mass and moment summed; friction set
- [ ] Difficulty is parameterised by slit width (and optionally chamber spacing); a `make_scenario(difficulty)` returns a configured space + load + goal
- [ ] Deterministic per seed; a fixed scenario reconstructs identically

### Technical Notes
- Top-down + damping makes the puzzle quasi-static (matches the ant/human task where the load does not coast).
- Choose slit widths so the T must rotate to pass; expose an "easy" preset (wider) for the G1 capability gate.

### Testing Requirements
- Unit: walls and gaps placed at expected coordinates; goal region correct
- Physics sanity: at a wide slit the load passes under a scripted nudge; at a narrow slit it jams without rotation; determinism across repeated builds

### Out of Scope
- Rendering (a later optional ticket); the force-handle action interface (DSE-007 covers actions)

### Definition of Done
- Arena and load builders merged with geometry and physics-sanity tests

---

## DSE-007: Action interface and physics step

**Epic:** E2 **Type:** feat **Priority:** P0 **Effort:** M (6-8h) **Phase:** 1 **Dependencies:** DSE-006

### Context
Agents act through a discrete macro-action interface (the primary design) with a higher-fidelity force-handle interface behind a flag (roadmap §2.1). This ticket implements actions, the settle-step, and collision/stuck detection.

### Acceptance Criteria
- [ ] `src/preceptx/sim/actions.py::apply_macro_action(space, body, action)`: realises one of `N,S,E,W,ROT+,ROT-,WAIT` as an impulse/torque, then steps the space to settle under damping
- [ ] Alternative `apply_force_handles(space, body, force_a, force_b)` behind a config flag for the two-grip-point design
- [ ] `read_state(space, body)` returns COM (x,y), angle, linear/angular velocity, and contact flags
- [ ] `detect_stuck` and `detect_collision` from velocity and contact history
- [ ] Step parameters (substeps, dt, impulse magnitude, rotation increment) are config-driven and documented for stability

### Technical Notes
- Small dt with multiple substeps avoids tunnelling through thin walls; document the chosen values.
- The macro-action interface keeps the handoff clean (one agreed action per turn); the force-handle interface is the fallback if macro-actions prove too coarse for a gradient (DSE-019 G2).

### Testing Requirements
- Unit: each macro-action moves the load in the expected direction/rotation; WAIT is a no-op modulo settling; stuck/collision detection fires on scripted jams
- Property: applying inverse actions returns the load near its origin within tolerance

### Out of Scope
- Reward shaping; continuous control beyond the two defined interfaces

### Definition of Done
- Actions, step, and detectors merged with tests; stability parameters documented

---

## DSE-008: State representation and serialisation (numeric / grid / NL)

**Epic:** E2 **Type:** feat **Priority:** P0 **Effort:** M (5-7h) **Phase:** 1 **Dependencies:** DSE-006

### Context
How the physics state is written into the prompt is an experimental factor (the RoCo lesson that prompt formatting masquerades as spatial reasoning). This ticket implements three serialisers selectable by config (roadmap §2.1, §3.2 serialisation A/B).

### Acceptance Criteria
- [ ] `src/preceptx/sim/serialise.py::serialise(state, mode)` with modes `numeric` (typed tuples), `grid` (ASCII occupancy grid faithful to geometry), `nl` (natural-language description)
- [ ] Each serialiser is deterministic and total (handles all valid states)
- [ ] The grid serialiser's occupancy matches the true arena/load geometry within the grid resolution
- [ ] A `deserialise_check` (for the grid/numeric modes) confirms the serialised state recovers COM/angle within tolerance, guarding against information loss in the representation

### Technical Notes
- Keep serialisers pure and side-effect-free; they are called from the agent nodes and from the featuriser (DSE-013).
- The NL serialiser should be templated, not model-generated, so it is deterministic and reproducible.

### Testing Requirements
- Unit: each mode deterministic; grid occupancy correct on known states; deserialise-check within tolerance
- Property: serialisers never raise on valid states (including extreme poses)

### Out of Scope
- Learned/model-generated serialisations; image renderings of the state

### Definition of Done
- Three serialisers merged with tests; the serialisation config flag wired

---

## DSE-009: Outcome labeller and the four Y options

**Epic:** E2 **Type:** feat **Priority:** P0 **Effort:** L (8-10h) **Phase:** 1 **Dependencies:** DSE-006, DSE-004

### Context
CPVI predicts an outcome variable Y; the roadmap pins four options (binary progress, continuous displacement, discrete config, terminal success) and computes all four so the choice is a config flag (roadmap §2.4). This ticket builds the geodesic distance-to-goal, the success check, and the four labels, writing them into the handoff schema.

### Acceptance Criteria
- [ ] `src/preceptx/sim/outcomes.py`: geodesic distance-to-goal through the slits (waypoint graph through slit centres), `reached_goal(state)`, and per-step `progress` (signed distance reduction)
- [ ] Four `Y` labellers: `y_binary_progress` (net progress over next k steps), `y_continuous_displacement`, `y_discrete_config` (bucketed pose region), `y_terminal_success` (did the episode ultimately succeed from here)
- [ ] All four labels are written to `HandoffRecord` (DSE-004); the active Y for an analysis is a config selection
- [ ] `k` and the bucketing are config-driven and documented

### Technical Notes
- The geodesic must route through the slit gaps, not straight-line through walls; a small waypoint graph is sufficient.
- `y_terminal_success` requires the full episode, so labelling runs as a post-episode pass over the trajectory.

### Testing Requirements
- Unit: geodesic distance decreases monotonically along a scripted solving trajectory and increases when the load is pushed away; success fires only in the goal region
- Unit: each Y labeller returns expected values on hand-constructed trajectories

### Out of Scope
- Choosing the headline Y (that is an analysis decision in DSE-022, informed by the pilot)

### Definition of Done
- Outcomes and all four Y labellers merged with tests; labels populate the dataset

---

# E3. Agent Loop (LangGraph) (Phase 1)

## DSE-010: Two-agent episode graph with structured actions and prompts

**Epic:** E3 **Type:** feat **Priority:** P0 **Effort:** L (8-10h) **Phase:** 1 **Dependencies:** DSE-007, DSE-008, DSE-002

### Context
The negotiation loop: a LangGraph `StateGraph` with propose (A), respond (B), and apply nodes, looping to a step budget, with B's action emitted via guided JSON decoding (roadmap §2.2). The single A-to-B message is the handoff the rest of the system scores.

### Acceptance Criteria
- [ ] `src/preceptx/agents/graph.py`: `StateGraph` with nodes `agent_A` (emits a natural-language handoff), `agent_B` (emits a structured `Action` via guided decoding), `apply` (calls the simulator), and a conditional edge looping until success or budget
- [ ] `src/preceptx/agents/prompts.py`: versioned `PROMPT_A` (observe state, produce a handoff message) and `PROMPT_B` (observe state + message, choose an action) templates
- [ ] `Action` Pydantic schema enforced via the structured-output path of `LLMClient` (DSE-002)
- [ ] Per-step it records a `HandoffRecord` (DSE-004) capturing the message and pre/post state
- [ ] Runs end to end against a mock LLM and against the 8B tier; terminates correctly at success and at budget

### Technical Notes
- Keep the graph framework-thin so a LangGraph API change touches only this module (the roadmap's durability point).
- Guided decoding removes action-parser brittleness; if the model emits an invalid action despite the schema, default to WAIT and log it.
- The handoff capture here is where Precept's gate (DSE-018) later intercepts.

### Testing Requirements
- Unit (mock LLM): graph runs to completion, produces valid actions, loops to budget, terminates on success; invalid-action fallback works
- Integration (8B tier): a short episode completes and writes well-formed records

### Out of Scope
- The communication channel degradations (DSE-011); the gate (DSE-018)

### Definition of Done
- Graph, prompts, and structured-action path merged; mock and 8B integration tests pass

---

## DSE-011: Communication-channel module (conditions C0-C4, C5 stub)

**Epic:** E3 **Type:** feat **Priority:** P0 **Effort:** M (6-8h) **Phase:** 1 **Dependencies:** DSE-010

### Context
The degradation ladder applied only to the A-to-B channel, so any outcome difference is attributable to the channel (roadmap §2.3). C3 is the structural guard against the floor effect.

### Acceptance Criteria
- [ ] `src/preceptx/agents/channel.py::apply_channel(message, condition, ...)` implementing: C0 full; C1 length/token cap; C2 delayed delivery (message arrives one step late); C3 asymmetric visibility (B's observation restricted to a local window); C4 lexical/semantic noise (token dropout or paraphrase corruption)
- [ ] C5 supervisor-relay stub behind a flag (full implementation in DSE-026)
- [ ] C2 implemented via a one-step message buffer in the graph state; C3 implemented by masking B's observation, not the message text
- [ ] Each condition is config-selected; the delivered message is recorded as `message_delivered` (DSE-004)

### Technical Notes
- C3 changes B's observation (the serialised state window), not the message, which is what forces the message to carry non-state information.
- Keep degradations deterministic given a seed (e.g. seeded token dropout) for reproducibility.

### Testing Requirements
- Unit: each condition transforms message/observation as specified; C1 respects the cap; C4 dropout is seed-deterministic; C2 buffering delays by exactly one step
- Integration: a 2-step episode under C3 confirms B's observation is windowed

### Out of Scope
- The supervisor relay logic (DSE-026)

### Definition of Done
- All conditions merged with tests; C5 stub in place; deliveries recorded

---

## DSE-012: Episode runner and batch sweep executor

**Epic:** E3 **Type:** feat **Priority:** P0 **Effort:** L (8-10h) **Phase:** 1 **Dependencies:** DSE-010, DSE-011, DSE-004, DSE-003

### Context
The driver that runs N episodes over a config grid (condition x serialisation x difficulty x seed), writes the handoff dataset, and is resumable and parallelisable against the vLLM endpoint (roadmap §3). This is the workhorse the pilot and RQ1 call.

### Acceptance Criteria
- [ ] `src/preceptx/experiments/runner.py::run_grid(config)`: expands the grid, runs episodes, writes `HandoffRecord`s via the writer, and tags each with the run manifest
- [ ] Resumable: an interrupted sweep restarts without duplicating completed cells (idempotent on episode_id)
- [ ] Parallelism across the endpoint (bounded concurrency) without corrupting the dataset
- [ ] A run summary (cells, episodes, success rates, wall time) printed and persisted
- [ ] Runs a small grid end to end against the 8B tier

### Technical Notes
- Bound concurrency to the served model's throughput (from DSE-005) to avoid queueing collapse.
- Resumability keys on `(dataset_hash, episode_id)`; completed episodes are skipped.

### Testing Requirements
- Unit (mock LLM): a small grid produces the expected number of well-formed records; resume skips completed cells; concurrency does not duplicate or drop records
- Integration (8B tier): a small real grid completes and the dataset loads

### Out of Scope
- The per-RQ analysis (E7); the gate (DSE-018)

### Definition of Done
- Runner merged with mock + 8B tests; resumability and bounded concurrency verified

---

# E4. Measurement Stack (CPVI) (Phase 2)

## DSE-013: Embedding featuriser

**Epic:** E4 **Type:** feat **Priority:** P0 **Effort:** M (5-6h) **Phase:** 2 **Dependencies:** DSE-004

### Context
CPVI is computed on frozen embeddings of the serialised state and the message. This ticket provides a pinned, cached, swappable featuriser (roadmap §2.4), with encoder-swap support for the sensitivity check.

### Acceptance Criteria
- [ ] `src/preceptx/measure/featuriser.py`: embeds `state_str` and `message_delivered` with a pinned sentence-transformer (revision-pinned); returns aligned arrays keyed to `HandoffRecord`s
- [ ] On-disk cache keyed by (model revision, text hash) so re-runs are cheap
- [ ] Encoder is config-selectable (default a strong retrieval embedder; a second encoder available for the DSE-022 sensitivity check)
- [ ] Batch encoding; deterministic outputs per revision

### Technical Notes
- Pin the encoder revision explicitly; this closes a repo-audit gap and is required for reproducibility.
- Cache hits make probe re-fitting and twin analysis fast across the sweep.

### Testing Requirements
- Unit: deterministic embeddings per revision; cache hit returns identical vectors; shapes align with input records
- Performance: batch encoding of a fixture dataset completes within a generous bound

### Out of Scope
- Training the probes (DSE-014); the runtime statistics (DSE-016)

### Definition of Done
- Featuriser with caching and encoder-swap merged with tests; encoder revision pinned

---

## DSE-014: PVI/CPVI estimator and probe training

**Epic:** E4 **Type:** feat **Priority:** P0 **Effort:** L (8-10h) **Phase:** 2 **Dependencies:** DSE-013, DSE-009

### Context
The core construct: fit a state-only baseline probe and a state-plus-message probe and take the per-instance log-likelihood difference (CPVI), reporting the PVI-minus-CPVI gap and the AUROC uplift (roadmap §2.4). The conditioning on shared state is the novelty-defining move and must be implemented with strict train/held-out discipline.

### Acceptance Criteria
- [ ] `src/preceptx/measure/pvi_cpvi.py`: `fit_probe(X, y)` (L2 logistic default; a 2-layer MLP behind a flag); `pvi(...)` (unconditional, message vs null); `cpvi(...)` (conditional on state) returning per-instance scores
- [ ] Strict split discipline: probes fit on train, scored on held-out; no instance is scored by a probe it trained
- [ ] Reports the conditional V-information (mean CPVI), the per-handoff CPVI distribution, the `PVI - CPVI` gap, and the held-out AUROC of `g_cond` vs `g_base`
- [ ] Supports the binary Y (classification) and the continuous Y twin (Gaussian-NLL difference) per DSE-009
- [ ] A circularity guard test: on synthetic data where the message is pure noise, CPVI is ~0; where the message is informative beyond state, CPVI is clearly positive

### Technical Notes
- This is the methodological heart; keep it dependency-light and unit-test it hard on synthetic data with known ground truth.
- The MLP fallback is for underfit; document when to switch (logistic AUROC near chance despite a known signal).
- Numerical stability: clamp probabilities with an epsilon before the log.

### Testing Requirements
- Unit: noise-message -> CPVI ~ 0; informative-message -> CPVI > 0; PVI >= CPVI on state-correlated messages; split discipline enforced (a test asserts no train/test leakage)
- Property: CPVI is finite and well-defined across class distributions

### Out of Scope
- The twin and divergence proxy (DSE-015); the runtime statistics (DSE-016)

### Definition of Done
- Estimator merged with synthetic ground-truth tests; classification and continuous paths verified

---

## DSE-015: Retrospective/prospective twin and divergence proxy

**Epic:** E4 **Type:** feat **Priority:** P0 **Effort:** M (6-8h) **Phase:** 2 **Dependencies:** DSE-014

### Context
RQ2's measurement primitive: a retrospective CPVI (scored with the realised Y) and a prospective twin (the same trained probe applied at the handoff using only state+message, no Y at inference), plus a divergence proxy (JSD over probe predictive distributions and embedding cosine) (roadmap §2.4, §3.3). The no-Y-at-inference constraint is the circularity discipline and must be enforced in code.

### Acceptance Criteria
- [ ] `src/preceptx/measure/twin.py`: computes the retrospective CPVI and the prospective twin per handoff; the prospective path provably uses no realised outcome (a test asserts the function signature and execution path never touch Y)
- [ ] `src/preceptx/measure/divergence.py`: JSD between `g_cond` and `g_base` predictive distributions; embedding-cosine bridge statistic
- [ ] Agreement metrics: correlation and Bland-Altman between retrospective and prospective scores
- [ ] Outputs are joinable back to `HandoffRecord`s for downstream analysis

### Technical Notes
- The prospective twin is the same `g_cond` from DSE-014 applied at the handoff; the distinction is purely that no Y enters at inference.
- JSD over the probes' predictive distributions is the cheap bridge to the runtime proxy (DSE-016).

### Testing Requirements
- Unit: twin agreement is high on synthetic data where the prospective signal is informative; the no-Y test passes; JSD computed correctly on known distributions
- Property: Bland-Altman bias is near zero on matched synthetic twins

### Out of Scope
- The runtime gating decision (DSE-016, DSE-017)

### Definition of Done
- Twin and divergence merged with tests; the no-Y circularity test is the gating criterion

---

# E5. Runtime Gate & Causal Arm (Phases 2 / 5)

## DSE-016: Target-free runtime statistics (s_info, s_fail, s_cos)

**Epic:** E5 **Type:** feat **Priority:** P0 **Effort:** M (6-8h) **Phase:** 2 **Dependencies:** DSE-014

### Context
The gate cannot threshold CPVI (it needs the outcome); it thresholds a statistic computable at the handoff (roadmap §2.5). Three statistics, one of which (`s_cos`) is probe-independent and so answers the circularity objection.

### Acceptance Criteria
- [ ] `src/preceptx/gate/statistics.py`: `s_info` (the trained `g_cond` predictive entropy/confidence about Y given state+message at the handoff, no realised Y), `s_fail` (a dedicated failure-risk classifier predicting episode failure from state+message), `s_cos` (cosine between the message embedding and a reference embedding, probe-independent)
- [ ] Each statistic is computable from a `HandoffRecord` at the handoff with no access to the realised outcome (enforced and tested)
- [ ] `s_fail` training routine (fit on train split's failure labels) included; persisted with its manifest
- [ ] All three statistics are joinable to records for calibration (DSE-017)

### Technical Notes
- `s_info` legitimately uses the offline-trained probe; training needs Y, inference does not - this is the key distinction the thesis makes.
- `s_cos` is the only statistic independent of the fitted probes; it is the one that defeats a circularity objection.

### Testing Requirements
- Unit: each statistic computable without Y (a test asserts no outcome access at inference); `s_cos` is independent of probe state; `s_fail` trains and predicts
- Property: statistics are finite and bounded as expected

### Out of Scope
- Choosing thresholds (DSE-017); wiring the gate into the loop (DSE-018)

### Definition of Done
- Three statistics merged with tests; the no-outcome-access test passes

---

## DSE-017: Offline gate calibration

**Epic:** E5 **Type:** analysis **Priority:** P0 **Effort:** M (6-8h) **Phase:** 4 **Dependencies:** DSE-016, DSE-012

### Context
Choose the gate operating point by validating each runtime statistic against realised outcomes (the D10 circularity fix: calibrate against outcomes, not against CPVI) (roadmap §2.5, §3.3). Output is a persisted threshold and a calibration report.

### Acceptance Criteria
- [ ] `src/preceptx/gate/calibration.py`: for each statistic, computes AUROC for predicting failure/low-outcome on a held-out split, reliability curves, and ECE
- [ ] Selects an operating point per a documented rule (e.g. maximise a target metric subject to a firing-rate budget) and persists `(statistic, threshold, calibration_report)`
- [ ] The calibration is explicitly against realised outcomes, never against CPVI (asserted in a test and stated in the report)
- [ ] A one-page calibration report (figures + chosen threshold) generated

### Technical Notes
- The firing-rate budget matters for the matched-firing-rate control (DSE-018); record it.
- Report ECE so the thesis can speak to calibration honestly.

### Testing Requirements
- Unit: AUROC/ECE computed correctly on fixtures; threshold selection reproducible; the against-outcomes (not CPVI) constraint asserted
- Integration: calibration runs on a fixture dataset and emits a report

### Out of Scope
- The gated experiment (DSE-025)

### Definition of Done
- Calibration merged with tests; threshold and report persisted

---

## DSE-018: Precept gate integration and causal-arm controls

**Epic:** E5 **Type:** feat **Priority:** P0 **Effort:** L (8-10h) **Phase:** 5 **Dependencies:** DSE-017, DSE-010, DSE-011

### Context
Wire the calibrated statistic+threshold as a Precept contract that blocks low-information handoffs and re-prompts in the loop, and implement the two controls that make the causal claim valid (roadmap §2.5, §3.5). This is the active-control-layer contribution and the answer to the Lowe et al. critique.

### Acceptance Criteria
- [ ] `src/preceptx/gate/precept_integration.py`: a Precept contract/evaluator that, at the A-to-B boundary, computes the chosen statistic and blocks (re-prompts A) when below threshold; fail-open if Precept or the threshold is unavailable
- [ ] Re-prompt path: on block, A is re-prompted (bounded retries) before proceeding; the block and retry are recorded
- [ ] `src/preceptx/gate/controls.py`: a matched-firing-rate control (block the same number of randomly chosen handoffs) and a random-trigger control
- [ ] A config flag selects {gate-active, matched-random, random-trigger, off}
- [ ] Runs end to end against the 8B tier in each mode

### Technical Notes
- Reuse the Precept contract layer rather than a bespoke gate, so the dissertation demonstrates the OSS harness as an active control layer.
- The matched-firing-rate control must block the same count as the real gate on the same episodes, so it is computed from the gate's firing rate (DSE-017).

### Testing Requirements
- Unit (mock LLM): gate blocks below threshold and re-prompts; controls block the correct counts; mode selection works; fail-open verified
- Integration (8B tier): a short episode runs in each mode and records blocks/retries

### Out of Scope
- The full RQ3b sweep and analysis (DSE-025)

### Definition of Done
- Gate integration and controls merged; mock + 8B tests pass in all modes

---

# E6. Pilot (Phase 1)

## DSE-019: Pilot gate harness (G1/G2/G3) and fallback report

**Epic:** E6 **Type:** analysis **Priority:** P0 **Effort:** M (6-8h) **Phase:** 1 **Dependencies:** DSE-012, DSE-014

### Context
The three gates that decide whether the headline task is viable, with an auto-generated go/no-go report so the pivot decision is evidence-based and fast (roadmap §3.1). Failing a gate triggers the documented fallback ladder, not a scramble.

### Acceptance Criteria
- [ ] `src/preceptx/experiments/pilot.py` computing: G1 capability (C0 self-play success rate vs a configured floor), G2 signal (C0-vs-hard success/efficiency gap and a CPVI difference on a small sample), G3 groundedness (a hallucinated-geometry check that messages reflect the true state)
- [ ] An auto-generated one-page report with each gate's pass/fail, the numbers behind it, and the recommended action (proceed, retune once, or invoke fallback)
- [ ] Runs on a small sweep produced by DSE-012 against the 8B or workhorse tier
- [ ] The fallback ladder (elevate RQ3a; simplify the task; reframe as a diagnostic negative) is documented in the report template

### Technical Notes
- G3's hallucinated-geometry check can compare entities/coordinates mentioned in the message against the true state via the structured `state`.
- Keep the floors and gaps configurable; the pilot is allowed exactly one retune before a pivot, per the roadmap.

### Testing Requirements
- Unit: each gate computes correctly on fixture episodes; the report renders with pass/fail and actions
- Integration: the harness runs on a small real sweep and emits a report

### Out of Scope
- The full RQ1 sweep (DSE-020)

### Definition of Done
- Pilot harness and report merged with tests; one report generated on a small sweep

---

# E7. Core Experiments (Phases 3-6)

## DSE-020: RQ1 information-gradient sweep driver and analysis

**Epic:** E7 **Type:** experiment **Priority:** P0 **Effort:** L (8-10h) **Phase:** 3 **Dependencies:** DSE-012, DSE-014, DSE-028

### Context
The headline result: a factorial sweep over conditions C0-C4 x serialisation x difficulty, self-play with the workhorse, and a mixed-effects analysis of outcome on condition with CPVI as a mediator (roadmap §3.2). Tests H1 (degradation) and H2 (CPVI tracks it).

### Acceptance Criteria
- [ ] `src/preceptx/experiments/rq1.py`: assembles the factorial config, invokes the runner, and produces the analysis (success rate, steps-to-goal, collisions, mean and distribution of CPVI, PVI-minus-CPVI gap per condition)
- [ ] Mixed-effects model (via `statsmodels`) of outcome on condition with random effects for seed and episode; CPVI entered as a mediator to test H2
- [ ] Figures: outcome-vs-condition and CPVI-vs-condition with uncertainty intervals
- [ ] Runs end to end on a small fixture grid (mock or 8B); the full-scale run is gated on the resolved compute (DSE-005)

### Technical Notes
- Use the shared analysis library (DSE-028) for effect sizes, intervals, and multiple-comparison correction.
- Self-play is the primary cell; the heterogeneous and serialisation cells are DSE-021.

### Testing Requirements
- Unit: the factorial assembles the expected cells; the analysis reproduces on a fixture dataset
- Integration: a small grid runs and the analysis emits figures + a results table

### Out of Scope
- Robustness cells (DSE-021); RQ2 analysis (DSE-022)

### Definition of Done
- RQ1 driver and analysis merged with tests; runnable on a small grid; ready for the full run

---

## DSE-021: RQ1 robustness cells (heterogeneous pair + serialisation A/B)

**Epic:** E7 **Type:** experiment **Priority:** P1 **Effort:** M (6-8h) **Phase:** 3 **Dependencies:** DSE-020

### Context
Tests whether the gradient is a single-model or single-serialisation artefact (roadmap §3.2): a heterogeneous workhorse-vs-70B-AWQ cell and the numeric/grid/NL serialisation A/B.

### Acceptance Criteria
- [ ] A heterogeneous-pair configuration (A and B on different models) wired into the runner; one cell, not the full factorial
- [ ] A serialisation A/B analysis comparing outcomes and CPVI across numeric/grid/NL, with a recommendation for the serialisation used elsewhere
- [ ] Figures/tables comparing self-play vs heterogeneous and across serialisations
- [ ] Runs on a small grid; full run gated on compute

### Technical Notes
- The serialisation A/B quantifies how much apparent spatial reasoning is prompt formatting (the RoCo lesson) and chooses the representation for RQ2/RQ3b.
- The heterogeneous cell uses the 70B-AWQ tier; size it small given its cost.

### Testing Requirements
- Unit: heterogeneous config assembles correctly; serialisation comparison reproduces on a fixture
- Integration: a small heterogeneous grid runs

### Out of Scope
- Cross-family tie-breaker beyond one cell (only if time allows)

### Definition of Done
- Robustness cells merged with tests; serialisation recommendation produced

---

## DSE-022: RQ2 analysis (twin agreement and proxy tracking)

**Epic:** E7 **Type:** analysis **Priority:** P0 **Effort:** M (6-8h) **Phase:** 4 **Dependencies:** DSE-015, DSE-016, DSE-020

### Context
Tests the measurement primitive on the RQ1 episodes: H3 (retrospective-prospective agreement) and H4 (the runtime proxy tracks CPVI), plus the encoder-sensitivity check (roadmap §3.3). Output includes the chosen Y and encoder for the rest of the study.

### Acceptance Criteria
- [ ] `src/preceptx/experiments/rq2.py`: computes twin agreement (correlation, Bland-Altman) and proxy tracking (rank correlation of each statistic with CPVI; AUROC of each statistic for predicting low-CPVI and for predicting failure)
- [ ] An encoder-sensitivity comparison (default vs second encoder from DSE-013)
- [ ] A documented recommendation: the headline Y (from the four options) and the encoder, justified by the analysis
- [ ] Figures: twin agreement and proxy-vs-CPVI

### Technical Notes
- This is where the four Y options and the encoder choice are resolved empirically; record the decision in the report for the thesis.
- The proxy that best tracks CPVI and outcomes feeds the gate calibration (DSE-017).

### Testing Requirements
- Unit: agreement and tracking metrics reproduce on a fixture; encoder comparison runs
- Integration: the analysis runs on RQ1 outputs and emits figures + a decision note

### Out of Scope
- The gate experiment (DSE-025)

### Definition of Done
- RQ2 analysis merged with tests; Y and encoder decisions recorded

---

## DSE-023: RQ3a data loaders and handoff/step extraction (Who&When + MAST)

**Epic:** E7 **Type:** feat **Priority:** P0 **Effort:** L (8-10h) **Phase:** 6 **Dependencies:** DSE-004

### Context
External validity needs real failure logs parsed into the handoff schema. This ticket builds loaders for Who&When (primary, agent+step+why annotations) and MAST-Data (secondary, trace-level taxonomy), and extracts agent-to-agent handoffs plus surrounding state (roadmap §3.4). TRAIL is a documented fallback loader if needed.

### Acceptance Criteria
- [ ] `src/preceptx/experiments/rq3a_load.py`: loaders for Who&When (from the ag2ai repo / HuggingFace) and MAST-Data (HuggingFace), normalising both into `HandoffRecord`-compatible rows with their native annotations attached (Who&When decisive-error-step and responsible-agent; MAST failure-mode/category labels)
- [ ] Handoff/step extraction: identify inter-agent handoffs in each trace and the surrounding state/context that serve as the CPVI inputs
- [ ] A documented mapping from each dataset's structure to the common schema, including how "the decisive error step" maps to a handoff
- [ ] Loaders parse a sample of each dataset and report counts; a TRAIL loader stub is present behind a flag

### Technical Notes
- Who&When splits into Algorithm-Generated (short) and Hand-Crafted (long, 5-130 steps); handle both.
- The state/context for CPVI on real logs is the prior conversation/state visible to the receiving agent; document this construction carefully as it is a methodological choice.

### Testing Requirements
- Unit: loaders parse fixture samples of each dataset into valid rows; extraction yields handoffs with attached annotations
- Property: extracted rows conform to the handoff schema

### Out of Scope
- The CPVI localisation analysis and baselines (DSE-024)

### Definition of Done
- Loaders and extraction merged with tests on dataset samples; schema mapping documented

---

## DSE-024: RQ3a CPVI localisation, baselines, and human-agreement audit

**Epic:** E7 **Type:** experiment **Priority:** P0 **Effort:** L (8-10h) **Phase:** 6 **Dependencies:** DSE-023, DSE-014

### Context
Tests H5: does boundary CPVI localise the responsible step/trace better than schema validity, mean cosine, and the published Who&When attribution methods, under both probe-transfer and probe-refit regimes (roadmap §3.4). Labelling uses the released LLM-as-judge with a small human-agreement audit (the Q8 decision).

### Acceptance Criteria
- [ ] `src/preceptx/experiments/rq3a.py`: computes CPVI per extracted handoff under two regimes - (a) transfer the simulator-trained probe, (b) refit probes on a held-out portion of the logs - and reports both
- [ ] Localisation test: does low CPVI coincide with the Who&When decisive-error step / the MAST inter-agent-misalignment trace label, scored against baselines (schema validity, mean embedding cosine, and the three published Who&When methods: all-at-once, binary search, step-by-step)
- [ ] LLM-as-judge labelling pipeline (released annotator) with a small human-agreement audit on a sample, reporting kappa
- [ ] A results table comparing CPVI-based localisation to all baselines on the relevant metric (agent/step accuracy for Who&When; category prediction for MAST)
- [ ] Runs end to end on dataset samples

### Technical Notes
- Reporting both transfer and refit is the honest treatment of the methodological fork the roadmap flags; do not silently pick one.
- The published baselines are weak (53.5% agent, 14.2% step on Who&When), so clearing them is a genuine, reportable contribution.

### Testing Requirements
- Unit: the localisation metric and baseline comparisons reproduce on a fixture; the transfer and refit paths both run
- Integration: the analysis runs on dataset samples and emits a comparison table

### Out of Scope
- Full manual annotation (explicitly avoided per the design decision)

### Definition of Done
- RQ3a analysis merged with tests; transfer and refit results and the baseline comparison produced; audit kappa reported

---

## DSE-025: RQ3b causal-gate experiment and analysis

**Epic:** E7 **Type:** experiment **Priority:** P0 **Effort:** L (8-10h) **Phase:** 5 **Dependencies:** DSE-018, DSE-017, DSE-028

### Context
Tests H6: gating low-CPVI handoffs improves outcomes over the matched-firing-rate and random-trigger controls (roadmap §3.5). This is the interventional result and the demonstration that detection becomes enforcement.

### Acceptance Criteria
- [ ] `src/preceptx/experiments/rq3b.py`: runs a subset of RQ1 conditions in each mode {gate-active, matched-random, random-trigger, off} via the runner + gate integration, and analyses outcomes
- [ ] H6 test: gate-active beats both controls on success/efficiency with effect sizes and uncertainty intervals
- [ ] Figures: outcome by mode with intervals; the gate's firing rate and retry counts reported
- [ ] Runs end to end on a small grid; full run gated on compute

### Technical Notes
- The matched-firing-rate control uses the gate's measured firing rate so the comparison is fair (same number of blocks).
- A clean null (gating does not beat controls) is itself a reportable, honest result and should be presented as such.

### Testing Requirements
- Unit: mode assembly correct; the H6 comparison reproduces on a fixture
- Integration: a small grid runs in all modes and the analysis emits figures + a table

### Out of Scope
- Optional arms (E8)

### Definition of Done
- RQ3b driver and analysis merged with tests; runnable on a small grid; ready for the full run

---

# E8. Optional Arms (Phase 6, P2)

## DSE-026: C5 principal-agent supervisor arm

**Epic:** E8 **Type:** experiment **Priority:** P2 **Effort:** M (6-8h) **Phase:** 6 **Dependencies:** DSE-011, DSE-014

### Context
Tests the Rauba et al. (2026) assumption that aligned-incentive asymmetry is benign by routing the handoff through a supervisor relay and measuring residual agency loss (roadmap §3.6). First to cut if Phases 1-5 run late.

### Acceptance Criteria
- [ ] Full implementation of the C5 supervisor relay stubbed in DSE-011: a supervisor agent intermediates the A-to-B handoff under aligned incentives
- [ ] CPVI measured across the supervised boundary; residual agency loss (gap between intended and realised outcome) reported vs the unsupervised C0 baseline
- [ ] A small experiment + analysis comparing supervised vs unsupervised
- [ ] Clearly labelled optional in the README and the results

### Technical Notes
- Keep incentives aligned (the supervisor is cooperative); the incentive-divergence half is explicitly future work, not this ticket.

### Testing Requirements
- Unit (mock LLM): the relay routes and records correctly; CPVI computed across the supervised boundary
- Integration: a small supervised grid runs

### Out of Scope
- Adversarial/scheming supervisors (future work)

### Definition of Done
- C5 arm merged with tests; a small supervised-vs-unsupervised comparison produced

---

## DSE-027: SocialJax MARL comparison arm

**Epic:** E8 **Type:** experiment **Priority:** P2 **Effort:** L (8-10h) **Phase:** 6 **Dependencies:** DSE-028

### Context
Runs the information-gradient idea in a learned-message MARL setting for contrast with the LLM-text result (roadmap §3.6). First to cut; a contrast, not load-bearing.

### Acceptance Criteria
- [ ] A minimal SocialJax environment + MARL training loop where a learned message channel can be degraded analogously to C0-C4
- [ ] A measurement of information at the learned channel (an MI/TE estimator on the low-dimensional learned messages, where it is valid) and an outcome gradient across degradations
- [ ] A short comparison framing: learned-message MARL vs frozen-model LLM text, making the transferability/distinction explicit
- [ ] Clearly labelled optional

### Technical Notes
- MI/TE is valid here only because the learned messages are low-dimensional; do not apply it to LLM text (the dimensionality argument that killed MI for the main study).
- Keep this minimal; it exists to contextualise, not to extend, the LLM result.

### Testing Requirements
- Unit: the environment steps and the channel degradations apply; the information estimator runs on low-dim messages
- Integration: a short training run completes and produces a gradient

### Out of Scope
- A full MARL study; anything that competes for time with the core RQs

### Definition of Done
- SocialJax arm merged with tests; a minimal comparison produced; clearly marked optional

---

# E9. Analysis Infrastructure & Reproducibility (Phases 7-8)

## DSE-028: Shared analysis library, statistics plan, and seed-sensitivity

**Epic:** E9 **Type:** analysis **Priority:** P0 **Effort:** M (6-8h) **Phase:** 3 (built early, used throughout) **Dependencies:** DSE-004

### Context
A common analysis library so every RQ uses the same effect-size, uncertainty-interval, multiple-comparison, and seed-sensitivity machinery (roadmap §3 statistical plan). Built early because RQ1 onward depend on it.

### Acceptance Criteria
- [ ] `src/preceptx/analysis/stats.py`: dataset loaders over the handoff schema; effect-size and confidence/credible-interval helpers; multiple-comparison correction (Holm and Benjamini-Hochberg); a seed-sensitivity reporter (variance of results across seeds)
- [ ] `src/preceptx/analysis/figures.py`: consistent figure styling used by all RQ analyses
- [ ] A documented analysis protocol (which test for which hypothesis) matching the roadmap's statistical plan
- [ ] Helpers unit-tested against known inputs

### Technical Notes
- Centralising this prevents each RQ re-implementing statistics inconsistently and makes the thesis defensible.
- Seed-sensitivity reporting is required given LLM non-determinism (DSE-003).

### Testing Requirements
- Unit: effect sizes, intervals, and corrections correct on known inputs; seed-sensitivity reporter aggregates correctly
- Property: corrections never increase significance

### Out of Scope
- RQ-specific analyses (those live in E7 and import this)

### Definition of Done
- Analysis library and figure styling merged with tests; analysis protocol documented

---

## DSE-029: Reproducibility hardening

**Epic:** E9 **Type:** infra **Priority:** P1 **Effort:** M (5-6h) **Phase:** 7 **Dependencies:** DSE-013, DSE-004

### Context
Closes the repo-audit reproducibility gaps so the artefact is examiner-runnable (roadmap §6): pin the encoder revision, add `CITATION.cff` and a `.bib`, normalise the package layout, ensure run manifests are complete, and render a committed Precept observatory demonstration trace.

### Acceptance Criteria
- [ ] Encoder (and any model) revisions pinned in config and verified by a check that fails on an unpinned revision
- [ ] `CITATION.cff` and a project `.bib` added (the contact field is a human step, flagged)
- [ ] Package layout normalised (the `Dev/`-style layout removed); imports clean
- [ ] Run manifests confirmed complete on a sample run (git SHA, config hash, revisions, dep versions)
- [ ] A committed Precept observatory demonstration trace renders from a canned dataset

### Technical Notes
- These map directly to the repo-audit gaps and to dissertation reproducibility marks.
- The CITATION contact email is the only human step; everything else is automatable.

### Testing Requirements
- Unit: the unpinned-revision check fails as intended; manifest completeness asserted
- Manual: the observatory renders the demo trace

### Out of Scope
- A full public release of the experiments repo

### Definition of Done
- Hardening merged; revision check passes; CITATION/.bib present; observatory demo renders

---

## DSE-030: Thesis figure/table export and reproducibility appendix generator

**Epic:** E9 **Type:** docs **Priority:** P1 **Effort:** M (5-6h) **Phase:** 7 **Dependencies:** DSE-028, DSE-003

### Context
Scripts that export the final figures and tables for each RQ and auto-generate the reproducibility appendix (seeds, model revisions, jobscripts, config hashes), so the thesis assembly in Phase 7 is mechanical, not manual (roadmap §6).

### Acceptance Criteria
- [ ] `src/preceptx/analysis/export.py`: exports each RQ's headline figures and tables to a thesis-ready directory with stable filenames
- [ ] An appendix generator that compiles run manifests into a human-readable reproducibility appendix (what ran, with which model revision, seed, and config hash)
- [ ] Re-running the export reproduces identical artefacts from the same datasets
- [ ] Output directory structure documented for the thesis

### Technical Notes
- Stable filenames mean the thesis references do not break when figures are regenerated.
- The appendix generator reads the manifests from DSE-003, so it is fully automatable.

### Testing Requirements
- Unit: export produces the expected files from a fixture; the appendix generator compiles manifests correctly
- Integration: a full export from a sample result set reproduces deterministically

### Out of Scope
- Writing the thesis prose (human)

### Definition of Done
- Export and appendix generator merged with tests; one full export produced

---

# Claude Code Execution Guide

Signals: most tickets are fully agent-executable end to end against fixtures or the 8B tier; a minority produce artefacts a human interprets or require cluster credentials. Build order follows dependencies; the critical path to a minimal dissertation is the P0 chain through RQ1, RQ2, and one of RQ3a/RQ3b.

1. **Fully agent-executable (code + tests against fixtures/mock):** DSE-001, 003, 004, 006, 007, 008, 009, 010, 011, 012, 013, 014, 015, 016, 018, 023, 028, 030. These have deterministic acceptance criteria an agent can satisfy and verify without cluster access.
2. **Agent-executable build, human-gated run or input:** DSE-002 and DSE-005 (scripts authored and unit-tested by the agent; running on Myriad needs credentials), DSE-017 and DSE-019 (harnesses built and tested by the agent; the operating-point and go/no-go are human decisions informed by the auto reports), DSE-020/021/022/024/025 (drivers and analyses built and tested on small grids by the agent; the full-scale runs need the served model and compute, and result interpretation is human), DSE-029 (hardening automatable except the CITATION contact).
3. **Optional, defer unless ahead of schedule:** DSE-026, DSE-027.
4. **Critical path (build first):** DSE-001 -> 004 -> {006,007,008,009} -> 010 -> 011 -> 012 -> {013 -> 014} -> 019 (pilot gate) -> 020 (RQ1) -> {015,016} -> 022 (RQ2) -> 017 -> 018 -> 025 (RQ3b), with 023 -> 024 (RQ3a) running in parallel after 014, and 028 built alongside 020. DSE-005 and DSE-002 run in Phase 0 to resolve the compute decision before the full sweeps.
