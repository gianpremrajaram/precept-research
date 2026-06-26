# DEPENDENCIES.md

> Offline mirror of cross-ticket structure for **precept-research**. GitHub Issues is the backlog source of truth; this file exists so CLAUDE.md's section references (`§1` critical path, `§2` graph, `§3` runtime deps, `§4` risks, `§5` cross-cutting, `§8` gates) resolve without network. Read only the section you need. Regenerate the graph from the `**Dependencies:**` lines in `ISSUES.md` if it drifts.

---

## 1. Critical-path priorities

RQ priority is **RQ1 > RQ2 > (RQ3a ∥ RQ3b)**. The shortest path to a frozen RQ1 headline drives sequencing.

1. **DSE-001** (scaffold) — unblocks everything. Do first.
2. **DSE-002, DSE-003, DSE-004** — serving, config/manifest/determinism, handoff schema. Parallelisable once 001 lands. DSE-004's schema is a stable contract that the measurement, gate, and experiment tickets all import.
3. **Sim+agent spine:** DSE-006 → (007, 008) → DSE-010 → DSE-011 → DSE-012. Produces episodes and handoff records.
4. **Measurement spine:** DSE-009 (needs 006+004) and DSE-013 (needs 004) → DSE-014 (CPVI). DSE-028 (shared analysis) built early, used throughout.
5. **RQ1 (headline):** DSE-020 (needs 012, 014, 028). Freeze gate: Y and V frozen before this sweep (roadmap Phase 2).
6. **RQ2:** DSE-015, DSE-016 → DSE-017 (calibration) → DSE-022.
7. **RQ3b causal gate:** DSE-018 → DSE-025. **RQ3a external validity:** DSE-023 → DSE-024 (pre-planned fallback; can carry the dissertation alone).

Phase-0 immediate work order: **001 first**, then 002/003/004 in parallel, then 005 (needs 002).

---

## 2. Ticket dependency graph

Derived from each ticket's `**Dependencies:**` line. `A ← B` means B depends on A.

| Ticket | Depends on |
|---|---|
| DSE-001 | — |
| DSE-002 | DSE-001 |
| DSE-003 | DSE-001 |
| DSE-004 | DSE-001 |
| DSE-005 | DSE-002 |
| DSE-006 | DSE-001 |
| DSE-007 | DSE-006 |
| DSE-008 | DSE-006 |
| DSE-009 | DSE-006, DSE-004 |
| DSE-010 | DSE-007, DSE-008, DSE-002 |
| DSE-011 | DSE-010 |
| DSE-012 | DSE-010, DSE-011, DSE-004, DSE-003 |
| DSE-013 | DSE-004 |
| DSE-014 | DSE-013, DSE-009 |
| DSE-015 | DSE-014 |
| DSE-016 | DSE-014 |
| DSE-017 | DSE-016, DSE-012 |
| DSE-018 | DSE-017, DSE-010, DSE-011 |
| DSE-019 | DSE-012, DSE-014 |
| DSE-020 | DSE-012, DSE-014, DSE-028 |
| DSE-021 | DSE-020 |
| DSE-022 | DSE-015, DSE-016, DSE-020 |
| DSE-023 | DSE-004 |
| DSE-024 | DSE-023, DSE-014 |
| DSE-025 | DSE-018, DSE-017, DSE-028 |
| DSE-026 | DSE-011, DSE-014 |
| DSE-027 | DSE-028 |
| DSE-028 | DSE-004 |
| DSE-029 | DSE-013, DSE-004 |
| DSE-030 | DSE-028, DSE-003 |

Roots (no deps): **DSE-001**. Highest fan-out (most tickets blocked by it): **DSE-004** (blocks 009, 012, 013, 023, 028, 029) and **DSE-014** (blocks 015, 016, 019, 020, 024, 026).

---

## 3. Runtime dependencies

Floor-and-ceiling pinned in `pyproject.toml`; `uv.lock` is the reproducibility anchor. **Never add a runtime dep without updating `pyproject.toml`, regenerating `uv.lock`, and this section.**

**Core** (analysis + sim + agents; installs with no GPU, no `vllm`, no `torch`): `pydantic`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `pymunk`, `langgraph`, `langchain-openai`, `openai`, `hydra-core`, `omegaconf`, `pyarrow`, `opentelemetry-api`, `opentelemetry-sdk`.

**Extras** (kept out of core so the analysis path stays light and torch-free):
- `serving` → `vllm` (Myriad GPU nodes only; not needed by analysis code).
- `embed` → `sentence-transformers` (the only `torch` puller; consumed by the DSE-013 featuriser onwards). **Deviation from roadmap §"stack baseline":** the roadmap lists `sentence-transformers` among primary deps; it is isolated to an extra here so core installs/CI stay fast. The science still requires it — install `.[embed]` for the measurement stack.
- `data` → `datasets` (HuggingFace; RQ3a loaders, DSE-023).
- `dev` → `pytest`, `pytest-cov`, `hypothesis`, `mypy`, `ruff`, `pip-audit`, `pre-commit`, `respx` (mocks the OpenAI/httpx endpoint for serving tests).

Standalone constraint: **precept is NOT a dependency** and is never imported (CLAUDE.md). The OTel capture (DSE-004) and the runtime gate (DSE-018) are in-repo.

---

## 4. Risk register (condensed)

Full treatment lives in `RESEARCH_ROADMAP.md` §5; the load-bearing risks for sequencing:

- **R1 — LLM non-determinism.** Batched inference is not bit-exact. Mitigation: greedy decoding, fixed seed, pinned revision; report seed sensitivity, never claim exact reproducibility. Surfaces in DSE-003 (determinism harness) and every run manifest.
- **R2 — Capability floor (G1).** If self-play can't solve C0 above the floor, the gradient has no headroom. Mitigation: model-ladder benchmark (DSE-005), pilot gate (DSE-019). Failing G1 after one retune → fallback ladder (elevate RQ3a).
- **R3 — No measurable signal (G2).** No C0-vs-hard gap in outcome *and* CPVI. Mitigation: pilot before main sweep; RQ3a is the pre-planned fallback that carries the dissertation alone.
- **R4 — Groundedness (G3).** Messages must reflect true state, not hallucinated geometry. Mitigation: groundedness check in DSE-019.
- **R5 — Circularity.** Calibrating the runtime statistic against CPVI instead of realised outcomes invalidates the gate. Mitigation: calibrate on outcomes only; `s_cos` is probe-independent by construction.
- **R6 — Researcher degrees-of-freedom.** Re-selecting Y or V after seeing results is leakage. Mitigation: freeze Y and V before the RQ1 sweep (Phase-2 gate); pre-register hypotheses.
- **R7 — Compute allocation unknown (DSE-014 open decision).** Until the Myriad allocation is confirmed, default to the 14B workhorse, which fits every Myriad GPU.

---

## 5. Cross-cutting concerns

Touch every phase; owned nowhere single:

- **Determinism & pinning** — seed, model revision, encoder revision, resolved config recorded in every `manifest.json`. A run with an unrecorded revision is not a result.
- **The channel degrades one thing only** — `apply_channel` touches the A→B message and nothing else; outcome differences across C0–C4 must be attributable to the channel.
- **CPVI is always conditioned** — always report the `PVI − CPVI` gap; never message value without the state-only baseline.
- **The runtime statistic never sees the realised outcome** — computed at the handoff; calibrated offline against outcomes, never against CPVI.
- **Fail loud** — research code crashes visibly; named exceptions (`ConfigError`, `GateBlockedError`, `GroundingError`); no bare `except`.
- **Coverage gate ≥ 80%** on `sim/`, `measure/`, `gate/`, `runner` (the load-bearing core).

---

## 6. Coverage & test-tier dependencies

- **unit** (every commit, < 30s, no I/O) — includes the determinism tests (fixed-seed identical trajectory; CPVI known-answer fixture).
- **integration** (every PR, 1–3 min) — full `EpisodeRunner` + measurement stack on a tiny fixture and a stub LLM.
- **e2e** (manual) — real vLLM + LLM calls; needs Myriad/GPU.
- Property-based (`hypothesis`) for config schema, channel transforms, serialisers.

---

## 7. External data & model dependencies

- **Models** — open-weight ladder served via vLLM (roadmap §0): 8B pilot, 14B workhorse (default), 32B / 70B-AWQ robustness. Revisions pinned per run.
- **Encoder** — one pinned sentence-transformer for embeddings; computed once, cached by content hash, frozen before probes fit.
- **RQ3a datasets** — Who&When (primary), MAST-Data (secondary), TRAIL (fallback), via HuggingFace `datasets`. Loaders in DSE-023.

---

## 8. Phase & result-freeze gates

Hard go/no-go points (roadmap §4); timeline never overrides correctness.

- **Phase-0 gate** — DSE-014 compute decision resolved; CI green; serving stood up.
- **G1 capability / G2 signal / G3 groundedness** (Phase 1, DSE-019) — failing one after a single retune triggers the fallback ladder (elevate RQ3a), not a scramble.
- **Y/V freeze** (before the RQ1 main sweep, Phase 2) — outcome variable Y and probe family V chosen and frozen; re-selection afterwards is forbidden.
- **Result-freeze** — a result is frozen when its sweep is complete, manifest written, analysis run, effect sizes + intervals reported, figure/table committed. A frozen result changes only via an explicit re-freeze (CHANGELOG migration note).
