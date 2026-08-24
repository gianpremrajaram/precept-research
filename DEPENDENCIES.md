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
7. **RQ3b causal gate:** DSE-018 → DSE-045 → DSE-025. **RQ3a external validity:** DSE-041 → DSE-042 → DSE-024 (rescoped; pre-planned fallback that can carry the dissertation alone).

**Current work order (23 August 2026).** Phases 0–2 are built; the critical path is now short and specific.

1. **[Built, 24 Aug 2026 — retained for the reasoning; see ISSUES.md for current state.]** **Three small blockers stand between the repository and its first result** — ticketed as **DSE-031**, **DSE-032** and **DSE-033** — and none is large: a **console driver entry point** (`run_grid` / `run_pilot` / `run_rq1` are library functions and `pyproject` declares no `[project.scripts]`); a **structured-output mode for non-vLLM endpoints** (the client sends vLLM's `guided_json` in `extra_body`; an OpenAI-compatible local server expects `response_format.json_schema`); and a **pinned encoder revision** (currently unpinned and warning, which contradicts the project's own "a result with an unrecorded revision is not a result" rule). **DSE-049** (per-role clients on the runner) is batched with these: not itself a blocker for the first result, but cheaper before call sites accrete and the structural prerequisite for the heterogeneous cell (DSE-021).
2. **Before the Y/V freeze:** DSE-043 (control tasks and selectivity) and DSE-044 (repeated cross-fits and length control). Both change *what gets frozen*, so landing them afterwards would force a re-freeze.
3. **Before the RQ3b arm:** DSE-045 (gate retry feedback template) is **built** — it blocked DSE-018 because under greedy decoding an unchanged re-prompt is a fixed point, so the arm would have passed its unit tests while being vacuous live. `GATE_FEEDBACK_VERSION` is manifested but deliberately excluded from `dataset_hash_for` until DSE-018 makes retries live, at which point it must join the hash.
4. **In parallel from now:** DSE-041 (TraceElephant loader) then DSE-042 (replay labeller). This track has no dependency on the pilot, the sweep or the gate, and it is the fallback — verify it early.
5. **Docs-only, no dependency:** DSE-047 (re-anchor baselines and framing). **Analysis-only, after RQ1:** DSE-046 (absent-versus-unused decomposition).
6. **The remaining pre-freeze step is a run, not a build (24 August 2026).** Every ticket above is built; what stands between the repository and the Y/V freeze is the **bf16 re-gate of DSE-019 on Myriad**, submitted as `scripts/myriad/pilot.sh` (DSE-050). The local 4-bit G1 reading is indicative only — capability is a property of the substrate, and the verdict of record is bf16 on the cluster. `docs/myriad.md` §9 lists what must be confirmed in the first cluster session, since both jobscripts were written from UCL documentation rather than from a live box.

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
| DSE-041 | DSE-004 · *supersedes DSE-023* |
| DSE-042 | DSE-041 |
| DSE-043 | DSE-014 · **blocks the Y/V freeze** |
| DSE-044 | DSE-014, DSE-028 · **blocks the Y/V freeze** |
| DSE-045 | DSE-017 · **blocks DSE-018** |
| DSE-046 | DSE-020 |
| DSE-047 | — |
| DSE-048 | DSE-018, DSE-041 · stretch, post-freeze |
| DSE-049 | DSE-012 |
| DSE-050 | DSE-002, DSE-031 · **blocks every cluster run** |

Roots (no deps): **DSE-001**, **DSE-047**. Highest fan-out (most tickets blocked by it): **DSE-004** (blocks 009, 012, 013, 028, 029, 041) and **DSE-014** (blocks 015, 016, 019, 020, 024, 026, 043, 044).

**Status of superseded tickets.** DSE-023 (Who&When + MAST loaders) is superseded by DSE-041, which keeps a Who&When loader behind the same interface but demotes it to a flagged transfer-only anchor. DSE-024 is **rescoped** rather than superseded: same question, new substrate and a replay-defined outcome, so it now depends on DSE-041 and DSE-042 rather than DSE-023. DSE-027 (SocialJax) is **cut on evidence**, not deferred — see roadmap §3.6.

---

## 3. Runtime dependencies

Floor-and-ceiling pinned in `pyproject.toml`; `uv.lock` is the reproducibility anchor. **Never add a runtime dep without updating `pyproject.toml`, regenerating `uv.lock`, and this section.**

**Core** (analysis + sim + agents; installs with no GPU, no `vllm`, no `torch`): `pydantic`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `pymunk`, `langgraph`, `openai`, `hydra-core`, `omegaconf`, `pyarrow`, `opentelemetry-api`, `opentelemetry-sdk`. (`langchain-openai` was removed — it was declared but never imported; the code uses the raw `openai` client.)

**Extras** (kept out of core so the analysis path stays light and torch-free):
- `serving` → `vllm` (Myriad GPU nodes only; not needed by analysis code).
- `embed` → `sentence-transformers` (the only `torch` puller; consumed by the DSE-013 featuriser onwards). **Deviation from roadmap §"stack baseline":** the roadmap lists `sentence-transformers` among primary deps; it is isolated to an extra here so core installs/CI stay fast. The science still requires it — install `.[embed]` for the measurement stack.
- `data` → `datasets` (HuggingFace; RQ3a loaders, DSE-023).
- `viz` → `matplotlib` (renders calibration/analysis figures, DSE-017 onwards). The JSON report is the load-bearing artefact and all tests pass without it, so figures stay optional and CI stays light; `gate/calibration.py` imports matplotlib lazily inside the render path and skips with a log line when absent.
- `dev` → `pytest`, `pytest-cov`, `hypothesis`, `mypy`, `ruff`, `bandit` (security scan, CI job), `pip-audit` (weekly scheduled Action), `pre-commit`, `respx` (mocks the OpenAI/httpx endpoint for serving tests).

Standalone constraint: **precept is NOT a dependency** and is never imported (CLAUDE.md). The OTel capture (DSE-004) and the runtime gate (DSE-018) are in-repo.

---

## 4. Risk register (condensed)

Numbering follows `RESEARCH_ROADMAP.md` §5 and `docs/methodology.md` §10.2, which are the full treatments; this section carries only what bears on **sequencing** — which ticket a risk attaches to, and what it blocks. (This list previously used its own R-numbering, which collided with the roadmap's. It now mirrors the roadmap.)

- **R1 — Models cannot ground 2D geometry.** The most likely single failure. Attaches to DSE-005 (ladder benchmark) and DSE-019 (pilot gate). Failing G1 or G3 after **one** retune → fallback ladder, elevating RQ3a. This is why the RQ3a track must be buildable in parallel.
- **R2 — No measurable gradient (G2).** No C0-vs-hardest gap in outcome *and* CPVI. Attaches to DSE-019. The CPVI half of G2 is directional only until the pilot reveals the bit-scale; a positive floor is pre-registered after that.
- **R3 — CPVI floor effect.** Mean CPVI near zero because the receiver's state already predicts the outcome. Structurally guarded by C3 (DSE-011) and by receiver-conditioning; the `PVI − CPVI` gap is reported as a finding in its own right.
- **R4 — Probe overfit at pilot N.** 1,536-dim concatenated features against a few hundred handoffs. Attaches to DSE-014; the overfit monitor ships, and the capacity-reduction rule is pre-registered.
- **R5 — "CPVI is just word count".** Made structural by C1. Attaches to DSE-044; must land before the freeze. **Both pre-registered controls now ship**: length as a model covariate, and the overlap-restricted contrast (`RQ1Result.length_matched`), which compares Ck to C0 only inside length strata both conditions populate and reports `interpretable=false` rather than extrapolating where they do not overlap. The second is a sensitivity analysis, not a length-free effect estimate — PREREGISTRATION §5 fixes the framing so the write-up cannot upgrade it.
- **R6 — Circularity, and researcher degrees of freedom.** Calibrating the runtime statistic against CPVI invalidates the gate — `calibrate()` takes no CPVI argument so the error cannot be made by accident, and `s_cos` is probe-independent by construction. Re-selecting Y or V after seeing results is leakage — the Phase-2 freeze gate is the control.
- **R7 — Stale baseline framing.** Attaches to DSE-047 (docs only, no dependency). Do it before the RQ3a chapter is written, not after.
- **R8 — Probe selectivity.** Probe capacity manufacturing apparent information at pilot N; `auroc_train_cond` does not settle it. Attaches to DSE-043. **Blocks the Y/V freeze.**
- **R9 — Replay non-determinism.** Attaches to DSE-042. Bounded by majority vote over *n* replays, a reported agreement rate, an agreement floor, and a hard spend cap with a dry-run projection.
- **R10 — Online-auditing adjacency.** Pre-outcome failure detection now exists in the literature. Documentation risk only, no code dependency; addressed in `docs/methodology.md` §7 and DSE-047.
- **R-exec — LLM non-determinism (cross-cutting).** Batched inference is not bit-exact even at temperature 0 with a pinned seed. Mitigation: greedy decoding, fixed seed, pinned revision; report seed sensitivity, never claim exact reproducibility. Surfaces in DSE-003 and in every run manifest.

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

- **Models** — open-weight ladder served via vLLM (roadmap §0): Qwen3-8B pilot, Qwen3-14B workhorse (default), Qwen3-32B robustness, all Apache 2.0. Revisions pinned to commit SHAs per run. Any identifier in the ladder that has not passed DSE-005 is a placeholder and must be treated as one in the thesis.
- **Local pilot runtime** — an OpenAI-compatible local server (LM Studio or equivalent) serving the 8B tier at 4-bit for the free pre-cluster pilot. Free, not a Python dependency, and not added to `pyproject.toml`; it is an external tool recorded in the run manifest via the serving-substrate field. It exposes structured output through the OpenAI-standard `response_format.json_schema` rather than vLLM's `guided_json`, which is why the client needs a structured-output mode.
- **Encoder** — one pinned sentence-transformer for embeddings; computed once, cached by content hash keyed on encoder **name and revision**, frozen before probes fit. The revision is currently unpinned and must be pinned before the Phase-2 freeze.
- **RQ3a datasets** — **TraceElephant** `TraceElephant/TraceElephant` (CC-BY-4.0, primary: records `input_context` per step, ~380 executions of which ~220 annotated failures, ships runnable environments); **Who&When** `Kevin355/Who_and_When` (MIT, transfer-only anchor, rows flagged as reconstructed observation); **MAST-Data** `mcemri/MAST-Data` (CC-BY-4.0, trace-level secondary). All via the `data` extra. Loaders in DSE-041, superseding DSE-023. TRAIL and TELBench are scope-adjacent — largely single-agent scaffolds with no inter-agent handoff to score — and are cited, not loaded.
- **Counterfactual replay (DSE-042)** — re-executes third-party agent systems against TraceElephant's shipped environments. This is the only external dependency in the project that consumes a budget at run time; it runs under a hard spend cap with a dry-run projection produced first.

---

## 8. Phase & result-freeze gates

Hard go/no-go points (roadmap §4); timeline never overrides correctness.

- **Phase-0 gate — PASSED.** Compute decision resolved: UCL Myriad allocation approved 23 August 2026 (roadmap §0). CI green; serving harness authored but **not yet executed against a live endpoint**.
- **G1 capability / G2 signal / G3 groundedness** (Phase 1, DSE-019) — failing one after a **single** retune triggers the fallback ladder (elevate RQ3a), not a scramble. Run locally and free first, then re-gate on Myriad.
- **Y/V freeze** (before the RQ1 main sweep, Phase 2) — outcome variable Y and its horizon *k*, probe family V and its selection rule, the encoder and its pinned revision, the serialisation, the channel parameters, the jitter and seed count, the budgets, the analysis protocol, the gate feedback template version, the G1/G2/G3 thresholds, the cross-fit repeat count *R*, the control-task expectation and the length controls. Re-selection afterwards is forbidden. **Blocked on DSE-043 and DSE-044**, both of which change what the freeze covers.
- **Result-freeze** — a result is frozen when its sweep is complete, manifest written, analysis run, effect sizes + intervals reported, figure/table committed. A frozen result changes only via an explicit re-freeze (CHANGELOG migration note).
