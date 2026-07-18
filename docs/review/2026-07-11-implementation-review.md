# Implementation & Research-Direction Review — 11 July 2026

**Purpose.** Private working document tracking the precept-research implementation against the dissertation targets, with targeted improvement recommendations and next steps. Not for submission; not committed unless you choose to.

**Inputs reviewed.** `Thesis_Abstract.md`; `LitReview_Methodology_ExperimentalDesign-10 July.md` (treated as the target spec per your instruction, but challenged where a better option exists); `RESEARCH_ROADMAP.md`; `CHANGELOG.md`; `DEPENDENCIES.md`; open-ticket specs in `ISSUES.md` (DSE-005, 018, 021–025, 029, 030); every module in `src/preceptx/` read line-by-line for the correctness-critical spine (`measure/`, `gate/`, `sim/outcomes.py`, `agents/channel.py`, `agents/graph.py`, `experiments/rq1.py`, `experiments/pilot.py`) and fully for the rest; configs, CI, serve script, gitignore; targeted test greps. PR #41 was checked per issue #42 — **it has no comments of any kind via the API** (no reviews, no inline comments, no issue comments); close #42 or re-point it.

**Context assumed.** Hard submission 21 September 2026. Myriad expected within ~10 days (~21 July). Supervisors have not yet reviewed the plan in detail, so pre-freeze design changes are still cheap. Priority RQ1 > RQ2 > (RQ3a ∥ RQ3b).

---

## 1. Executive summary

**Verdict: the engineering is genuinely strong and ahead of schedule; the science has not started; and there are two P0 defects that would have invalidated or badly weakened the headline result had the sweep run tomorrow.** Both are cheap to fix now and expensive to fix after data exists. The single most valuable thing you can do this week is not more building — it is running the pilot on interim compute, after landing the P0 fixes.

The five things that matter most, in order:

1. **P0-1 — C3's CPVI conditions on the wrong state.** `HandoffRecord.state_str` always stores A's *full* serialised state; B's C3-restricted observation is never persisted, and the featuriser embeds the full state as `e_s`. CPVI for C3 therefore subtracts a baseline that already contains the goal/global information the message is supposed to uniquely carry — the floor-effect guard the whole design leans on (lit review §9.2) is arithmetically inert. Fix: condition on the **receiver's observation** (persist it; schema v2), and pre-register that semantics.
2. **P0-2 — the seed axis doesn't replicate anything for C0–C3.** `make_scenario(difficulty)` has a fixed start pose and goal; the LLM is greedy; the seed reaches only C4's dropout RNG. Different-seed episodes in C0–C3 are nominally identical trajectories, so "seeds for power" is pseudo-replication, the mixed model's seed random effect is degenerate, and the probe-training data collapses toward duplicated rows. Fix: seeded initial-pose jitter, recorded in the manifest.
3. **P0-3 — the model configs will not serve as written, and Qwen3's thinking mode is unhandled.** `Qwen/Qwen3-14B-Instruct` is (as of knowledge cutoff) not a HuggingFace repo id — Qwen3 dense models are `Qwen/Qwen3-14B` etc., with hybrid *thinking* enabled by default. Unhandled, A's `message_raw` becomes a chain-of-thought dump, C1's 8-token cap cuts mid-`<think>`, and greedy decoding in thinking mode is explicitly discouraged by Qwen. Fix ids, pin revisions, disable thinking via `chat_template_kwargs`, assert no `<think>` in messages.
4. **Pilot now, not after Myriad.** The whole plan's risk posture (G1–G3 in week 2, fallback ladder ready) is currently blocked on cluster access while ~19/30 tickets sit merged and idle. A rented A100 running your **own `serve.sh` vLLM stack** reproduces the Myriad serving semantics exactly for under ~£10 of compute; §9 gives the concrete plan. With Myriad ~10 days out, the pilot can be *done* (with one prompt-retune iterated) before the cluster arrives, converting Myriad time into headline sweeps rather than debugging.
5. **RQ3a has an unsolved methodological hole nobody has written down: what is Y on real logs?** CPVI needs a per-handoff outcome. Who&When is (essentially) a failure corpus, so trace-level Y is single-class; using the decisive-error annotation as Y is circular with the localisation claim. This must be designed before DSE-023 is built (§10 proposes three options and a recommendation).

Beyond these: ~16 P1 findings (construct mismatches between the runtime statistic's label and the headline Y, an embedding-cache key that would silently poison the encoder-sensitivity check, missing feasibility certificates for medium/hard, a fail-soft that can record a dead endpoint as a passing run, unmanifested labelling parameters, a stale frozen-protocol string) and a set of pre-freeze research-design opportunities (§8) that are free now and forbidden after the Y/V freeze.

---

## 2. Status dashboard

19 of 30 tickets implemented and merged (all GitHub issues remain open regardless of state — fine per your call). Two of the 19 carry deferred Myriad-only verification.

| Ticket | Status | Notes |
|---|---|---|
| DSE-001 scaffold, CI | ✅ done | Coverage gate still not enforced in CI (P2-1). |
| DSE-002 serving harness | ✅ built / ⏸ live check deferred | `serve.sh` + `LLMClient`; needs the on-cluster health check. |
| DSE-003 config/seed/manifest/determinism | ✅ built / ⏸ live check deferred | Real fixed-seed run pending serving. |
| DSE-004 handoff dataset | ✅ done | Atomic part writes; solid. |
| DSE-005 model-ladder bench | ❌ open | Now the natural first job on interim compute (§9). |
| DSE-006/007 arena, load, actions | ✅ done | No solvability certificate for medium/hard (P1-4). |
| DSE-008 serialisers | ✅ done | Grid has no legend (P1-5); isomorphism claim is approximate (RD-8). |
| DSE-009 outcome labeller | ✅ done | `y_discrete_config` is not an outcome (P1-10); end-of-episode censoring silent (P1-12). |
| DSE-010/011/012 graph, channel, runner | ✅ done | C3 observation not persisted (P0-1); `ServingError→WAIT` over-broad (P1-3). |
| DSE-013 featuriser | ✅ done | Cache key omits encoder name — cross-encoder poisoning (P1-16). |
| DSE-014 PVI/CPVI estimator | ✅ done | Leakage discipline is genuinely good. Pooled-fit decision unregistered (P1-15). |
| DSE-015 twin + divergence | ✅ done | Prospective twin is one-sided by construction — thesis-text item (RD-2). |
| DSE-016 runtime statistics | ✅ done | `s_info` predicts terminal success, not the headline Y (P1-1). |
| DSE-017 calibration | ✅ done | Deployment firing-rate drift to note for DSE-018 (P2-5). |
| DSE-018 gate integration + controls | ❌ open | **Design trap identified before build:** greedy re-prompt is a fixed point (P0-4). |
| DSE-019 pilot gates | ✅ done | G2 measures a different CPVI construct than RQ1 (P1-2); G3 heuristic gaps (P2-12). |
| DSE-020 RQ1 driver | ✅ done | No efficiency model (P1-11); per-handoff scores not persisted (P1-17); provenance gaps (P1-8). |
| DSE-021 robustness cells | ❌ open | Blocked on a per-role-client refactor nobody has ticketed (P1-14). |
| DSE-022 RQ2 analysis | ❌ open | Depends on RQ1 data. |
| DSE-023/024 RQ3a loaders + localisation | ❌ open | **Blocked on the Y-on-logs design decision (§10), not on compute.** |
| DSE-025 RQ3b experiment | ❌ open | Depends on DSE-018. |
| DSE-026 C5 / DSE-027 SocialJax | ❌ open (optional) | Recommend formally deferring SocialJax now (§13). |
| DSE-028 analysis library | ✅ done | `ANALYSIS_PROTOCOL["H2"]` stale vs the shipped mediation (P1-7). |
| DSE-029 repro hardening | ❌ open | No `.gitattributes` LFS allowlist, no `CITATION.cff` yet — both expected here. |
| DSE-030 figure/appendix export | ❌ open | Fine to leave for Phase 7. |

**Phase reality vs roadmap:** by the roadmap clock this is ~week 4 (Phase 2–3 territory); in practice Phase 1 (pilot) has not run. The saving grace: Phases 1–4's *code* is built, so once serving exists the science phases compress. With the hard deadline now 21 September (not late August), there is real slack — but only if the pilot lands before or with Myriad.

---

## 3. Tracking matrix — dissertation commitments vs implementation

Every load-bearing claim in the abstract and methodology §8–9, against what exists. ✅ implemented and faithful; ⚠️ implemented with a drift or caveat; ❌ missing.

| Dissertation commitment (source) | Status | Where / gap |
|---|---|---|
| CPVI = log-loss reduction of state+message probe over state-only probe, per handoff (§8) | ✅ | `measure/pvi_cpvi.py::cpvi`; cross-fitted, episode-grouped. |
| Report unconditional PVI and the PVI−CPVI gap alongside (§8) | ✅ | `estimate`, `rq1.py::ConditionSummary.pvi_cpvi_gap`. |
| Binary progress Y + continuous twin; four candidates (§9.3) | ⚠️ | All four labelled, but `y_discrete_config` is the *current* chamber — a state feature, not an outcome (P1-10); horizon k=3 unmanifested (P1-6). |
| Probe family: L2 logistic + heteroscedastic regressor (§9.3) | ⚠️ | Logistic ✅; continuous twin is homoscedastic with the deviation recorded — decide before freeze: implement or amend text (D-3). |
| Conditioning on the **shared** state; C3 forces non-state information (§9.2) | ❌→fixable | `e_s` is always A's full state; C3's restricted observation never persisted. **P0-1.** |
| Retrospective/prospective twin agreement, H3 (§9.4) | ⚠️ | Implemented as per-instance KL(g_cond‖g_base) — defensible and elegant (it is E[CPVI] under the probe's own belief) but not what §9.4's text says, and it is non-negative while retrospective CPVI is signed (RD-2, D-2). |
| Three runtime statistics: confidence/entropy, failure-risk, cosine (§9.5) | ⚠️ | All three in `gate/statistics.py`; but the abstract names **JSD** as statistic one, the methodology names **entropy** — three-way naming drift with code (D-1); and `s_info` predicts terminal success rather than the headline Y (P1-1). |
| Calibration against realised outcomes, never CPVI (§9.5) | ✅ | `gate/calibration.py`; the target is a literal `"realised_failure"`, the entry point has no CPVI parameter, and a quantitative test pins the leakage delta. Genuinely well done. |
| Gate blocks + re-prompts before the receiver acts (§9.5) | ❌ open (DSE-018) | Design trap documented before build: greedy re-prompt reproduces the identical message (P0-4). |
| Five conditions C0–C4, one degradation ladder, channel touches only the message/observation (§9.2) | ✅ | `agents/channel.py`; clean. C4 = dropout only — "paraphrase" in §9.2 unimplemented (D-4); C1 cap default 8 vs the text's "a few tokens"/roadmap's ≤30 — pre-register the value (P2-10). |
| Serialisation A/B numeric/grid/NL as an experimental factor (§9.6) | ⚠️ | Serialisers ✅; but the RQ1 analysis has no serialisation term and `rq1_sweep` defaults to numeric-only — the A/B lives in open DSE-021; grid ships without a legend (P1-5). |
| Self-play primary + heterogeneous-pair cell (§9.2, §9.6) | ❌ | Runner drives both roles from one client; per-role clients needed (P1-14). |
| Mixed-effects model, seed+episode random effects, CPVI mediator (§9.9) | ⚠️ | Implemented (LPM MixedLM + episode-level Baron-Kenny with bootstrap CI) — but seed replication is currently vacuous (P0-2), efficiency has no inferential model (P1-11), and the frozen protocol string still describes the old test (P1-7). |
| Effect sizes + intervals, Holm/BH, seed sensitivity (§9.9) | ✅ | `analysis/stats.py`; BCa bootstrap; seed sensitivity tracks the C0-vs-hardest gap (good design). |
| Pilot gates G1/G2/G3 + fallback ladder (§roadmap 3.1) | ✅ | `experiments/pilot.py`; G2's CPVI construct differs from RQ1's (P1-2); G3 is a numbers-only heuristic with known blind spots (P2-12). |
| Determinism = seed-pinned, revision-pinned, low variance; never exact (§9.9) | ⚠️ | Client/harness ✅; but model revisions are `main` TODOs, encoder revision unpinned (warned), pymunk version untracked in manifests (P1-9). |
| Who&When primary / MAST secondary external validity (§9.8) | ❌ open | Loaders unbuilt — and blocked on the undefined Y-on-logs (§10), which the methodology itself does not specify. |
| Matched-firing-rate + random-trigger controls (§9.8) | ❌ open (DSE-018/025) | Firing rate is recorded by calibration for exactly this purpose ✅. |
| Pre-registered hypotheses and analysis plan (§9.9) | ❌ | No artefact exists. §11 provides the skeleton. |

---

## 4. P0 findings — fix before any recorded run

### P0-1. C3 CPVI conditions on the full state, defeating the asymmetric-visibility design

**Where.** `agents/graph.py:180` (`state_str=state["state_str"]` — always the full serialisation), `data/schema.py` (no `observation` field), `measure/featuriser.py:137` (`e_s = embed(state_str)`), `agents/channel.py:78–97` (`_restrict` produces B's window, which is then dropped after prompting).

**Why it matters.** The construct is "what the message adds *beyond what the receiver already has*". In C0/C1/C2/C4, B sees the full state, so conditioning on `state_str` is right. In C3, B sees a window; the message must carry the goal and global layout. Conditioning on the *full* state means `g_base` already knows the goal — so a perfectly informative message scores ≈ 0 added bits, C3's CPVI will not exceed C0's, H2's strongest cell goes flat, and the "the conditional score cannot be near zero by construction" argument (lit review §9.2) is false as implemented. This is exactly the kind of silent construct error that survives every unit test (all of which use full-visibility fixtures) and only surfaces as a confusing null after the sweep.

**Fix (recommend all three).**
1. Persist B's delivered observation on the record: add `observation: str` to `HandoffRecord`, bump `SCHEMA_VERSION` to 2, extend `_ARROW_SCHEMA` (`data/writer.py:35`). Do it now, while zero real datasets exist — this is the cheapest schema change you will ever make. Bundle the DSE-018 gate fields (`gate_blocked: bool`, `gate_retries: int`, `message_blocked: str | None`) into the same bump so the schema changes once, not twice.
2. Point the featuriser's state input at the receiver's observation (`Featuriser.featurise` reads `r.observation`). For C0/C1/C2/C4 it equals `state_str`, so nothing changes elsewhere.
3. Pre-register the semantics explicitly: *"the conditioning state s is the state observable to the receiver at the handoff; under C3 this is the windowed view, by design."* Add the honest caveat that C3's CPVI is therefore against a different (smaller) baseline than C0–C2/C4 — that is the point of the condition, but a reviewer will ask, and the cross-condition comparison of CPVI *levels* involving C3 should be framed as "against the receiver's own baseline", not as same-baseline magnitudes.

**Also worth doing:** keep `state_str` (the full state) on the record too — you then get, for free, a C3-only diagnostic contrast: CPVI conditioned on B's window (the construct) vs conditioned on the full state (how much of the message was reconstructable from the world). That pair is a genuinely nice thesis figure.

### P0-2. The seed axis is not replication for C0–C3

**Where.** `sim/arena.py:87` (`make_scenario(difficulty)` — fixed start `(chamber_w/2, slit_y)`, angle 0, fixed goal), `agents/graph.py:141` (seed reaches only the C4 dropout RNG), `serving/client.py:44` (temperature 0), `experiments/sweep.py` ("one episode per cell, replication carried by the seed axis").

**Why it matters.** With greedy decoding, a fixed scenario, and deterministic physics, two C0 episodes at seeds 0 and 7 issue *identical prompts at every step* — the only divergence source is uncontrolled server-side batching noise, which is exactly the noise you have promised not to lean on. Consequences: (a) power claims built on "seeds" are pseudo-replication; (b) the H1 model's seed random intercept has no true variance to absorb; (c) probe training sees near-duplicate rows — the CPVI estimator's effective N is a fraction of nominal, and `StratifiedGroupKFold` folds become copies of each other; (d) G2's requirement of both outcome classes in the C0∪hard subset can fail spuriously (all C0 replicas succeed identically). DSE-006's own acceptance criterion said "deterministic per seed", which implies seeded variation was intended and got lost in implementation.

**Fix.** Seeded initial-condition jitter, threaded from `cell.seed`:
- `make_scenario(difficulty, *, rng)` jitters the start pose within a safe region of chamber 1 (e.g. x ∈ [1.2, 2.8], y ∈ [1.5, 4.5], θ ∈ [−π/2, π/2)) with a collision-free rejection check, and (optionally, second knob) jitters the goal y.
- Record the jitter distribution parameters in `SweepConfig` (hence `sweep_hash` and the manifest) and the realised start pose per record (it is already in `pre_state` of step 0 — sufficient).
- Keep physics/geometry (slit widths, arena) fixed — jittering the *pose* varies the problem instance without confounding difficulty.

This one change simultaneously fixes replication, probe-data diversity, class balance for G2, and makes "random effects for seed and episode" mean what the methodology says. It is ~40 lines plus tests. Do it before the pilot; the pilot's own G-gate statistics need it.

### P0-3. Model ids and Qwen3 thinking mode will break serving day one

**Where.** `configs/model/*.yaml` (`Qwen/Qwen3-14B-Instruct`, `-32B-Instruct`, `-8B-Instruct`), `serve.sh` default, roadmap ladder (`Llama-3.3-70B-Instruct-AWQ-INT4`, "Gemma-4", "Mistral Small 4"), `serving/client.py` (no `chat_template_kwargs` support).

**Why it matters.** As of knowledge cutoff, Qwen3 dense checkpoints are `Qwen/Qwen3-14B` (no `-Instruct` suffix); the configured ids will 404 at `vllm serve`. More insidious: Qwen3 defaults to *hybrid thinking* — chat completions emit reasoning (`<think>…</think>` or a parsed reasoning field depending on vLLM version/flags). If unhandled, A's `message_raw` is a CoT dump: the C1 8-token cap truncates mid-reasoning, C4 drops tokens from the reasoning rather than the instruction, message-length and CPVI become confounded with visible deliberation, and G3's number-matching will fish numbers out of the thinking trace. Qwen's own model card advises against greedy decoding in thinking mode (degeneration/repetition) — your determinism doctrine (temperature 0) is only compatible with **non-thinking** mode.

**Fix.**
1. Verify the exact repo ids and pin commit-SHA revisions (this is precisely DSE-005's job — run it on interim compute, §9). If a 2507-style `-Instruct` refresh of the 14B has shipped since my knowledge cutoff, prefer it (thinking off by default) and pin it.
2. Add `chat_template_kwargs: dict[str, Any]` to `ServingConfig`, passed via `extra_body` on both `chat` and `structured`, set to `{"enable_thinking": false}` for Qwen3.
3. Add a cheap guard where the message is captured (`agents/graph.py::agent_a`): assert/strip `<think>` blocks, and fail loud if the raw message starts with reasoning markup — a passing-looking run with CoT messages is a category error, not a degraded mode.
4. Treat the rest of the ladder as unverified until DSE-005 prints a table: the 70B-AWQ id is a placeholder, "Gemma-4"/"Mistral Small 4" are speculative names. None of this blocks the pilot (8B/14B only).

### P0-4. (Design-ahead for DSE-018) The gate's re-prompt is a fixed point under greedy decoding

Not yet built — flagging **before** it is, because the ticket as written will pass its unit tests and still be vacuous live. On block, DSE-018 re-prompts A. But A is deterministic: same prompt → same message → same statistic → blocked again, for all bounded retries. The "block and rewrite" intervention would then reduce to "block and act on the stale/absent message", and H6 tests nothing about rewriting.

**Fix (pre-register it, since it is part of the intervention):** the retry prompt must differ — append explicit gate feedback to A's prompt (e.g. "Your previous instruction was blocked as low-information. State the push direction, whether rotation is needed and which way, and the goal's direction explicitly."). Record `gate_retries` and the blocked message(s) on the record (bundled into the P0-1 schema bump). Decide and freeze the feedback template before RQ3b; it is as much a part of the treatment as the threshold. An alternative is nonzero temperature on retries only — I recommend against it (it breaks the determinism story mid-episode); the feedback-injection route keeps greedy decoding and changes the input instead.

---

## 5. P1 findings — fix before the phase that consumes them

**P1-1. `s_info` predicts terminal success; the headline Y is next-k progress.** `gate/statistics.py:56–60` (`outcome_label` → `y_terminal_success`, self-annotated as a pilot shortcut) vs `experiments/rq1.py:177` (CPVI's Y = `y_binary_progress`). H4 ("the proxy tracks the offline CPVI") is then a cross-construct comparison — entropy about episode fate vs information about local progress — and any weak tracking result is uninterpretable (construct mismatch vs genuine failure). At Y-freeze, point `InfoStatistic.label` at the frozen Y. One line plus tests.

**P1-2. G2's CPVI is also computed against terminal success.** `experiments/pilot.py:170` builds per-handoff labels from episode success, while RQ1 uses progress. The pilot gate that certifies "a measurable CPVI gap exists" should certify it for the construct the headline will use. Switch G2 to `y_binary_progress` (already labelled by the time the gate runs) or record an explicit justification.

**P1-3. `ServingError → WAIT` can record a dead endpoint as a passing run.** `agents/graph.py:160–163` catches `(ServingError, ValidationError)` and defaults to WAIT. A network outage or OOM-killed server mid-sweep then produces fully-recorded episodes of WAITs — the "passing-looking broken run" CLAUDE.md names as the worst outcome. The ticket sanctioned WAIT for *invalid actions*, not for transport failure. Fix: let transport-level `ServingError` propagate (the runner already fails loud); keep WAIT only for `ValidationError` (and, if you want, a distinct `SchemaError` subclass for JSON-shape failures).

**P1-4. No solvability certificate for medium/hard; the step budget is untested against them.** Only easy has a scripted solve (`tests/unit/agents/test_graph.py:74`, 7 east pushes); hard has only a jam test. If hard is not solvable via the 7-action interface within budget, G1/G2 will read as "models can't do it" (R1) when the truth is "nobody can" — the most expensive misdiagnosis available. One E-push moves ≈ 1 unit (impulse 3, mass 1, damping 0.2, 0.5 s settle), so easy already needs ~7–8 of 12 budgeted steps; hard adds rotate-thread-rotate. Fix: a breadth-first search over macro-actions on the deterministic sim (cheap — the state space is tiny at this resolution) proving a solution exists per difficulty and reporting its length; set `max_steps` per difficulty at ~2–3× the found optimum (the default 12 in `sweep.py:41` is almost certainly too small for hard). This doubles as a beautiful thesis statistic: "optimal path length k vs LLM path length".

**P1-5. The grid serialisation has no legend.** `sim/serialise.py::_grid` emits raw `T/G/#/.` rows; neither serialiser nor `prompts.py` explains the symbols. B is asked to act on an unexplained ASCII matrix — the serialisation A/B would then measure legend absence, not representation. Add a constant legend/axis line (constant text across cells preserves the information-isomorphism argument).

**P1-6. `StepConfig` and `OutcomeConfig.k` are unmanifested.** `EpisodeRunner` defaults them (`experiments/runner.py:65`); `SweepConfig` carries neither, so the labels' horizon k=3 — "the only free knob", per `outcomes.py` — is invisible to `sweep_hash`, the manifest, and any future audit. A silent change to a default would relabel a re-run dataset without changing its hash. Add both to `SweepConfig` (they then flow into the manifest and hash) and thread them into `EpisodeRunner`.

**P1-7. `ANALYSIS_PROTOCOL["H2"]` describes the superseded test.** `analysis/stats.py:44` still says "refit the MixedLM with per-handoff CPVI … attenuation (Baron-Kenny step)" while the shipped H2 is episode-level mediation with a bootstrapped indirect effect (`rq1.py`). This dict is the freeze-and-cite artefact; it must match the code on freeze day. Update it (and add entries for the efficiency model of P1-11 and the length-control of RD-14 when you adopt them).

**P1-8. Analysis artefacts are under-provenanced.** `RQ1Result` carries no encoder name/revision, probe config, or git SHA; `CalibrationReport` has git SHA + dataset hash but no encoder revision or calibration config. CLAUDE.md's own rule: a result with an unrecorded revision is not a result. Add a small shared `AnalysisProvenance` block (encoder name+revision, probe config, git SHA, timestamp) embedded in both.

**P1-9. Tracked dependency versions omit the ones that shape results.** `manifest.py:27` `_TRACKED_DEPS` lacks `pymunk` (the physics engine version literally shapes trajectories), `scipy`, `statsmodels`, `joblib`, `sentence-transformers`. Add them; and when serving goes live, capture the *server-side* `vllm`/`torch` versions into the sweep manifest (echo from the jobscript, or query the endpoint's `/version` where available) — the client-side environment does not see them.

**P1-10. `y_discrete_config` is a state feature, not an outcome.** `sim/outcomes.py:92` labels the chamber at the *handoff* (pre-state); the roadmap's option was "bucketed **next** pose region". As a Y candidate it is degenerate — `g_base` reads the chamber straight off `e_s`, so CPVI ≡ 0 by construction. Either relabel as the chamber at step i+k (post-state of the window end) or strike it from the candidate set before the freeze.

**P1-11. Efficiency has no inferential treatment.** H1 claims success *and* step-efficiency degrade; `analyse_rq1` models success only (steps are a descriptive mean, conflating successes with budget-censored failures). Add a pre-registered efficiency contrast — simplest defensible: Cliff's δ on steps-to-goal treating failures as censored at budget (rank-based handles the censoring mass), or a survival framing if you want polish. Cheap; closes an obvious examiner question.

**P1-12. Forward-window censoring at episode end is silent.** `outcomes.py:85` clamps the window (`end = min(i+k−1, n−1)`), so the last k−1 handoffs carry shortened horizons, and short episodes shorten everything. Fine as a choice — but make it visible: either mark truncated labels (a boolean, or exclude the final k−1 handoffs from probe fits) or pre-register "labels near termination use the truncated window" with the k-sensitivity analysis. Silent is the only wrong option.

**P1-13. RQ3a's Y-on-logs is undefined — a design decision, not a coding task.** See §10. Do not start DSE-023 before this is written down.

**P1-14. Heterogeneous pairs are unsupported in the runner.** One `client` drives both roles (`graph.py`). DSE-021 needs A and B on different models. Small refactor now — `EpisodeRunner(client_a, client_b=None)` defaulting to self-play — is far cheaper than after more call-sites accrete, and it also serves the C5 supervisor stub later.

**P1-15. Pooled probe fitting across conditions is an unregistered analysis decision.** `analyse_rq1` fits one cross-fitted probe family over all conditions' handoffs. This is defensible (one V, comparable bit-scale across conditions; Ethayarajh-style) and I would keep it — but it is a choice with an alternative (per-condition probes), and the mediation results can shift with it. Name it in the pre-registration; optionally report a per-condition-fit sensitivity check in the appendix.

**P1-16. The embedding cache key omits the encoder name — the encoder-sensitivity check would silently compare an encoder with itself.** `measure/featuriser.py:101`: `digest = sha256(revision + text)`. Both the default (`bge-base`) and the second encoder (`all-mpnet`) default to revision `"main"`, so identical text yields the *same cache path* for both encoders. Running the DSE-022 sensitivity check with a shared `.embed_cache` would serve bge vectors to the mpnet run and report near-zero sensitivity — a fabricated robustness result. Fix: include `cfg.name` in the digest (and re-key or clear the cache). Two lines; the failure it prevents is a thesis-level embarrassment.

**P1-17. Per-handoff CPVI/PVI scores are not persisted.** `write_rq1` writes summaries, contrasts, and figures; the per-instance vectors (row-aligned to records — the "join key" the estimator's docstring advertises) are discarded. The methodology promises the *distribution* of per-handoff CPVI, RQ2 needs exactly these scores, and re-computing them re-fits probes (probe-seed noise makes the re-computation non-identical). Persist a `scores.parquet` (episode_id, step, cpvi, pvi) beside `rq1.json`.

---

## 6. P2 findings — hygiene, in one pass

1. **CI never gained the ≥80% coverage gate** (`ci.yml:40`); the pyproject comment says "raise when the core lands" — it landed. Add `--cov-fail-under=80` scoped to the load-bearing core, plus a `bandit -r src/ -ll` job and the weekly scheduled `pip-audit` CLAUDE.md promises.
2. **`langchain-openai` is a declared, unused dependency** (zero imports; the code uses the raw `openai` client). Remove from `pyproject.toml` + DEPENDENCIES §3, regenerate `uv.lock`. (`langgraph` is used; keep.)
3. **No `.gitattributes` LFS allowlist and no `CITATION.cff`** — both are named DSE-029/CLAUDE.md commitments; the allowlist matters *before* the first figure is committed, not after.
4. **Calibration→deployment threshold drift:** `_choose_threshold` picks the operating point on out-of-fold ensemble scores; DSE-018 will deploy one statistic fit on all calibration data, whose score distribution shifts slightly — firing rate at deployment ≠ budget. Mitigate in DSE-018: refit-on-all, then re-measure the firing rate on a held-out slice and record both.
5. **Docstring nit:** `_platt_reliability` says "1-parameter logistic"; sklearn fits slope+intercept.
6. **Episode-iid bootstraps ignore seed clustering** (`_delta_ci`, `_bootstrap_indirect`) — self-noted ponytail; revisit after the pilot shows whether seed variance is material. Also: persist the *retained-draw count* from `_bootstrap_indirect` (degenerate draws are skipped; a CI from 40 of 400 draws should be visibly flagged).
7. **Run-layout doc drift:** CLAUDE.md specifies `runs/<experiment>/<run_id>/` with `handoffs.jsonl`; the implementation is `data/<hash>/part-*.parquet` + `<hash>-run/`. The implementation is better; update CLAUDE.md and the stale `.gitignore` patterns (`runs/*/handoffs.jsonl`) to match reality.
8. **`register_dataset`'s index.json read-modify-write** is unlocked across *processes* (in-process lock only). Fine solo; add a note or an `flock` if two sweeps might ever share a root.
9. **G3 heuristic blind spots (concrete ones):** degree-vs-radian mentions ("rotate 90 degrees" → falsely ungrounded vs angle 1.57); grid-mode messages citing cell coordinates; and under `nl`, purely directional messages ("push east") are vacuously grounded (1.0), so G3 can pass on a corpus that never cites geometry. Consider reporting `n_with_numbers/n_records` next to the gate value (it is already in `detail`) and eyeballing a transcript sample before trusting a pass — the renderer (§7) makes that cheap.
10. **`serve.sh` records nothing about the serving environment** — add `vllm --version`, `python -c 'import torch; print(torch.__version__)'`, GPU name, and the resolved revision to the job log, and copy them into the sweep manifest for live runs (pairs with P1-9).
11. **Issue #42** points at a PR-41 comment that does not exist via the API — close it or replace with the real content if it lives somewhere else (e.g. an email).
12. **Config tree gaps:** no `llama70b`/`gemma` model yamls (needed by DSE-021), channel parameters not exposed as a Hydra group (only via `SweepConfig` defaults). Minor; do with DSE-021.

---

## 7. Architecture gaps — things no ticket currently builds

1. **An interim serving path (the pilot unblock).** Covered in §9 — the good news is `LLMClient` already takes `base_url`, so a rented-GPU vLLM needs *zero code changes*; only hosted-API fallbacks need a structured-output adapter.
2. **A trajectory renderer / episode transcript dumper.** Nothing in the repo can draw an episode or dump a readable transcript, yet: (a) debugging G1 failures without watching trajectories is guesswork; (b) prompt iteration (the one retune you are allowed) needs eyes on messages next to states; (c) DSE-029's acceptance criterion says "a committed demonstration trace **renders**" and no ticket builds the renderer; (d) the thesis needs the arena figure anyway. ~80 lines of matplotlib (arena patches + T polygon per step + message annotations → PNG strip or GIF) plus a `render_transcript(records) -> markdown` sibling. Highest tooling leverage per line in the repo right now.
3. **Per-role clients** (P1-14) — unlocks DSE-021 heterogeneous and the C5 supervisor.
4. **Schema v2 as one deliberate bump** — `observation` (P0-1), gate fields (P0-4), done together before any real data exists.
5. **A message-length covariate path.** No analysis controls for message length, and "is CPVI just length?" is the first sceptical question (Lowe et al. is the whole reason the design is causal). No schema change needed — token counts derive from `message_delivered` — but the length-controlled CPVI analysis (partial correlation / length as a covariate in the mediation) should be pre-registered (RD-14).
6. **A shuffled-message estimator audit on real data.** The synthetic tests pin CPVI ≈ 0 on noise; add the same audit on pilot data — recompute CPVI with messages permuted within condition; it must collapse toward 0. Cheap, and it is the single most convincing "the estimator isn't hallucinating signal" exhibit for the thesis (RD-15).
7. **Interim-substrate labelling in the manifest.** Add a `serving_substrate`/endpoint field to `SweepManifest` so interim-compute pilot data is permanently distinguishable from Myriad data (one string field; pairs with §9).

---

## 8. Research-design red-team — free now, forbidden after the freeze

Everything here is pre-freeze by definition (nothing has run). Each item: what, why, recommendation.

**RD-1. Conditioning-set semantics (the P0-1 decision).** Recommend: receiver-observed state, pre-registered, with the dual-baseline C3 diagnostic (§4). This is the single most important construct decision left.

**RD-2. The prospective twin cannot see negative CPVI — turn the bug into a finding.** `prospective_twin` = KL(g_cond‖g_base) ≥ 0, while retrospective CPVI is signed (a *misleading* message scores negative). Agreement (H3) will therefore be structurally attenuated wherever messages mislead — which is exactly the C4 tail. This is not fixable (no target-free statistic can know the message is wrong; if the probe knew, it wouldn't be misled) — it is a *property*: the prospective twin estimates E[CPVI | s, m] under the probe's own belief. Recommendation: pre-register H3 agreement stratified by retrospective sign; state the one-sidedness in §9.4 (fix D-2's text at the same time); frame "prospective scoring is blind to deception/error, retrospective is not" as a finding about the limits of runtime measurement — it strengthens, not weakens, the RQ2 chapter.

**RD-3. Per-handoff CPVI carries probe-fit noise; stabilise it.** A single K-fold cross-fit gives each handoff a score from one fold-model; per-instance signs can flip under probe reseeding. For the scores that feed mediation and RQ2, average over R repeated cross-fits (R=5–10 with different fold seeds; sklearn logistic refits are milliseconds at this scale) and report the across-repeat SD as the score's measurement error. Pre-register R. This materially de-noises the headline mediation at near-zero cost.

**RD-4. Probe capacity vs pilot N.** e_s;e_m is 1536-dim; pilot handoffs are O(10²–10³). `auroc_train_cond` is already the overfit monitor (good); add the decision rule to the pre-registration: how C (and logistic-vs-MLP) get chosen on pilot data only, then frozen. Do not leave the probe-selection procedure implicit — that is where researcher-DoF accusations land.

**RD-5. C1's cap and C4's rate are pre-registration parameters.** Defaults (8 whitespace tokens; 0.4 dropout) differ from the roadmap's illustrative ≤30. Fine — but freeze them (and the C3 window of ±2 rows) in the pre-registration, with the pilot allowed to set them once.

**RD-6. "Agents must agree an action" vs implementation.** B decides alone after reading A's message; there is no agreement round. The implementation is the cleaner measurement object (one boundary, one direction). Recommend amending the thesis wording ("B, informed by A's handoff, selects the joint action") rather than adding a negotiation round — an ack round would create a second boundary and muddy the unit of measurement (D-6).

**RD-7. Serialisation isomorphism is approximate — don't overclaim.** The grid quantises pose to 0.25 cells and shows angle only via footprint; numeric's velocity line is constantly ≈0 (velocities are zeroed by quasi-static settling before the read) and thus dead weight. Soften "isomorphic in information" to "matched in task-relevant information (pose, goal, contact) up to representation precision" in the thesis; optionally drop the vel line from `numeric` at the same time as the legend fix (P1-5) so both serialisation changes land in one PROMPT/serialisation version bump.

**RD-8. Prompt v1 is a placeholder — treat prompt iteration as a budgeted, versioned pilot activity.** `_SYSTEM_A` already leaks strategy ("pushed rightward through narrow slits") — acceptable, but note it reduces what messages must carry under C0–C2/C4 (it does *not* leak the C3-hidden goal specifics — good). B receives no coordinate convention (which way is N?) — under `numeric`/`nl` the frame is implicit. Plan: 2–3 prompt iterations on interim compute against transcripts (renderer!), each a `PROMPT_VERSION` bump, all before the pilot's formal gate run; freeze v_final with Y/V.

**RD-9. Pre-register the multiplicity structure across Y candidates.** Four Y labels exist; the freeze picks one primary (recommend `y_binary_progress`, k frozen from pilot) + one continuous twin, with `y_terminal_success` as secondary and `y_discrete_config` dropped or relabelled (P1-10). Say explicitly that all headline inference is on the primary Y and everything else is descriptive — that sentence is what stops "you had four outcomes" at the viva.

**RD-10. Efficiency endpoint** — adopt the P1-11 model and add it to the protocol dict.

**RD-11. Gate-threshold transport is a known limitation — say it.** The RQ2-calibrated threshold is applied in RQ3b runs whose message distribution the gate itself perturbs downstream. The matched-firing-rate control neutralises the *count* concern; the distribution-shift concern should be acknowledged (and the deployment firing rate reported vs budget, per P2-4).

**RD-12. Add the message-length control** (RD-14 in my notes, stated here once): pre-register a length-controlled CPVI analysis — partial Spearman of CPVI with outcome given length, and length as a covariate in path b of the mediation. Without it, "CPVI is a fancy word-count" is an open attack line; with it, C1 (which manipulates length directly) becomes the demonstration that the construct and the confound dissociate.

**RD-13. Shuffled-message audit** (§7 item 6) — pre-register as a manipulation check with an expected ≈0 result.

**RD-14. Report CPVI distributions, not just means** — persist per-handoff scores (P1-17) and show per-condition violin/ECDF; the methodology promises "the full distribution", and the per-handoff distribution is the contribution's selling point over an aggregate.

---

## 9. Unblocking the pilot before Myriad (Q1 — the detailed plan)

**Recommendation: rent a single GPU and run your own stack on it.** The repo's serving path is a vanilla vLLM OpenAI server plus a `base_url`; nothing about it is Myriad-specific. That makes the interim decision easy — fidelity ranks the options:

| Option | Fidelity to Myriad | Cost | Structured output | Verdict |
|---|---|---|---|---|
| **Rented GPU (RunPod/Modal/Lambda) + your `serve.sh` vLLM** | Identical semantics: same vLLM, same flags (`--seed`, guided xgrammar), same pinned revision; only the hostname differs | A100-80GB ≈ $1.5–2.5/h; L40S/4090 ≈ $0.5–1/h; pilot ≈ 2–5 GPU-h → **≈ $5–15** | `guided_json` works as-is | **Primary. Zero code changes beyond `base_url`.** |
| Local Apple-Silicon (Ollama, `qwen3:8b`) | Different runtime/quantisation; fine for smoke, not for gate evidence | Free | Ollama supports JSON-schema `format`; needs a small client adapter | **Secondary: free loop-smoke + prompt iteration.** An 8B at ~20–40 tok/s ≈ 1–2 min/episode → a 90-episode grid overnight. |
| Hosted APIs (Together/Fireworks/DeepInfra) | Weakest: no revision pinning guarantees, structured-output APIs differ (no xgrammar), batching opaque | Cents | Needs a `response_format` adapter in `LLMClient` | Fallback only if renting is blocked. |

**Concrete sequence (compressed into ~4–6 working days, all pre-Myriad):**
1. Land P0-1/P0-2/P1-16 + the schema-v2 bump (one branch); P0-3's client/config changes (second branch).
2. Spin up the rented GPU; run `serve.sh` with the corrected 8B id at a pinned revision; run **DSE-005's benchmark harness** there (build it now — it is Phase 0 work and its table also feeds the Myriad allocation request): throughput, schema-adherence rate, and the 10-episode C0 smoke.
3. Feasibility certificates (P1-4): BFS solvability per difficulty on the local machine (no GPU); set per-difficulty `max_steps`.
4. Prompt iteration: 5–10 episodes per serialisation on the rented 8B/14B, read transcripts via the renderer, bump `PROMPT_VERSION` at most 2–3 times, stop.
5. Formal pilot grid on the rented 14B: C0/C1/C4 × {easy, hard} × ≥3 seeds (now real replicates via jitter), `run_pilot` → G1/G2/G3 report. One retune allowed, per the roadmap.
6. Label everything `serving_substrate="interim-<provider>"` in the manifest. Policy: interim data informs **decisions** (gates, retunes, k, prompts, budgets); headline data is re-generated on Myriad. Re-running the pilot grid on Myriad to confirm the gates costs an hour and inoculates the thesis against "your pilot ran on different hardware".

**Why this beats waiting:** if a gate fails, you want the retune (or the RQ3a elevation decision) to happen *this* week, not to discover it during your Myriad allocation window. The fallback ladder only protects the schedule if the gates fire early — that was the entire point of the roadmap's front-loaded pilot, and it is currently the plan's biggest unexecuted idea.

---

## 10. RQ3a pull-forward (Q6) — and the design hole to close first

**The hole (P1-13).** CPVI needs a per-handoff outcome Y. On the simulator, physics supplies it. On Who&When — a corpus of *failure* logs — trace-level Y is (near-)single-class, so "refit probes on a held-out portion of the logs" (lit review §9.8) is unfittable as stated; and using the decisive-error-step annotation as Y would train the probe on the very label the localisation claim is evaluated against — circularity with extra steps. The methodology text currently glosses this; it is the one place RQ3a could quietly die during build week.

**Three workable definitions (present all three in the design note; recommend (a)+(c)):**
- **(a) Mixed-outcome corpus, trace-level Y.** Use MAST-Data (which contains successes — failure rates 41–86% imply 13–59% successes) or augment Who&When failures with matched successful traces from the same frameworks where available. Y = trace success; CPVI per handoff = what this message adds, beyond context, about eventual task fate. Weak per-handoff signal (long credit path — the same critique as `y_terminal_success`), but honest, refittable, and matches "low usable information localises the decisive step" as *low CPVI ⇒ the message stopped informing the outcome*.
- **(b) Local next-step Y.** Define a per-step outcome from the log itself, e.g. "the receiver's next action is schema-valid / advances the recorded plan / is not the annotated error". Sharper per-handoff signal, but the label construction is itself a methodological choice needing its own audit, and "is not the annotated error" variants re-import circularity — only the annotation-free variants are safe.
- **(c) Target-free statistics as the transfer object.** `s_cos` needs no Y at all, and `s_info`/the simulator-trained probe transfer without refitting (the "transfer" regime DSE-024 already names). Localisation via runtime statistics sidesteps the Y problem entirely for the transfer arm; only the refit arm needs (a)/(b).

**Recommended shape:** transfer arm = simulator-trained `g_cond` + `s_cos` (no log-side Y needed); refit arm = (a) on MAST/mixed traces; report both, as DSE-024 already requires. Write this as a one-page design note (it will become a methodology subsection) *before* DSE-023.

**Practicalities to verify during the loader spike (all CPU-side, Myriad-independent):** Who&When lives on HuggingFace (Algorithm-Generated + Hand-Crafted splits, 127 systems; agent + step + reason annotations); MAST-Data is released with the taxonomy paper alongside its LLM-as-judge annotator. Embedding with bge-base on CPU/MPS is fine at corpus scale; probes are sklearn. Timebox the first spike to: download both, parse 10 traces each into `HandoffRecord`-compatible rows, count extractable handoffs, and confirm which corpus has enough successful traces for (a). That one afternoon converts RQ3a from "assumed viable fallback" to "verified viable fallback" — which is what a fallback is for.

---

## 11. Pre-registration and power — high level, per your scoping

**Freeze checklist** (each item: value + where it is recorded; freeze point = pilot verdict, before the RQ1 main sweep). Suggested artefact: `PREREGISTRATION.md` at repo root, committed before the main sweep, with a deviations log appended thereafter.

- Y: primary (`y_binary_progress`, k = value from pilot) + continuous twin; secondaries named descriptive-only (RD-9). k-sensitivity set {1, 3, 5} reported.
- Conditioning state: receiver-observed observation (P0-1 semantics).
- V: probe family + hyperparameters and the selection rule used on pilot data (RD-4); repeated-cross-fit R (RD-3).
- Encoder: name + commit-SHA revision (pin `bge-base-en-v1.5`; second encoder for sensitivity only); cache key fixed (P1-16).
- Serialisation: the mode chosen by the pilot/A-B rule, and the rule itself.
- Channel parameters: C1 cap, C3 window, C4 rate (RD-5). Jitter distribution + seed count.
- Budgets: per-difficulty `max_steps` from feasibility search (P1-4).
- Hypotheses + tests: the corrected `ANALYSIS_PROTOCOL` (P1-7) including the efficiency endpoint (P1-11), length control (RD-12), shuffled-message check (RD-13), H3 sign-stratification (RD-2).
- Gate: statistic, firing-rate budget, threshold rule, retry-feedback template (P0-4), and the calibration target (realised failure — already structural).
- G-gate thresholds: G1 floor, G2 gaps (including re-setting `g2_min_cpvi_gap` to a positive value once the pilot reveals the bit-scale — the code already documents this intent), G3 floor + tolerances.

**Power, one honest paragraph.** For the headline episode-success contrast at α=0.05 with Holm over four contrasts and 80% power: a C0→hard gap of 0.4 (e.g. 0.75→0.35) needs roughly 25–30 episodes per condition; a gap of 0.3 needs ~45–50; a gap of 0.2 needs ~90–100. With jitter × seeds supplying true replicates, 10 jittered instances × 5 seeds = 50 episodes/condition ≈ 250 episodes for C0–C4 ≈ 3–6k steps ≈ 6–12k LLM calls — single-digit GPU-hours on a batched 14B. The pilot's measured C0-vs-hard gap picks the row; the compute request follows from it. (Deliberately no further planning detail, per your instruction.)

---

## 12. Document ↔ code drift register (fix-code vs fix-text)

| # | Drift | Recommendation |
|---|---|---|
| D-1 | Abstract's runtime statistics: "failure-risk, **JSD**, cosine"; methodology §9.5 + code: **entropy**, failure-risk, cosine (JSD is an RQ2 bridge) | **Fix text (abstract)** to entropy; keep JSD as the reported RQ2 bridge. (JSD *is* computable at the handoff, so promoting it is also coherent — but pick one and align all three artefacts.) |
| D-2 | §9.4 prospective twin = "the same fitted conditional probe … predictive distribution"; code = per-instance KL(g_cond‖g_base) | **Fix text**: state the KL form + the E[CPVI] interpretation + one-sidedness (RD-2). |
| D-3 | §9.3 heteroscedastic continuous probe; code homoscedastic (recorded deviation) | Decide at freeze: implement the 2-headed Gaussian-NLL **or** amend text. Default: amend text; upgrade only if the pilot's continuous twin misbehaves. |
| D-4 | §9.2/roadmap C4 = "dropout **or paraphrase**"; code = dropout only | **Fix text** (drop paraphrase) — a paraphrase channel needs a model call inside the channel, which violates "the channel is cheap and deterministic". Optional post-pilot stretch arm at most. |
| D-5 | Abstract: outcome = "whether **the joint action** after a handoff makes net progress"; code/methodology: next-k window (k=3) | **Fix text** after k freezes (or set k=1 primary if the pilot supports it — k=1 is the abstract's literal reading). |
| D-6 | "The two agents must agree a single macro-action" vs B-decides implementation | **Fix text** (RD-6). |
| D-7 | CLAUDE.md run layout (`runs/…`, `handoffs.jsonl`) vs implemented parquet layout | **Fix CLAUDE.md + .gitignore** (P2-7). |
| D-8 | Model ladder names (Qwen3-\*-Instruct, 70B-AWQ id, "Gemma-4", "Mistral Small 4") | **Fix configs/roadmap** after DSE-005 verification (P0-3). |
| D-9 | Roadmap C1 "≤30 tokens" vs default 8; roadmap G1 "≥60%" vs default 0.5 | Neither is wrong; **pre-register the chosen values** (RD-5). |
| D-10 | `ANALYSIS_PROTOCOL["H2"]` vs shipped mediation | **Fix code string** (P1-7). |
| D-11 | Roadmap action schema had a `rationale` field; implementation dropped it | Fine (keeps B's reasoning out of the scored channel); note in thesis §3 that B emits action-only JSON. |
| D-12 | CLAUDE.md/DEPENDENCIES name a `tests/e2e/` tier; directory absent | Create with the first live-endpoint test (Myriad week); no action now. |

---

## 13. Sequenced next steps

**This week (pre-Myriad, all local/interim):**
1. *(code)* Branch 1 — measurement-validity fixes: schema v2 (`observation` + gate fields), receiver-conditioned featurisation, seeded pose jitter, cache key fix, StepConfig/OutcomeConfig into SweepConfig/manifest, per-handoff score persistence, `ANALYSIS_PROTOCOL` H2 text. (P0-1, P0-2, P1-6, P1-7, P1-16, P1-17.)
2. *(code)* Branch 2 — serving readiness: model ids + pinned revisions, `chat_template_kwargs` in `ServingConfig`, `<think>` guard, ServingError propagation split, grid legend (+ drop the dead `vel` line), per-role clients. (P0-3, P1-3, P1-5, P1-14, RD-7.)
3. *(code)* Branch 3 — pilot tooling: DSE-005 benchmark harness; BFS feasibility certificates + per-difficulty budgets; the episode renderer/transcript dumper. (P1-4, §7.)
4. *(experimentation)* Rent the GPU, serve 8B→14B with your own `serve.sh`, run the DSE-005 table, iterate prompts (≤3 versions), then the formal pilot grid → G1/G2/G3 report; one retune if needed. (§9.)
5. *(research)* Write two one-pagers: the RQ3a Y-on-logs design note (§10) and `PREREGISTRATION.md` v0 (§11). Run the loader spike on Who&When/MAST samples (CPU).
6. *(hygiene, 1 h total)* Drop `langchain-openai`; add coverage gate + bandit to CI; add `.gitattributes` allowlist + `CITATION.cff`; close issue #42; extend `_TRACKED_DEPS`. (P2-1/2/3, P1-9.)

**Myriad week (~21 July):**
7. DSE-002/003 live checks; re-run DSE-005 on-cluster per available GPU class; resolve the allocation decision; re-confirm pilot gates on-cluster (cheap).
8. **Freeze**: Y (k), V, encoder revision, serialisation, channel params, prompts — commit `PREREGISTRATION.md` v1.
9. RQ1 main sweep (DSE-020 full run) + DSE-021 cells; freeze the RQ1 result; draft chapter 5 the same week (the roadmap's own writing rule).

**Then, in priority order:** DSE-022 (RQ2, with the encoder-sensitivity check now guaranteed honest by P1-16) → DSE-017 re-calibration on RQ1 data → DSE-018 (with the P0-4 retry design) → DSE-025 (RQ3b) ∥ DSE-023/024 (RQ3a, per §10) → DSE-029/030.

**Cut-lines, exercised now rather than later:** formally defer DSE-027 (SocialJax) — with a 21 September hard deadline and the paper reusing RQ1+RQ2, it cannot pay for itself; decide C5 (DSE-026) only after RQ1 freezes. Recommend annotating both issues today so the backlog reflects the real plan.

---

## Appendix A — proposed new tickets (so none of this lives only in this document)

| Proposed | Title | Covers |
|---|---|---|
| DSE-031 | Schema v2: receiver observation + gate fields; receiver-conditioned featurisation | P0-1, P0-4 fields |
| DSE-032 | Seeded scenario jitter and replication semantics | P0-2 |
| DSE-033 | Serving correctness: model ids/revisions, thinking-mode control, think-guard, error-propagation split | P0-3, P1-3 |
| DSE-034 | Feasibility certificates + per-difficulty step budgets (BFS over macro-actions) | P1-4 |
| DSE-035 | Episode renderer + transcript dumper (also satisfies DSE-029's "demo trace renders") | §7-2 |
| DSE-036 | Interim serving substrate: rented-GPU runbook + manifest substrate field (+ optional Ollama adapter) | §9 |
| DSE-037 | Analysis provenance + per-handoff score persistence + protocol/text sync | P1-7, P1-8, P1-17 |
| DSE-038 | RQ3a Y-on-logs design note (blocking DSE-023) | §10 |
| DSE-039 | PREREGISTRATION.md + freeze procedure | §11 |
| DSE-040 | Statistic/Y construct alignment at freeze (`s_info`, G2) + `y_discrete_config` fix-or-drop | P1-1, P1-2, P1-10 |

*End of review. Findings are point-in-time against commit `54e7b85`; anything landed after that is not reflected.*
