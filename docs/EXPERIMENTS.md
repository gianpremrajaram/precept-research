# EXPERIMENTS.md

> **What this document is.** The operational register of every experiment this project runs: what
> each one asks, what it needs before it can start, the exact command that starts it, what it emits,
> how its result is recorded, and what a null looks like. It is the counterpart to
> `docs/methodology.md`, which says *why* the design is what it is. Where the two disagree, the
> methodology is the authority on construct and interpretation and this file is the authority on
> execution.
>
> **How the docs relate.** `RESEARCH_ROADMAP.md` is the strategic spine (phases, gates, architecture).
> `docs/methodology.md` is the thesis text. `docs/EXPERIMENTS.md` is this file — the run ledger.
> `docs/experiment_design_log.md` records *why the design changed*. `CHANGELOG.md` records *what the
> code now does*. GitHub Issues is the backlog; `ISSUES.md` mirrors it offline.
>
> **Standing rule.** No experiment is "done" until its row in §1 carries a run identifier, a manifest
> path, and a written result paragraph in §5's format. A run whose result was never written down is a
> run that has to be done again.
>
> **Last revised:** 26 August 2026.

---

## 1. Status board

The task is certified, headline data exists on Myriad bf16, and **the E3 ledger is closed.** The
one-retune ledger was spent on 26 August (attempt 1 `retune_once` → prompt v5 → attempt 2
`fallback`), the ladder fired, and rung 2's single permitted re-gate ran on 29 August on the
successor bar task: **`af50c7c12d65540f` returned `fallback` again** — G1 PASS 0.800, G2 FAIL −0.250,
G3 groundedness PASS 1.000, G3 correctness FAIL 0.242. There is no attempt 3 and no second rung-2
attempt. **Rung 3 is the finding for the arena track**, and the mechanism is named rather than
absent: only the lossy channel (C4) produces a receiver whose actions beat their own within-episode
permutation null. Rung 1 (RQ3a) is frozen on its transfer regime. The interim `local-lmstudio`
episodes remain permanently labelled and are never pooled with cluster data.

| ID | Experiment | Stage | Needs | Status |
|---|---|---|---|---|
| E0 | Task certification (CPU only, no model) | S0 | Nothing | **Passed 24 Aug 2026** |
| E1 | Serving smoke test and transcript read | S1 | A served model | **Run 24 Aug 2026 (v2, then v3)** |
| E2 | Model-ladder benchmark (DSE-005) | S1 / S2 | A served model | **Local row 24 Aug 2026; cluster rows open** |
| E3 | Formal pilot gate run — G1/G2/G3 (DSE-019) | S1 → re-gate S2 | E1, the driver | **CLOSED.** Attempt 2 on the T load, 26 Aug (`eddd19c654515bb2`): `fallback` — G1 FAIL 0.300, G2 FAIL −0.200 sign-inverted, G3 PASS 0.999. G1 returned on the rung-2 successor 29 Aug (`86ecbbdf35322dc3`, C0-only): PASS 0.500. **The rung-2 re-gate, 29 Aug (`af50c7c12d65540f`, 80 episodes, job 238085): `fallback`** — G1 PASS 0.800, G2 FAIL −0.250 (CPVI gap −0.015), G3 groundedness PASS 1.000, **G3 correctness FAIL 0.242 vs null 0.257**. The pre-registered prediction came out half confirmed (C1 0/20 as predicted; C4 14/20 inverted again), so the instruction account is insufficient and **rung 3 stands**. Frozen at `runs/rq1/af50c7c12d65540f/` |
| E4 | RQ1 information-gradient main sweep (DSE-020) | S3 | **rung-2 re-gate = proceed**; the freeze | **Will not run.** Its precondition failed on 29 Aug; §6 permits no further attempt. RQ1 is written up as rung 3 — the absent gradient as the finding, with the receiver-competence mechanism |
| E5 | RQ1 robustness cells (DSE-021) | S3 | E4; per-role client refactor | **Not run** |
| E6 | RQ2 measurement primitive (DSE-022) | S4 | ~~E4~~ E3 re-gate episodes (no new compute) | **Run 30 Aug 2026 on `af50c7c12d65540f`** (`runs/af50c7c12d65540f-rq2/`). H4 supported on all three statistics — shuffle-corrected ρ +0.275 / −0.275 / +0.269, every interval excluding zero, `fail` AUROC 0.906. Encoder invariance holds (ρ 0.816, ranking invariant). Selects `y_terminal_success` as RQ3b's gate target by the rule declared before any outcome was read. **Awaiting freeze (F2)** |
| E7 | Gate calibration (DSE-017 run) | S4 | E4 episodes | **Not run** |
| E8 | RQ3b causal gate + controls (DSE-025) | S5 | E7 threshold; DSE-018, DSE-045 | **Not run — explicitly deferred behind rung 2**: calibrating an outcome-thresholded gate on a task where degrading the message improves outcomes inherits the inversion |
| E9 | RQ3a corpus spike and counts (DSE-041) | S6, parallel | Nothing but network | **Run 24 Aug 2026.** Conditioning state confirmed (220 traces, 5,960 steps, 2,488 handoffs, recorded input contexts); **trace outcome falsified — 0 non-failures on the primary corpus**. Promotes DSE-042 to load-bearing |
| E10 | RQ3a replay labelling (DSE-042) | S6 | E9; a spend cap | **Not run — now the only route to a two-class step-level *Y*** (E9) |
| E11 | RQ3a localisation and baselines (DSE-024, rescoped) | S6 | E9, ~~E10~~, transfer calibration | **Transfer regime run and frozen 29 Aug 2026** (`runs/rq3a/{traceelephant,who_and_when}/`). TraceElephant: CPVI transfer agent accuracy **0.525 [0.432, 0.610]** against schema validity 0.263 and mean cosine 0.254 — non-overlapping intervals. Who&When: 0.333, indistinguishable from mean cosine's 0.367. MAST secondary 0.088 bits [0.070, 0.107]. **The refit regime stays undefined without E10, and the judge arm is unrun** — so `judge` and `agreement` are `null` and no comparison to the published 53.5 % / 14.2 % is stated |

---

## 2. Execution stages

The stages are ordered by what unblocks the most, not by research priority. S0 and S6 need no GPU at
all; S1 needs no cluster and no money.

### S0 — Certify the task on CPU, with no model in the loop

The A\* feasibility oracle re-derives the optimal path for each difficulty against the frozen budgets.
If a difficulty comes back unsolved the geometry has drifted, and every downstream result would
misdiagnose an impossible task as a coordination failure. This costs seconds and is the cheapest
insurance in the project.

```bash
uv sync --extra dev --extra embed --extra viz
uv run ruff check . && uv run mypy --strict src/preceptx && uv run pytest
uv run python -m preceptx.sim.feasibility     # expect easy 7, medium 13, hard 13 vs budgets 18/33/33
```

### S1 — Free local pilot: the first real model calls

The pilot runs against a local OpenAI-compatible server on the development machine, at zero cost and
with no cluster dependency. This is where every integration defect in the serving path, the prompts,
the guided-decoding schema and the transcript surfaces — none of which should be discovered inside a
scheduler queue. The local model is the 8B pilot tier; the workhorse configuration is not run here.

The local runtime is LM Studio, driven headless through its `lms` CLI (model download, server start)
and serving `mlx-community/Qwen3-8B-4bit` at `http://localhost:1234/v1` with
`structured_mode=response_format`. Runs are labelled `PRECEPTX_SERVING_SUBSTRATE=local-lmstudio` and
write to their own data root. The model identity recorded in the manifest is the quantised repository
and its revision SHA — not the upstream bf16 repository — so the quantisation is visible in every
record. The constrained-decoding engine also differs from the cluster's (Outlines / llama.cpp grammars
locally, xgrammar under vLLM), which is one more reason the local schema-adherence rate is recorded as
its own number.

Runs at this stage are permanently labelled as interim through the manifest's serving-substrate field,
which is deliberately excluded from the configuration hash so that the substrate is recorded without
becoming part of experiment identity. **Interim data informs decisions — the one allowed retune, the
label horizon, prompt wording, budgets. Headline data is regenerated on the cluster.** The two can
never be silently pooled.

**One limit on what a local run can conclude, and it is not cosmetic.** The local tier is 4-bit
quantised and the cluster tier is bf16, so the two are not the same model in the sense that matters
for a capability claim: quantisation changes the output distribution, and G1 is precisely a capability
threshold. A local pass on G1 is therefore **indicative, not the verdict of record**, and a local
*failure* on G1 is likewise not sufficient grounds to invoke the fallback ladder — it must be
reproduced at bf16 first. What a local run does settle, and settles cheaply, is everything that is not
a capability claim: whether the loop runs at all, whether the guided-decoding schema is honoured,
whether the messages are grounded (G3, which is about content rather than difficulty), whether the
coordinate convention is legible to the model, and whether the prompts are worth iterating. The formal
gate run of record is the Myriad re-gate in S2. Record the local schema-adherence rate as its own
number; a 4-bit 8B may fail the action schema more often than a served 14B, and that belongs in the
model-ladder table rather than in a footnote.

### S2 — Myriad: re-gate, then benchmark the ladder

The pilot grid is re-run on the allocated cluster to confirm the gate verdicts hold on the hardware
that will produce the headline results. This costs roughly an hour and pre-empts the obvious question
of whether the pilot ran somewhere else. The model-ladder benchmark runs here too, because its purpose
is to fix the workhorse tier against real throughput and real memory.

### S3 — RQ1 main sweep

The headline. Runs only after the pilot verdict is `proceed` and the pre-registration is committed.

### S4 — RQ2 and gate calibration

Consumes S3's episodes and needs no additional model calls. Emits the operating point S5 depends on.

### S5 — RQ3b causal arm

Re-runs a subset of conditions in four modes: gate off, gate active, matched-firing-rate control,
random-trigger control.

### S6 — RQ3a external validity (parallel from the start)

Depends only on the measurement stack, not on the gate and not on the simulator sweep, so it runs in
parallel with S3–S5. The corpus spike (E9) has no dependency at all beyond network access and should
start immediately, because RQ3a is the pre-planned fallback that can carry the dissertation alone — and
a fallback that has not been verified is not a fallback.

---

## 3. Experiment specifications

### E0 — Task certification

**Question.** Is every difficulty in the ladder solvable within its budget on the current physics?
**Method.** Exhaustive A\* over the seven macro-actions on the real simulator, per difficulty.
**Emits.** Oracle optimum and budget per difficulty, to stdout and to the regression test.
**Pass.** easy 7 / medium 13 / hard 13 against budgets 18 / 33 / 33.
**If it fails.** Stop. Do not run any model. The geometry has changed and the difficulty axis must be
re-derived before anything else is meaningful.

### E1 — Serving smoke test and transcript read

**Question.** Does the whole loop run against a real model, and are the messages grounded?
**Method.** Five to ten episodes on the unrestricted channel at easy difficulty, then read the rendered
transcript by hand. This is a human-in-the-loop step and cannot be automated away.
**What to look for, in order.** (1) Do the messages cite geometry that matches the true state, or are
they hallucinating coordinates? (2) Does B's chosen action follow from A's message, or is B ignoring it?
(3) Is the coordinate convention unambiguous to the model — does "north" mean what the arena means by it?
**Emits.** `render_transcript(records)` output, read and annotated.
**Budget.** At most two or three prompt-version bumps, all before E3. This is a budgeted activity, not
an open loop, and the final prompt version freezes with *Y* and *V*.

### E2 — Model-ladder benchmark (DSE-005)

**Question.** Which tier is the workhorse, and does the ladder hold on real hardware?
**Method.** Per tier: throughput, time to first token, peak memory, JSON-schema adherence rate, and a
ten-episode unrestricted-channel smoke.
**Emits.** One comparison table, which is also the evidence for the compute plan.
**Note.** Any model identifier in the roadmap ladder that has not passed through this benchmark is a
placeholder, and must be treated as one in the thesis.

### E3 — Formal pilot gate run (DSE-019)

**Question.** Is the headline task viable at all? Three hard gates.

| Gate | Tests | Fails when |
|---|---|---|
| G1 capability | Self-play solves the task on the unrestricted channel, **at easy difficulty** | Success on easy C0 sits below the floor |
| G2 signal | A measurable C0-to-hardest gap exists in **both** outcome and CPVI | The outcome gap is negligible, or CPVI does not fall in the right direction |
| G3 groundedness | Messages reflect true state | Numbers cited in messages do not match the simulator |

Current implementation defaults, frozen or re-set at F0: the G1 success floor is 0.5 on easy C0;
G2's CPVI half is directional-only until the pilot reveals the bit-scale, at which point a positive
floor is pre-registered; the G3 groundedness floor is 0.8.

**Design.** C0, C1, **C3** and C4 crossed with easy and hard, at **three or more seeds**; the cell
runs **seeds 0–9** (80 episodes), widened from 0–2 to 0–4 on 2026-08-24 and from 0–4 to 0–9 on
2026-08-26 after attempt 1 (PREREGISTRATION §6 carries the amendment ledger; both are dated
pre-freeze). Both widenings answer the same problem: G1 is the one gate bf16 can plausibly flip and
the one that has already failed, and at five easy-C0 episodes it read 2/5 — a Wilson 95% interval of
[0.12, 0.77], with a design sitting exactly on the 0.5 threshold failing the gate half the time. Ten
seeds also doubles G2's success half, which passed attempt 1 by exactly zero margin (2/10 against
1/10 — one episode). C3 is in the cell because it is the only condition carrying a genuine observation asymmetry and it
is in the headline design; a pilot that never exercises it certifies an instrument the main sweep will not
use. **G2 fits the CPVI probe once over the whole cell** and contrasts the resulting per-instance scores —
refitting per contrast is a different estimator from the RQ1 analysis's and understated the E3-local
gradient seventeen-fold (+0.012 against +0.211 bits). Reported per-condition CPVI intervals are
**episode-cluster** bootstrap — the episode is the sampling unit — because iid handoff resampling
read roughly half the honest width at pilot scale. CPVI is scored with **R = 5 repeated cross-fits**
(the pre-registered value) and every summary carries **selectivity** against a control task on
random labels, so the report answers "could a probe of this capacity have manufactured this?" in
the same artefact that reports the score. G2 reports **unassessable** rather than FAIL when
every handoff carries the same progress label; an unassessable gate never yields `proceed` and never spends
the retune or invokes the fallback. Three seeds is a hard floor:
a one- or two-seed pass is language-model noise rather than a stable gradient, and the harness downgrades
it to `retune_once` rather than reporting a proceed.
**Emits.** A pilot report with a verdict of `proceed`, `retune_once` or `fallback`.
**How the re-gate is run.** Pre-pull on a login node (`bash scripts/myriad/prefetch.sh`), then one
SGE job, because a login node driving a compute node's `localhost` reaches the wrong machine:
`qsub scripts/myriad/pilot.sh` (DSE-050; no `-P` — the Free allocation is the default), and
`qsub -v ATTEMPT=2 scripts/myriad/pilot.sh` for the one retune. It serves, waits for the endpoint, warms the
embedding encoder, runs the cell and tears the server down on every exit path. Two things the
command deliberately does not carry: the **revision**, which is read from
`configs/model/<TIER>.yaml` — the same file the manifest records it from, so the served and recorded
checkpoints cannot disagree — and the **grid axes**, which come from the `preceptx-pilot` defaults,
so the executed cell and the pre-registered one cannot drift apart. Cluster access, node classes and the first-session order are
in [`myriad.md`](myriad.md); the retune ledger opens at this run, not at the local pilots.
**Acting on the verdict.**
- `proceed` → freeze *Y*, *V*, the encoder revision, the serialisation, the channel parameters and the
  prompt version; commit `PREREGISTRATION.md` v1; start S3.
- `retune_once` → apply **exactly one** retune and re-gate. Not two.
- `fallback` → RQ3a becomes the headline. This is a planned branch, not a failure, and the simulator
  work becomes a documented negative result with a well-specified task behind it.

### E4 — RQ1 information-gradient sweep (DSE-020)

**Question.** RQ1 / H1 and H2.
**Design.** Factorial over conditions C0–C4 × serialisations × difficulty, self-play with the workhorse,
seeded start-pose jitter crossed with seeds for replication. Target roughly **50 episodes per condition,
about 250 total**, which is 6,000–12,000 model calls and single-digit GPU-hours; the pilot's measured gap
selects the exact row from the power table in `docs/methodology.md` §9.10.
**Emits.** Per-condition summaries with bootstrap intervals; C*k*-versus-C0 contrasts with Cliff's delta
and multiplicity correction; a mixed model of per-handoff progress; episode-level mediation of success
through episode-mean CPVI with a bootstrapped indirect effect; the shuffled-message audit; the
control-task selectivity figure; the absent-versus-unused two-by-two.
**Pre-registered secondary analyses.** Length-controlled partial Spearman and the length covariate on
the mediation's second path; the absent-versus-unused decomposition, split on the within-condition
median.
**Null reading.** A flat gradient is a task-design finding; a failed mediation routes to the
absent-versus-unused decomposition and is reported as a statement about which half of the channel failed.

### E5 — RQ1 robustness cells (DSE-021)

Heterogeneous pair and the serialisation A/B. Blocked on a small refactor (DSE-049): the runner currently drives
both roles from one client and needs `client_a` plus an optional `client_b`. Do the refactor before S3
so it does not have to be retrofitted across more call sites later.

### E6 — RQ2 measurement primitive (DSE-022)

**Question.** RQ2 / H3 and H4. Consumes E4's episodes; **no new model calls**.
**Emits.** Twin agreement (correlation and Bland–Altman, stratified by the sign of the retrospective
score); rank correlation of each runtime statistic with CPVI; AUROC of each statistic for predicting low
CPVI and for predicting failure, reported separately; the Jensen–Shannon bridge; encoder sensitivity
under a second pinned encoder.
**Null reading.** Agreement failing only on the negative stratum is the structural one-sidedness of the
KL twin and is a finding, not attenuation. If only probe-dependent statistics track CPVI, that bounds
the gate's deployability and is stated as a limitation rather than buried.

### E7 — Gate calibration (DSE-017 run)

Validates each statistic against **realised failure**, never against CPVI. Grouped out-of-fold scoring,
orientation flip, threshold chosen as the most aggressive within the firing-rate budget (default 0.2), Platt-mapped
expected calibration error with reliability bins and an explicit unreliable-below-N flag (N = 200).
**Emits.** The operating point and the chosen statistic, which E8 consumes.

### E8 — RQ3b causal gate (DSE-025)

**Question.** RQ3b / H6.
**Design.** Four modes over a subset of RQ1 conditions: gate off, gate active, matched-firing-rate
control, random-trigger control. Gating wins only if it beats **both** controls.
**Prerequisite that is easy to miss.** The gate's retry must issue a *different* prompt from the
original, or under greedy decoding every retry is a fixed point and the arm is vacuous while still
passing its unit tests (DSE-045). The feedback template is versioned and manifested.
**Emits.** Success and efficiency per mode with effect sizes and intervals, plus firing rate and retry
counts per arm.
**Null reading.** A clean null is a legitimate causal answer and is presented as one.

### E9 — RQ3a corpus spike (DSE-041)

**Question.** Does the primary substrate exist in the form the design assumes?
**Method.** Download TraceElephant; parse ten traces; confirm the per-step fields carry the receiver's
input context as well as the output; count traces, steps, extracted inter-agent handoffs, failures and
non-failures. Repeat for MAST-Data, specifically counting the non-failure proportion. Load ten Who&When
traces through the same interface with the reconstructed-observation flag set.
**Why it is first.** Everything in RQ3a rests on this corpus having the schema the design assumes.
Verify before writing the chapter, not after.
**Emits.** A counts table per corpus and a field-by-field schema mapping document.

### E10 — RQ3a replay labelling (DSE-042)

**Question.** What is *Y* on real logs?
**Method.** Counterfactual replay: re-run from step *t* with the step's output substituted; *Y* is
whether the outcome changed. Majority vote over *n* replays with the agreement rate reported as a
data-quality statistic and sub-floor steps flagged rather than dropped.
**Controls on cost and correctness.** A hard spend cap and a dry-run projection produced before any
execution; stratified step sampling recorded in the manifest; the trace-success label computed for
every trace regardless, so the refit arm survives if replay is cut.
**Discipline.** The labeller must not read the annotation fields at all, enforced by a signature test
rather than by convention.

### E11 — RQ3a localisation (DSE-024, rescoped)

Reports both regimes separately and never pooled: **transfer** (simulator-fitted probe applied directly)
and **refit** (probes fitted on held-out logs, grouped on trace identifier). Baselines are schema
validity and mean embedding cosine, with published attribution methods tabulated at their reported
numbers **and their dates**. The comparison to specialised tracers is made on the operating
characteristic — localisation per unit of compute, and availability before the outcome exists — not on
raw accuracy.

---

## 4. What must exist in code before each stage

| Before | What is missing | Size | Status |
|---|---|---|---|
| S1 | **A driver entry point** (DSE-031): `preceptx-pilot` / `preceptx-rq1` console scripts with a `--dry-run` pre-flight | S | **Built 24 Aug 2026** |
| S1 | **A structured-output mode for non-vLLM endpoints** (DSE-032): `ServingConfig.structured_mode`, `guided_json` vs `response_format.json_schema` | S | **Built 24 Aug 2026** |
| S1 | **A model-ladder benchmark** (DSE-005): throughput, TTFT, peak memory, schema adherence, C0 smoke, into one accumulating table | M | **Built 24 Aug 2026** |
| S3 | **A pinned encoder revision** (DSE-033): both encoders pinned to commit SHAs; the real load path refuses an unpinned one | S | **Built 24 Aug 2026** |
| S3 | Control-task selectivity and repeated cross-fits (DSE-043, DSE-044) | M | **Built 24 Aug 2026** — F0 is now gated on the rung-2 re-gate, not on these |
| S3 | Per-role clients on the runner (`client_a`, optional `client_b`) (DSE-049) | S | **Built 24 Aug 2026** |
| S5 | Gate integration and controls (DSE-018) | M | Open. DSE-045 (retry feedback template) **built 24 Aug 2026** |
| S6 | Corpus loaders and the replay labeller (DSE-041, DSE-042) | L | Open |

---

## 5. Recording a result so the thesis chapter writes itself

Every run gets a written entry the day it completes, in this format. The point is that a results chapter
is assembled from these entries rather than reconstructed from memory and log files in September.

```markdown
### <ID> — <name> — <UTC date>

- **Run id / manifest:** <run_id> · runs/<experiment>/<run_id>/manifest.json
- **Substrate:** local-<runtime> | myriad | interim-<provider>   ·  **Model + revision:** <id>@<sha>
- **Encoder + revision:** <id>@<sha>   ·  **Prompt / gate-template version:** <v> / <v>
- **Config hash / sweep hash:** <hash>   ·  **Seeds:** <list>   ·  **Episodes:** <n>
- **Command:** <exact command>

**What was asked.** One sentence.

**What came back.** Point estimates with intervals and effect sizes — never a bare p-value. For any
CPVI figure, report PVI and the PVI − CPVI gap alongside, and the control-task selectivity.

**How it reads.** Two or three sentences of interpretation, including the reading if the result is null.

**Seed sensitivity.** The headline gap across seeds, not overall success.

**Deviations.** Anything that departed from the pre-registration, with the reason. If none, say "none".

**What it changes.** The next action, or the freeze it unlocks.
```

Three reporting rules apply to every entry without exception. Never report a message-value number
without its state-only baseline and the gap. Never claim exact reproducibility of a run that involved a
language model — determinism here is low-variance, seed-pinned and revision-pinned. Never report bare
significance; effect sizes and intervals, always.

---

## 6. Freeze register

A result is frozen when its sweep is complete, its manifest is written, its analysis has run, its effect
sizes and intervals are reported, and its figure or table is committed. A frozen result is not silently
re-run: if it must change, that is an explicit re-freeze with a migration note, not an overwrite. F0's artefact is
`PREREGISTRATION.md` at the repository root: v0 is drafted during S1 while the system is fresh, and
v1 is committed the day the E3 verdict is `proceed`.

| Freeze | Contents | Precondition | Status |
|---|---|---|---|
| F0 Pre-registration | *Y* and *k*, conditioning semantics, *V* and its selection rule, encoder + revision, serialisation, C1/C3/C4 parameters, jitter and seed count, budgets, analysis protocol, gate feedback template, G1/G2/G3 thresholds, *R* repeats, the control-task expectation, the length controls | E3 verdict = proceed | **Will not fire.** The E3 ledger closed `fallback` on 29 Aug 2026. The file stays at v0; §8a now states which parts are amendable and under what protocol, and §8b is the amendment register |
| F1′ RQ1 (negative) | The absent gradient as the finding, with the per-condition receiver-competence mechanism, the length-matched control, and the absent-versus-unused decomposition | E3 re-gate complete | **Ready to freeze** — data and analysis in hand at `runs/rq1/af50c7c12d65540f/` |
| F2 RQ2 | Twin agreement, proxy tracking, encoder sensitivity, operating point | ~~E4~~ E3 re-gate episodes | **Ready to freeze** — analysis complete at `runs/af50c7c12d65540f-rq2/`; E7 calibration outstanding |
| F3 RQ3b | Causal contrast against both controls, against the direction declared in §8c | E8 complete | **Open** — direction pre-declared, run not submitted |
| F4 RQ3a | Localisation under the transfer regime, baselines table with dates; the refit regime and judge arm reported as unavailable with their reasons | E11 complete | **Transfer half frozen**; judge arm and refit regime open |

---

## 7. Results log

Entries in the §5 format, most recent last. Interim-substrate entries are marked as such in their
substrate line and inform decisions only; headline data is regenerated on the cluster.

### E0 — Task certification — 2026-08-24

- **Run id / manifest:** n/a — no model, no dataset; the oracle is deterministic and its output is
  asserted by `tests/integration/test_feasibility_certificate.py`
- **Substrate:** CPU only (no model in the loop)  ·  **Model + revision:** n/a
- **Encoder + revision:** n/a  ·  **Prompt / gate-template version:** n/a
- **Config hash / sweep hash:** n/a  ·  **Seeds:** n/a  ·  **Episodes:** 0
- **Command:** `uv run python -m preceptx.sim.feasibility`

**What was asked.** Is every difficulty in the ladder still solvable within its frozen step budget on
the current physics?

**What came back.** All three solvable, at the expected optima: easy 7 steps (budget 18, 21 A\*
expansions), medium 13 (budget 33, 5,455 expansions), hard 13 (budget 33, 5,114 expansions). Every
difficulty therefore has ≈2.5× headroom over its oracle optimum, and hard is not starved relative to
easy. Environment certified alongside: `ruff`, `ruff format --check`, `mypy --strict` and the full
test suite all clean on Python 3.11 with the `dev`, `embed` and `viz` extras installed.

**How it reads.** The geometry has not drifted since the budgets were frozen, so a downstream failure
cannot be an impossible task masquerading as a coordination failure. Medium and hard share an optimum
of 13 — the difficulty axis is slit width, and threading cost is not linear in it — which is expected
and already recorded in the design log, not a new finding.

**Seed sensitivity.** Not applicable; the oracle is exhaustive and deterministic.

**Deviations.** None.

**What it changes.** Unblocks S1. Model calls are now permitted.

### E1 — Serving smoke test and transcript read — 2026-08-24

- **Run id / manifest:** `8b20dad88e7f3514` (prompt v2) and `aff493e8f045ba92` (prompt v3) ·
  `runs/local/<hash>-run/manifest.json`
- **Substrate:** `local-lmstudio` — **interim, never pooled with cluster data** ·
  **Model + revision:** `mlx-community/Qwen3-8B-4bit`@`545dc4251c05440727734bcd94334791f6ab0192`
  (4-bit MLX quantisation of the 8B pilot tier, served by LM Studio at `localhost:1234/v1`,
  `structured_mode=response_format`, `thinking_switch=/no_think`)
- **Encoder + revision:** not used (no CPVI at this stage) · **Prompt version:** v2, then v3
- **Sweep hash:** `099bac32a23766f4` · **Seeds:** 0-4 · **Episodes:** 5 per prompt version
- **Command:** `run_grid` over C0 × numeric × easy × seeds 0-4, then `render_transcript`

**What was asked.** Does the whole loop run against a real model, and are A's messages grounded in
the true state?

**What came back.** The loop runs. The transcript, not the score, carried the finding.

| | prompt v2 | prompt v3 |
|---|---|---|
| Distinct A-messages | **7 of 75** | **66 of 76** |
| Actions chosen by B | `E` × 75 | `E` 53, `N` 15, `S` 8 |
| Median message length | 20 words | 45 words |
| Numeric mentions in messages | **0** | 552 |
| Episode success | 2 / 5 | 2 / 5 |
| Wall time | 193 s | 426 s |

Under v2, A emitted the same two sentences of generic advice all episode and B pressed east 75 times
out of 75. The cause was upstream of both: the `numeric` serialisation contained no wall or slit
geometry, so A could not describe the obstacle it was being asked to describe. v3 adds `walls_x` and
`slit_y` to the numeric form and rewrites both system prompts (see the design log, 2026-08-24).

**How it reads.** The v2 run would have produced a **flat information gradient in every condition**,
because seven near-identical messages carry nothing for a probe to find — a null that is
indistinguishable from the honest negative this design is prepared to report, and that reads as a G2
failure. It would also have made G3 vacuous rather than failed: zero numeric mentions is an empty
denominator, not a low score. **Episode success is identical across the two prompt versions (2/5),
which is the point**: v3 restored message variation and grounding without buying success, so this was
not a success-hacking prompt change. v3's messages are not always correct — one asserts that
y = 2.0074 lies within [2.1, 3.9] — but a wrong grounded claim is measurable by G3, whereas
no claim at all is not.

**Seed sensitivity.** Not meaningful at n = 5 on a 4-bit tier; recorded so the E3 comparison has a
baseline. Both prompt versions succeeded on the same 2 of 5 seeds.

**Deviations.** None from the pre-registration — `PREREGISTRATION.md` v0 explicitly allocates up to
three pre-E3 prompt bumps, and this is the first. Two serving-substrate adapters were added and are
recorded in the manifest (`thinking_switch`, `structured_mode`).

**What it changes.** Prompt v3 is now the version of record for the pilot; two of the three budgeted
bumps remain. Unblocks E2 and E3-local. **A local G1 verdict remains indicative only** — the verdict
of record is the bf16 re-gate on Myriad.

### E2 — Model-ladder benchmark, local row — 2026-08-24

- **Run id / manifest:** `runs/bench/ladder.{md,csv,json}` + `recommendation.md`; smoke dataset under
  `runs/bench/smoke/`
- **Substrate:** `local-lmstudio` — **one row of the ladder, not the ladder** ·
  **Model + revision:** `mlx-community/Qwen3-8B-4bit`@`545dc4251c05440727734bcd94334791f6ab0192`
- **Encoder + revision:** not used · **Prompt version:** v3
- **Seeds:** 0-9 · **Episodes:** 10 (C0 × numeric × easy)
- **Command:** `PRECEPTX_SERVING_SUBSTRATE=local-lmstudio uv run python scripts/benchmark_models.py
  --tier 8b-4bit --model mlx-community/Qwen3-8B-4bit --revision 545dc42… --base-url
  http://localhost:1234/v1 --structured-mode response_format --thinking-switch /no_think`

**What was asked.** What does the pilot tier cost per token, and can it drive the loop at all?

| Tier | Substrate | Mode | tok/s | TTFT (s) | Peak mem | Schema adherence | C0 smoke |
|---|---|---|---|---|---|---|---|
| 8b-4bit | local-lmstudio | `response_format` | 22.8 | 0.37 | n/a (no GPU) | **100%** (20/20) | **40%** (10 ep) |

**What came back.** Schema adherence is **perfect at 20/20** under `response_format` — the local
constrained-decoding path (llama.cpp/Outlines) honours the same schema vLLM's xgrammar will, which is
the specific risk DSE-032 was written against. Throughput of 22.8 tok/s and a 0.37 s short-completion
latency put a 918-call E3 cell at roughly 45 minutes on this machine, so the whole pre-cluster pilot
is affordable. The C0 easy smoke came in at **40%**, matching E1's 2/5 on the same cell.

**How it reads.** The harness **refuses to name a workhorse**, which is the correct output: 40% is
below the 0.5 capability floor, so the note says so rather than picking the only tier it has seen.
That 40% is a 4-bit quantised 8B on the *easiest* cell in the design — informative about the local
tier, and **not** evidence about the bf16 workhorse, which has no row yet and is therefore still a
placeholder. Peak memory is recorded as absent rather than zero, since there is no GPU here.

**Seed sensitivity.** 4 of 10 seeds succeeded; not analysed further at this n on an interim tier.

**Deviations.** None. The TTFT probe was changed from a one-token to an eight-token completion
before this row was taken, because a one-token request returns empty content from a runtime in
thinking mode and the client now rejects that (see the CHANGELOG).

**What it changes.** Confirms the local substrate is viable and cheap enough for E3-local. The
ladder's 14B and 32B rows stay open until Myriad, and until they exist the workhorse choice is
**undecided by evidence**.

### E3 — Formal pilot gate run, local substrate — 2026-08-24

- **Run id / manifest:** `caab86b866360bff` · `runs/local/caab86b866360bff-run/manifest.json`
- **Substrate:** `local-lmstudio` (**interim — not the verdict of record**) · **Model + revision:** `mlx-community/Qwen3-8B-4bit@545dc4251c05440727734bcd94334791f6ab0192`
- **Encoder + revision:** `BAAI/bge-base-en-v1.5@a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` · **Prompt / gate-template version:** v3 / n/a (no gate in the loop)
- **Config hash / sweep hash:** `caab86b866360bff` / `94e800eafa2bf240` · **Seeds:** 0, 1, 2 · **Episodes:** 18 (445 handoffs, 40 min wall)
- **Command:** `PRECEPTX_SERVING_SUBSTRATE=local-lmstudio uv run preceptx-pilot --model qwen8b --overrides model.name=mlx-community/Qwen3-8B-4bit model.revision=545dc4251c05440727734bcd94334791f6ab0192 model.tier=8b-4bit --base-url http://localhost:1234/v1 --structured-mode response_format --thinking-switch /no_think --concurrency 1 --timeout 300 --root runs/local`

**What was asked.** The three pilot gates over the documented E3 cell — C0/C1/C4 × easy/hard × seeds 0–2 — to establish whether the headline task is viable, indicatively, at the 4-bit 8B local tier.

**What came back.** Verdict `retune_once`; one gate of three passed.

| Gate | Value | Threshold | Verdict |
|---|---|---|---|
| G1 capability (easy C0 success) | 0.000 (0/3) | ≥ 0.5 | **FAIL** |
| G2 signal (C0 − C4 success gap) | −0.167 | ≥ 0.1 | **FAIL** |
| G3 groundedness | 0.811 | ≥ 0.8 | **PASS** |

Success by condition (6 episodes each, easy+hard): **C0 0/6, C1 2/6, C4 1/6** — the clean channel came
last. All 9 hard episodes, in every condition, ended jammed at the *first* wall (final `com_x` 3.05–4.09
against a wall at x = 4), so the hard cell contributed a pure floor effect.

Information at the boundary, on the C0+C4 subset (n = 305 handoffs, *Y* = `y_binary_progress`, base rate
0.711, 2 000-resample percentile intervals):

| Quantity | C0 | C4 |
|---|---|---|
| CPVI | +0.033 [+0.010, +0.057] bits | +0.026 [−0.005, +0.055] bits |
| PVI | +0.033 [+0.010, +0.057] bits | +0.026 [−0.005, +0.055] bits |
| **PVI − CPVI gap** | **+0.0000** [−0.0005, +0.0005] bits | **+0.0001** [−0.0004, +0.0007] bits |

**How it reads.** The gate values above are the ones the *shipped* code produced, and three of them were
wrong. Auditing them against the raw data found four defects in the pilot gate itself, all corrected on
this branch and recorded as D19–D21 and in `docs/experiment_design_log.md`; re-gated with the corrected
code this run reads **G1 FAIL 0.000, G2 FAIL on the success half (−0.167) with a CPVI gap of +0.012 bits,
G3 PASS 0.999** (not 0.811 — G3's truth set had been the load body only, so a sender citing the wall and
slit coordinates the serialiser printed for it was scored as fabricating them). What the failures mean:
**G1 is a capability reading and is indicative only.** The easy episodes were not lost to coordination
breakdown — all three reached chamber three (final `com_x` 8.44, 8.91, 9.17 against a goal at x = 10,
r = 0.8) and ran out of an 18-step budget short of the goal, having spent steps on a repeated alignment
error: the model asserts the load is "aligned with the slit" when its `com_y` is *outside* the printed
slit range (2.0074 called within (2.1, 3.9)), then pushes east into the wall. **G2's success half fails
in the wrong direction** — C4 outscored C0 — which with C0 at 0/6 has no headroom to mean anything yet.

**Seed sensitivity.** C0 scored 0 of 2 episodes at every one of the three seeds. The per-seed C0 − C4
success gap is 0.0, 0.0 and −0.5, so the overall −0.167 rests entirely on one C4 episode at seed 2, and
C1's two successes came from seeds 0 and 1. At two episodes per condition per seed no gap is separable
from noise, which is the honest statement at this scale; the CPVI gap (+0.007 bits) likewise sits well
inside its own interval. E3-local sizes the pipeline, not the effect.

**Deviations.** Two from the shipped code, both corrected on this branch before the numbers above were
taken, and both recorded in `docs/experiment_design_log.md`. (i) G1 averaged C0 success across *both*
difficulties, contradicting the pre-registered "at easy difficulty"; corrected to easy-only, which leaves
this verdict unchanged (0/3 easy rather than 0/6 mixed) but would have read a pair solving every easy and
no hard episode as exactly 0.5, passing the floor by accident. (ii) C3's numeric restriction blacklisted
only the `goal=` line, so the `walls_x=`/`slit_y=` lines added by v3 were still delivered to B and C3's
asymmetry was only nominal; replaced with a whitelist of B's own state, which fails closed when a key is
added. Neither affected any C0/C1/C4 record in this run.

**What it changes.** The retune is **not** spent: the pre-registration states that a local G1 failure at
4-bit is indicative and must be reproduced at bf16 before the ladder is invoked, and the one-retune ledger
opens at the Myriad re-gate. Two design decisions are now open and are the gate on re-running E3: whether
C3 joins the pilot cell so G2's CPVI half is testable at all, and whether the serialisation names the
load's own extent so that "aligned with the slit" is decidable from the prompt. Transcript for the human
read: `runs/local/e3_transcript_c0_easy.md` (interim, gitignored).

### E3 — Formal pilot gate run, corrected cell and corrected gates — 2026-08-24

- **Run id / manifest:** `c0bd4d7499f01d97` · `runs/local/c0bd4d7499f01d97-run/manifest.json`
- **Substrate:** `local-lmstudio` (**interim — not the verdict of record**) · **Model + revision:** `mlx-community/Qwen3-8B-4bit@545dc4251c05440727734bcd94334791f6ab0192`
- **Encoder + revision:** `BAAI/bge-base-en-v1.5@a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` · **Prompt / gate-template version:** v4 / n/a
- **Config hash / sweep hash:** `c0bd4d7499f01d97` / `c163d616d1608140` · **Seeds:** 0, 1, 2 · **Episodes:** 24 (599 handoffs, 65 min wall)
- **Command:** as the previous entry, on the corrected cell (`_PILOT_CONDITIONS = ["C0","C1","C3","C4"]`)

**What was asked.** Re-run E3 after the four gate defects the first run exposed were fixed, with C3 in
the cell and prompt surface v4 (`load_size`, and A told the whole load must clear the gap).

**What came back.** Verdict `retune_once`. **G1 FAIL** 0.000 (0/3 easy C0) · **G2 FAIL** on its success
half, −0.333 · **G3 PASS 0.977**.

CPVI by condition, one probe fitted over all 599 handoffs, *Y* = `y_binary_progress` (base rate
0.482), 2 000-resample **episode-cluster** percentile intervals (episodes resampled with
replacement, handoffs pooled; corrected 2026-08-24 — this entry first carried handoff-level
intervals, e.g. C0 [+0.141, +0.243], which treat 153 handoffs from 6 episodes as independent and
read roughly half the honest width):

| | C0 | C1 | C3 | C4 |
|---|---|---|---|---|
| CPVI | **+0.192** [+0.059, +0.307] | +0.083 [−0.197, +0.320] | +0.051 [+0.002, +0.103] | −0.018 [−0.081, +0.038] |
| PVI | +0.156 [+0.046, +0.244] | +0.135 [−0.204, +0.416] | +0.088 [−0.002, +0.203] | −0.007 [−0.056, +0.052] |
| **PVI − CPVI** | −0.036 | +0.051 | +0.037 | +0.012 |
| Episode success | 0/6 | 1/6 | 0/6 | 2/6 |

**How it reads.** **The information half of H1 holds and the outcome half cannot yet be assessed.** CPVI
falls monotonically with channel degradation — C0 > C1 > C3 > C4, with C4 at zero and C0's interval
excluding it — which is the predicted direction, on 6 episodes per condition, from a model that never
solved the task. Episode success went the other way (C4 2/6 against C0 0/6), and with C0 at zero there
is no headroom for success to fall, so the outcome contrast has nothing to measure until a tier that
can do the task is in the loop. Read together: the *instrument* is producing signal and the *task* is
above this tier. **C3's CPVI is positive — its cluster interval just clears zero — but lower than C0's**, which methodology
§8.3's "most room to be positive" did not predict; at six episodes per condition that is flagged for
the bf16 re-gate, not explained. **v4 fixed what it was meant to fix and bought no success**: A now
says "The load's center is at y=2.03, which is below the slit's lower bound of 2.1" where v3 said
"currently aligned with the slit … since its y-position (2.0074) is within the slit's y-range", and
easy C0 stayed 0/3 — so the alignment error was a genuine prompt-surface defect and *not* the cause of
the G1 failure. The third prompt bump stays unspent.

**Seed sensitivity.** The three successes are C1/easy/seed 1 (11 steps), C4/easy/seed 2 (13) and
C4/easy/seed 0 (14); no condition succeeded at more than two seeds and C0 and C3 succeeded at none. At
6 episodes per condition no success contrast is separable from noise. The CPVI ordering is the stable
part of this run, and even it is a pilot-scale observation, not an effect estimate.

**Deviations.** Four corrections to the gate implementation and two design decisions, all pre-freeze,
all logged (`docs/experiment_design_log.md`, D19–D21). The gate corrections: G2 fits the CPVI probe
once over the whole cell rather than refitting on the C0-plus-hardest subset (which read the same
data's gap as +0.012 rather than +0.211 bits); G3 takes its truth set from what the sender was shown
rather than from the load body alone (0.977 rather than 0.720); G1 scores easy C0 only; C3's numeric
restriction whitelists B's own state. The design decisions: C3 added to the pilot cell, and prompt
surface v4 — bump two of three, one remaining.

**What it changes.** The retune is still **not** spent (pre-registration: a 4-bit local G1 failure is
indicative and must be reproduced at bf16). E3 is now blocked on the Myriad bf16 re-gate, which is the
verdict of record, and the local pilot has done what it exists to do: four instrument defects found and
closed, and a CPVI gradient in the predicted direction to size the bf16 run against. Transcripts for
the human read: `runs/local/e3v4_transcript_c0_easy.md`, `runs/local/e3v4_transcript_c3_easy.md`.

**Last-pass audit (2026-08-24; same dataset, no new model calls; artefact:
`runs/local/c0bd4d7499f01d97-report/reanalysis.json`).** Three checks run before handing the design
to Myriad. (i) *Determinism:* re-scoring the dataset reproduces every point estimate above to the
digit — the grouped folds and probe fits are deterministic given record order. (ii) *Probe overfit
monitor:* held-out AUROC 0.727 for `g_cond` against 0.566 for `g_base` and 0.807 in-sample — the
conditional probe genuinely predicts progress out-of-fold and the in/out gap is modest, so
cross-fitting is doing its job at dim 1536 on n = 599. (iii) *Shuffled-message null (RD-15):* the
real pooled mean CPVI (+0.078 bits) exceeds all 20 within-condition permutations (null
+0.043 ± 0.006, max +0.057; permutation p ≈ 1/21) — but the null does **not** collapse to zero, and
cannot: within-condition permutation preserves the condition-level signatures message style
carries, and per-handoff progress base rates differ by condition (C0 0.255, C1 0.735, C3 0.379,
C4 0.575), so a permuted message still betrays its condition. The pre-registered "must collapse"
criterion is corrected to the permutation-test form, pre-freeze (PREREGISTRATION §8); the null's
height now reads as the identity component of CPVI and the real-minus-null excess (+0.035 bits) as
per-handoff message content. The C0 − C4 CPVI gap of +0.211 bits holds a cluster interval of
**[+0.060, +0.349]** — the headline pilot observation survives the honest uncertainty.

**Frozen-estimator re-score (2026-08-24; same dataset, no new model calls; artefact:
`runs/local/c0bd4d7499f01d97-report/rescore_frozen_estimator.json`).** DSE-043 and DSE-044 were
built before the Myriad re-gate rather than after it, so the verdict of record is produced by the
estimator that gets frozen at F0. The v4 table above **stands as recorded** — it is the dated
reading of the estimator of the day; this is the reading of the frozen one (ℓ₂ logistic, C = 1.0,
`n_splits = 5`, **R = 5 repeated cross-fits**, PREREGISTRATION §5) on identical records.

| | C0 | C1 | C3 | C4 |
|---|---|---|---|---|
| CPVI (R = 5) | **+0.181** [+0.050, +0.302] | +0.050 [−0.250, +0.317] | **+0.058** [+0.013, +0.108] | −0.031 [−0.084, +0.020] |
| control CPVI | −0.005 | −0.003 | −0.005 | −0.011 |
| selectivity | **+0.187** | +0.053 | +0.062 | −0.020 |
| per-handoff `cpvi_sd` | 0.033 | 0.076 | 0.025 | 0.036 |

Four readings matter. (i) *The headline is invariant:* the C0 − C4 gap is **+0.212** bits against
+0.211 under the recorded estimator. (ii) *The probe is selective:* control-task CPVI — the same
features, splitter and probe family against random labels drawn at the observed base rate — is
**−0.006 bits** pooled, so selectivity (**+0.072** pooled, **+0.187** in C0) is essentially the
whole score. Negative is the pre-registered expectation, not a surprise: `g_cond` carries twice the
features, so against noise labels it overfits harder and scores worse held out. The check is not
inert — an almost-unregularised probe at n = 30, d = 128 reads +0.93 — so PREREGISTRATION §5's
capacity ladder has something to fire on; it does not fire here. (iii) *Repeats were needed:* the
mean per-handoff across-repeat SD is **0.042 bits**, comparable to C3's whole mean CPVI, and the
pooled point estimate moves 16% (+0.078 → +0.066) between one fold assignment and the mean of five.
(iv) *CPVI is not message length:* Spearman(CPVI, delivered token count) = **−0.085**. The
permutation test still passes at R = 5 (real +0.066 against a null of +0.033 ± 0.006, max +0.046,
p = 1/21), and the pilot verdict is unchanged (`retune_once`; G1 0.000, G3 0.977).

### E9 — RQ3a corpus spike — 2026-08-24

- **Run id / manifest:** n/a — a corpus spike, no episodes and no model calls · counts reproduced by the command below
- **Substrate:** none (CPU + network only) · **Model + revision:** none
- **Encoder + revision:** none · **Prompt / gate-template version:** n/a
- **Corpus revisions:** `TraceElephant/TraceElephant@a78a57cd`, `Kevin355/Who_and_When@59b9fcba`, `mcemri/MAST-Data@95118ac9`
- **Command:** `scripts/fetch_rq3a.sh <root>` then `uv run python -m preceptx.experiments.rq3a_load --root <root>`

**What was asked.** Does the primary RQ3a substrate exist in the form the design assumes — per-step
receiver input context, and a two-class trace outcome?

**What came back.** Half of it.

| Corpus | Traces | Steps | Inter-agent handoffs | Failures | Non-failures | Observations |
|---|---:|---:|---:|---:|---:|---|
| TraceElephant | 220 | 5,960 | 2,488 | 220 | **0** | recorded |
| Who&When | 184 | 4,092 | 3,505 | 184 | **0** | reconstructed |
| MAST-Data | 1,642 | — | — | 1,237 | **405** (24.7%) | trace-level only |

*Confirmed:* TraceElephant records the receiver's input context per step — as `input.messages`, an
OpenAI-shape object, not the string `input_context` the roadmap named. 2,488 inter-agent handoffs
across five run families and three multi-agent systems. This is the reason the substrate moved off
Who&When and it holds.

*Falsified:* the corpus is **220 traces, not 380**, and all 220 are annotated failures. Only the 44
`swe-agent` traces carry an annotation-free outcome (`tests_status`); **0 of 44 pass every test**.
The other 176 have no outcome field at all. Trace-level outcome on the primary corpus is
single-class.

*Counted, closing one of DSE-047's two unverified numbers:* MAST's non-failure class is **405 of
1,642 (24.7%)** — the assumption holds — but it is severely unbalanced across systems (AG2 52.1%,
OpenManus 3.3%), so a pooled MAST probe can score by recognising the system rather than by reading
the message.

**How it reads.** The conditioning state RQ3a needs is real and plentiful; the outcome is not. *Y1*
(trace success) is degenerate on both step-level corpora, so **DSE-042's counterfactual replay moves
from an upgrade path to the load-bearing route** to a two-class step-level *Y*. *Y2*
(annotation-as-Y) stays forbidden — and this is exactly the pressure that makes it tempting, which
is worth naming now rather than rationalising later. The transfer regime is unaffected: a
simulator-fitted probe needs log labels only to evaluate, and `mistake_step` exists on all 220.

**Seed sensitivity.** n/a — no stochastic component; the counts are exact at the pinned revisions.

**Deviations.** None from the pre-registration, which does not yet cover RQ3a. Two roadmap §3.4
claims are corrected in place with the correction named, and DSE-041's assumed field names
(`input_context` / `output_content`) are corrected to `input` / `output`.

**What it changes.** DSE-042 is promoted on the RQ3a critical path and its spend cap becomes a
load-bearing control rather than hygiene. Any MAST arm must stratify by `system_name` or report the
system-identity-only baseline beside it. Methodology §9.8's *Y* table needs the Y1 row marked
unavailable at step level. Remaining before E10: the second DSE-047 number (AgenTracer accuracies)
still rests on a secondary summary and needs checking against the paper's own result table.

### E3 — Formal pilot gate run, Myriad bf16 — **attempt 1, the verdict of record** — 2026-08-26

- **Run id / manifest:** `1c994b87bbca8257` · `runs/1c994b87bbca8257-run/manifest.json` · report
  `runs/1c994b87bbca8257-report/pilot.{md,json}` · job log `precept-pilot.o214590`
- **Substrate:** `myriad-nvidia-a100-pcie-40gb` — **the verdict of record; the first non-interim
  data this project has produced** · **Model + revision:**
  `Qwen/Qwen3-14B`@`40c069824f4251a91eefaf281ebe4c544efd3e18` (bf16, vLLM, guided JSON)
- **Encoder + revision:** `BAAI/bge-base-en-v1.5`@`a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` ·
  **Prompt / gate-template version:** v4 / n/a (no gate in the loop)
- **Config hash / sweep hash:** `1c994b87bbca8257` / `2fb965…` · **Seeds:** 0–4 ·
  **Episodes:** 40 (965 handoffs, 1,287 s of sweep inside a 24 m 46 s job)
- **Command:** `qsub scripts/myriad/pilot.sh` (SGE job 214590, exit 0; `-ac allow=L`, no `-P`)

**What was asked.** The three pilot gates over the pre-registered E3 cell — C0/C1/C3/C4 × easy/hard
× seeds 0–4 — at bf16 on the cluster. This is the run the one-retune ledger opens at: both earlier
E3 runs were 4-bit local and indicative by construction.

**What came back.** Verdict `retune_once`. One gate failed, one passed with margin, one passed by
exactly zero margin.

| Gate | Value | Threshold | Verdict |
|---|---|---|---|
| G1 capability (easy-C0 episode success) | 0.400 (2/5) | ≥ 0.5 | **FAIL** |
| G2 signal — success half (C0 − C4) | 0.100 (2/10 − 1/10) | ≥ 0.1 | **PASS, by zero margin** |
| G2 signal — CPVI half (C0 − C4) | **+0.243 bits** | > 0 (directional) | **PASS** |
| G3 groundedness | 0.986 (965/965 records cite numbers) | ≥ 0.8 | **PASS** |

Per-condition CPVI, one probe fitted over all 965 handoffs, *Y* = `y_binary_progress`, R = 5
repeated cross-fits: **C0 +0.228, C4 −0.015**. Control-task CPVI on random labels **−0.004**,
**selectivity +0.136** — so a probe of this capacity did not manufacture the score. Episode success
by condition: **C0 2/10, C1 2/10, C3 1/10, C4 1/10**.

Two things must be said about the two gates that passed, because the headline table flatters both.
**G2's success half is one episode.** 0.100 is 2/10 minus 1/10; flip any single episode in either
arm and the gate fails. **G1 has almost no power at this n.** 2/5 carries a Wilson 95 % interval of
**[0.12, 0.77]**, and a design whose true success rate is exactly the 0.5 threshold fails this gate
**50 %** of the time (true 0.6 fails 32 %, true 0.7 fails 16 %). Neither number is evidence about
the design at the precision the verdict implies. The **CPVI half is the one solid measurement in the
run**: a +0.243-bit gradient with the control at zero, on 965 handoffs, from a pair that solved 6
episodes out of 40.

**How it reads. The instrument works and the task is jammed, and the jam has a named mechanism.**

*Every failure ran to the step budget.* Maximum handoffs for the cell is 20 × 18 (easy) + 20 × 33
(hard) = 1,020; 965 were written with 6 successes. `agents/graph.py` has exactly one exit,
`done = success or next_step >= max_steps`, so all 34 failures spent their whole budget. Nothing
crashed, nothing timed out, every one of the ~1,930 completions returned HTTP 200.

*But they did not run out of road — they ran on the spot.* Reconstructing the action sequences from
the Parquet (`action["action"]` per step, ordered):

| Failure mode | Episodes | Signature |
|---|---|---|
| Period-2 rotation cycle | 5 | `ROT+,ROT-,ROT+,ROT-,…` for up to 27 consecutive steps |
| Period-2 translation cycle | 6 | `N,S,N,S,…` — one episode alternates for 26 of 33 steps |
| Period-1 push cycle | 7 | `E` repeated into a wall it cannot pass — `E×18`, `E×33` |
| Mixed thrash (no clean cycle ≥ 6) | 16 | |
| **Success** | **6** | |

**18 of 40 episodes — 53 % of the failures — end in a period-1 or period-2 limit cycle, and the
cycle consumes 68 % of the steps those episodes spent** (mean terminal cycle 18.4 actions of a mean
27.2-step episode).

*The obvious remedy was tested against the data and refuted.* The natural first reading of "every
failure hit the budget" is that the budget is too small. It is not. **Mean geodesic distance still
to run at the end of a failed episode is 7.02** against a goal radius of 0.8, in a task whose total
geodesic span is about 8 — and **only 1 of 34 failures ends within 1.5 of the goal**. Meanwhile
every success finished in **8, 8, 8, 9, 12 and 12 steps** against an easy budget of 18 and an oracle
optimum of 7. The budget is generous for anything that works and irrelevant to anything that does
not: extending it buys more cycling, not more arrivals. This hypothesis was raised, tested and
dropped before any parameter was touched, and it is recorded here because the refutation is the
evidence that the retune below is aimed at the right thing.

*The mechanism is a fixed point of a memoryless greedy policy.* Decoding is greedy
(`temperature = 0`) by design. The v4 prompt surface contains the current scene and nothing else —
no action history, no record of what the previous action achieved. So the policy is a pure function
of the current state. Any state *s* whose action *a* returns the system to *s* (a wall press, a pure
rotation) is a **period-1 fixed point**, and any pair *s → s′ → s* is a **period-2 cycle**; in both
cases the agents cannot escape, because escaping would require the prompt to differ and the prompt
is a function of the state alone. This is not a capability failure and it is not stochastic: it is a
structural property of a deterministic memoryless policy on a deterministic environment. Note the
repository had **already reasoned this out for the gate** — `GATE_FEEDBACK` exists precisely because
"under greedy decoding a re-prompt is a fixed point" — and had never applied the same argument to
the base loop.

*Corroborating detail: the failure is not model capability.* The same cell at 4-bit 8B scored 0/3 on
easy C0; bf16 14B scores 2/5. Real improvement, same wall. And at easy seed 0 the action sequences
under **C0, C3 and C4 are byte-identical** (`N,E,E,E,E,E,E,E`) — where the task is trivially
solvable the channel is irrelevant, which is a ceiling effect on the easiest cell rather than a
broken channel (every other seed diverges across conditions).

*A condition-specific finding worth carrying forward.* **C1 (the 8-token length cap) degenerates
into a single-action policy**: 7 of its 8 failures are pure cycles, 5 of them `E` repeated for the
entire budget (`E×33` three times over). A truncated instruction collapses to a direction, and B
executes that direction until the budget ends. This is *why the success gradient is flat rather than
merely noisy*: at easy seed 1 the capped channel **succeeded in 8 steps** where the clean channel
spent 15 straight `ROT-`. A degraded channel can outscore a clean one when the degraded message
happens to encode the near-optimal policy and the clean one talks the actuator into rotating. That
is a systematic mechanism, not sampling noise, and it is the honest explanation for C0 2/10 against
C4 1/10.

*One measurement instrument was blind to all of this.* The `stuck` field scored **False for all 18
handoffs** of the `N,S,N,S,…` episode and for the `E×18` wall press, because pre-v5 `detect_stuck`
tested the *span* of the COM over a 3-state window: an alternating trajectory genuinely moves a full
unit each step, and a jittering wall press exceeds the 0.02 threshold. The field that exists to name
a trajectory going nowhere missed the failure mode that consumed the run. See "What it changes".

**Seed sensitivity.** All 6 successes are easy-difficulty; **0 of 20 hard episodes succeeded in any
condition**. Success concentrates on seeds 0, 1 and 3 (start poses already near the slit line) and is
absent at seeds 2 and 4 entirely — at seed 2 the load never moves in *x* at all in four of the eight
cells, ending at its start abscissa of 1.27. The C0 − C4 success gap is carried by a single episode,
so it has no per-seed structure to report. The CPVI ordering is again the stable part of the run.

**Deviations.** None from the pre-registration for the run itself: the executed cell, the thresholds,
the estimator, the encoder and the model revision are the pre-registered ones, and the jobscript
carries neither the revision (read from `configs/model/qwen14b.yaml`) nor the grid axes (the CLI
defaults) precisely so the executed and pre-registered cells cannot drift. The retune applied *after*
this run is recorded under "What it changes" and amends PREREGISTRATION §6 prospectively.

**Working on Myriad — what this run cost to make possible.** Four defects stood between a correct
local pipeline and a single successful cluster job, each found only on the cluster and each fixed as
its own ticket:

| | Symptom | Cause | Fix |
|---|---|---|---|
| DSE-051 | `uv sync` cannot build an environment at all | Myriad login *and* compute nodes are RHEL 7.9 / glibc 2.17; every locked wheel is manylinux_2_28+, and torch ships no sdist | Run inside a digest-pinned Debian-bookworm Apptainer image (glibc 2.36). `uv.lock` unchanged — the lock stays the anchor and simply executes where its wheels are valid |
| DSE-052 | vLLM structured-output flag mismatch | vLLM's config surface moved | Assert `structured_outputs_config` on a login node, no GPU needed |
| DSE-053 | `FileNotFoundError: 'icc'` minutes into vLLM start-up, A100 already allocated | Myriad's login shell exports `CC=icc` from `compilers/intel/2018`; Apptainer passes the host environment straight through; Triton JIT-compiles a CUDA shim at engine start and reads `CC` with no existence check | Override `CC`/`CXX` unconditionally inside the container — `${CC:-gcc}` is useless here, the broken value is already set so the default never fires |
| DSE-054 | Job 212796 died in under a second | SGE executes a **spooled copy** of the jobscript from `/var/opt/sge/<node>/job_scripts/<jobid>`, so `BASH_SOURCE` names the spool directory and `$HERE/_common.sh` does not exist | Recover the checkout from `$PWD` (`#$ -cwd` puts us in the submit directory) |

Three habits that paid for themselves, worth stating as method rather than as anecdote: **(i)** serve
and drive in **one** job — a login node driving a compute node's `localhost` reaches the wrong
machine, and splitting them means two queue waits and a hostname discovery problem; **(ii)** warm the
embedding encoder **before** any GPU time is spent, because it is first used at analysis time and a
node without outbound network would otherwise fail after a full GPU-hour with the dataset already
paid for; **(iii)** derive the served model *and its revision* from the same `configs/model/*.yaml`
the manifest records them from, so a job physically cannot serve one checkpoint while claiming
another — `/v1/models` carries no revision, so nothing else in the repository could catch that.
Measured throughput for sizing later runs: **~1.5 model calls/s at concurrency 4**, 150–165 tok/s
generation, 56–57 % prefix-cache hit rate, 965 handoffs in 1,287 s on one A100-PCIE-40GB.

**What it changes. This spends the one permitted retune (PREREGISTRATION §6), as prompt surface v5.**

1. **The retune — `recent=` in the prompt surface (PROMPT_VERSION v4 → v5).** The state now carries
   the last four actions and the geodesic distance each gained, oldest first, plus their net. The
   prompt therefore differs on the second visit to a state, which is exactly what a fixed point
   requires to stop being one. It is deliberately **fact, not instruction** — it reports what was
   done and what it gained, and leaves "so try something else" as the agent's inference; a directive
   would make this a behavioural intervention rather than an observability fix, and the two are not
   separable after the fact. The line is appended **after** `apply_channel`, so the channel still
   degrades exactly what it degraded before: C3 windows B's view of the *world*, and B's memory of
   its own actions is not the world. The same line is appended to all three serialisations, so the
   serialisation axis stays a contrast over representation.
2. **The cell widens to seeds 0–9 (80 episodes).** A precision change, not a retune: it moves no
   threshold and no estimator. It is not optional stopping, and the direction is the proof — the
   attempt-1 point estimate (0.4) sits **below** the 0.5 threshold, so added *n* moves the expected
   verdict **toward FAIL**. It also doubles G2's success half, which currently rests on one episode.
3. **`detect_stuck` measures net displacement across the window, not span** (window 3 → 5 states).
   Diagnostic only — no gate reads `stuck` and nothing terminates on it — but the retune run has to
   be able to *see* the pathology it targets, and the pre-v5 field could not.
4. **The step budget is deliberately unchanged** at 2.5 × the oracle optimum. See the refutation
   above; this is a recorded decision, not an omission.

Attempt 2 lands in a **different dataset** by construction: `PROMPT_VERSION` and the per-difficulty
budgets both feed `dataset_hash_for`, so v5 resolves to `eddd19c654515bb2` and `1c994b87bbca8257`
can be neither resumed into nor overwritten. **Attempt 1 is preserved as evidence** — dataset,
Parquet, manifest, report and job log — and remains a valid result even though it does not clear the
gate. If attempt 2 still fails, the fallback ladder fires and RQ3a becomes the headline; there is no
attempt 3.

---

### E3 — Formal pilot gate run, Myriad bf16 — **attempt 2, the re-gate; verdict `fallback`** — 2026-08-26

- **Run id / manifest:** `eddd19c654515bb2` · `runs/eddd19c654515bb2-run/manifest.json` · report
  `runs/eddd19c654515bb2-report/pilot.{md,json}` · diagnostic `runs/cycles-a1-vs-a2.txt`
- **Substrate:** `myriad-nvidia-a100-pcie-40gb` · **Model + revision:**
  `Qwen/Qwen3-14B`@`40c069824f4251a91eefaf281ebe4c544efd3e18` (bf16, vLLM 0.18.1, torch 2.10.0,
  guided JSON)
- **Encoder + revision:** `BAAI/bge-base-en-v1.5`@`a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` ·
  **Prompt version:** v5 · **Git SHA:** `a327080` · **Simulation digest:** `aea80c8ea9faf072`
- **Sweep hash:** `abdb854838992a87` · **Seeds:** 0–9 · **Episodes:** 80 (1,925 handoffs,
  2,199.7 s of sweep) · **Timestamp:** 2026-08-26T21:19:15Z
- **Command:** `qsub -v ATTEMPT=2 scripts/myriad/pilot.sh`

**What was asked.** The re-gate after the one permitted retune (PREREGISTRATION §6), on prompt v5
and the widened seeds 0–9 cell. A second failure fires the fallback ladder; there is no attempt 3.

**What came back.** Verdict `fallback`. Two gates failed.

| Gate | Value | Threshold | Verdict |
|---|---|---|---|
| G1 capability (easy-C0 episode success) | 0.300 (3/10) | ≥ 0.5 | **FAIL** |
| G2 signal — success half (C0 − C4) | **−0.200** (0.150 − 0.350) | ≥ 0.1 | **FAIL, sign inverted** |
| G2 signal — CPVI half (C0 − C4) | **+0.100 bits** (0.153 − 0.053) | > 0 (directional) | **PASS** |
| G3 groundedness | 0.999 (1,923/1,925 records cite numbers) | ≥ 0.8 | **PASS** |

Control-task CPVI **−0.002**, **selectivity +0.128** — the probe did not manufacture the score.
Episode success by cell: C0 easy 3/10, C1 easy **7/10**, C3 easy 1/10, C4 easy **7/10**;
**every hard cell 0/10**.

#### The verdict is procedurally correct and its stated reason is wrong

Both gates failed as pre-registered and the ladder fires as written. But the failure is **not** the
absence of an information gradient, and it is not a capability ceiling. Three numbers separate the
gate's arithmetic from what the run actually established:

| Statistic | Value | Reading |
|---|---|---|
| G1 at 3/10 | Wilson 95 % **[0.11, 0.60]** | still spans the 0.5 threshold — underpowered even at *n* = 10 |
| C0 vs C4, easy, episode success (3/10 vs 7/10) | Fisher **p = 0.179** | the sign inversion that failed G2 is **not** statistically established |
| C0 vs C4 handoff-level stuck rate (0.350 vs 0.205) | episode-cluster **p = 0.085**, 95 % CI **[−0.005, +0.294]** | the difference is **not** established either |

**Corrected 2026-08-27 (DSE-057).** This row previously read "Fisher *p* ≈ 10⁻⁷ nominal — the
mechanism underneath it is not in doubt", followed by a sentence asserting that "the effect survives
a cluster correction comfortably but the exact figure needs one". The figure has now been computed
and it **contradicts** that assertion. Resampling episodes rather than handoffs (20 episodes per
condition, 4 000 resamples, seed 0) gives:

| Contrast (handoff stuck rate) | Difference | Episode-cluster 95 % CI | Permutation *p* |
|---|---|---|---|
| C0 − C1 | +0.119 | [−0.040, +0.281] | 0.165 |
| C0 − C3 | −0.116 | [−0.264, +0.037] | 0.152 |
| C0 − C4 | +0.145 | [−0.005, +0.294] | 0.085 |

Every interval crosses zero. The nominal 10⁻⁷ was **entirely** an artefact of treating ~500
clustered handoffs as independent when the design has only 20 episodes per condition. No statistical
claim in this entry rests on the handoff-level contrasts any more.

What survives is not statistical. **The C1-hard signature is categorical**: 10/10 failures are the
literal action sequence `E,E,E,E,E,E`, with no test required. And the task-validity finding below is
an *exhaustive search result*, not an estimate. **The gate failed on an underpowered statistic, and
the reason to change the design is a proof about the task rather than a *p*-value** — which is why
this entry still ends in a design change rather than in a null, but on different grounds than it
originally claimed.

#### What the retune did, measured

v5 was aimed at the greedy fixed point diagnosed in attempt 1. **It hit its target and missed the
outcome.** Terminal period-1/2 cycling among failed episodes, `scripts/diagnose_cycles.py`:

| Cell | attempt 1 (v4) | attempt 2 (v5) |
|---|---|---|
| C0 easy — cycled | 0.667 | **0.143** |
| C0 hard — cycled | 0.600 | 0.600 |
| C0 easy — success | 0.4 (2/5) | 0.3 (3/10) |
| whole run — failures ending in a cycle | 21/34 = 62 % | 28/62 = **45 %** |

C0-easy periodicity collapsed and the success rate did not follow. Breaking the fixed point was
**necessary and not sufficient**, and the reason is the next section: the erroneous instruction that
drove the cycle is regenerated fresh at every step, so B keeps receiving it however much of its own
history it can see. v5 converted clean limit cycles into aperiodic rotational wandering
(`ROT-,ROT-,ROT-,ROT-,S,E`) without making the policy goal-directed.

#### The mechanism: the channel degrades a wrong instruction, not information

The T's **y-extent never exceeds 1.553** at any orientation in the circle (minimum 1.300 at 0°,
maximum 1.553 at −146.8°), and the easy slit is **1.8**. The load therefore clears the gap head-on
**from every possible angle**, with at least 0.247 of clearance at its worst orientation — so on easy,
rotation is not merely unused, it is *geometrically incapable of being necessary*. This is the
documented design intent (`arena.py`: "easy 1.8 (head-on, trivial)"), not an accident of one starting
pose. Simulated over the ten jittered pilot seeds, a rotation-free policy — close the y gap, then
push east — solves **10/10 within budget** (`scripts/check_rotation_need.py`). At medium (1.2) and
hard (1.1) the same policy solves **0/10**: those slits are below the extent at every angle and do
require the threading maneuver.

Against that ground truth:

1. **A's clean message is grounded and wrong.** A representative C0 delivered message (95 tokens
   mean) reports the true pose and then concludes: *"the load is oriented at an angle, it may not fit
   through the slit unless rotated … Rotate the load so it is aligned with the slit (i.e. vertical)
   before pushing east."* Every number in it is true — which is exactly why **G3 scores 0.999**.
   The inference drawn from those numbers is false.
2. **B complies.** C0's failed easy episodes are rotation-dominated; C0's three *successes* burn
   10–18 steps of an 18-step budget hunting for an angle (`ROT+,E,ROT+,ROT+,ROT+,E,E,ROT+,ROT-,…`).
   The per-action rotation quantum is coarse — measured mean 31.3°, modal 33.7°, damped to 11–19° on
   wall contact — so "align it vertically" is not a command this action set can execute cleanly.
3. **Truncating the message removes the instruction and leaves the coordinates.** C1 delivers 8
   tokens against a 103-token raw message. **All seven C1-easy successes are pure `E,E,E,…`,
   8–13 consecutive pushes — the A\* optimum.** 7/10 against C0's 3/10.
4. **Corrupting it (C4, 0.4 dropout, 95 → 56 tokens) leaves the numbers and the imperative
   fragments** — *"Push the load to moving toward goal"* survives the word-salad — and also gets 7/10.
5. **C3 is the control that identifies the direction.** C3 is the only condition that degrades what
   B **observes** rather than what A **says**, and it is the only degradation that *hurts*: easy
   success 1/10, the worst stuck rate in the sweep (0.466), the lowest mean progress (0.073) and the
   lowest `y_binary_progress` rate (0.275). The same nominal "degradation" carries **opposite signs
   depending on which channel it lands on.**

That dissociation is the finding. It is not explicable by noise-breaks-cycles alone (C1's truncation
is deterministic and helps as much as C4's stochastic dropout), and it is not explicable by
terseness (within C0-easy, message length does not predict success — the three successes average
99.7 tokens against the failures' 95.6, and the single longest episode succeeded).

#### Both difficulty cells fail task validity, independently of the channel

- **Easy cannot require rotation at all**, by the geometry above, and a rotation-free policy solves
  **10/10** jittered seeds within budget. The cell does not test coordination; it tests whether B can
  avoid being talked out of the obvious. The strongest single piece of evidence in the run is that
  the seeds C1-easy actually succeeded on — {1, 3, 5, 6, 7, 8, 9} — are **exactly** the seeds a pure
  push-east policy solves in simulation. Deprived of A's instruction, B converged on push-east and
  inherited its success set precisely.
- **Rotation is not necessary at *any* difficulty — the whole ladder, not just easy (DSE-057).**
  This entry originally said medium "requires rotation (0/10 rotation-free)" and proposed running it.
  That was inferred from a hand-written rotation-free policy failing, which proves nothing: a policy
  that fails shows only that *that* policy fails. Replacing the policy with the A\* oracle restricted
  to `N/S/E/W` and exhausting it gives the opposite answer. Over 10 jittered seeds × 3 difficulties,
  **0/30 seeds meet the necessity criterion** — 28 admit a translation-only path to the goal inside
  budget, and 2 have a full-action optimum containing no rotation at all. On the canonical pose the
  translation-only optimum is **13 steps at medium (equal to the full-action optimum: rotation buys
  nothing) and 14 at hard (against 13: rotation saves exactly one step)**.

  The mechanism is the arena's own documented geometry, now followed to its conclusion. The internal
  walls are `pymunk` segments of radius 0.05 — effectively planes with no depth — so the T never has
  to fit through the gap all at once. The bar crosses at its 0.3 thickness, then the stem crosses at
  its 1.0 length, each individually clearing a 1.1 slit. `sim/arena.py` states this outright ("the
  TIGHTEST threadable slit is the shorter member = the stem = 1.0"); what had not been drawn from it
  is that the staged crossing is a **translation** manoeuvre, so rotation is redundant everywhere.

  This subsumes and outranks the easy-only defect. DSE-006 calls rotation through the slits "the
  cognitive core of the task"; that core is absent from every rung of the shipped ladder, and no
  choice of slit width restores it, because the band is bounded below by the shorter member however
  the widths are set. It also explains the `ROT+,ROT-,ROT+,ROT-` oscillation that consumes up to
  30 of 33 steps in the hard cell: the pair is hunting an angle that never needed finding.
- **The pilot cell skipped medium**, which was proposed as the diagnostic rung. On the corrected
  criterion it is not a rung worth running: it fails necessity exactly as easy and hard do.
- **Hard is solved by nobody: 0/60 episodes across both attempts** (0/20 in attempt 1, 0/40 here),
  all four conditions, both prompt versions. The entire hard half of the sweep contributes no
  outcome variance, so G2's contrast rests on the easy cell alone.

Neither cell is currently a coordination test. That is a task-validity defect discovered by running
the instrument, and it is independent of — and prior to — the channel question.

#### The headline methodological finding (dissertation-facing)

*This subsection is written to be cited directly. It states the finding, its evidence, its scope and
its limits, and it deliberately rests on nothing that a reviewer can contest as a statistic.*

> **A coordination benchmark can appear to require a capability while its simulator admits
> degenerate, policy-independent solutions that never exercise it — and when it does, the
> information-theoretic read of the communication channel inverts. Restricted-action oracle checks
> and collision-fidelity checks are therefore necessary before task success can be treated as
> evidence about communication or coordination.**

**The evidence is a proof and an observation, not a test.**

1. **Exhaustive search (proof).** The task is built around threading a T-shaped load through narrow
   slits; DSE-006 calls that rotation "the cognitive core of the task". Running the same A\* oracle
   that certified the step budgets, but with the action set restricted to `N/S/E/W`, finds a
   translation-only path to the goal at **every shipped difficulty**: easy in 7 steps (its
   full-action optimum is itself rotation-free), medium in 13 (**equal** to the full-action optimum —
   rotation buys nothing) and hard in 15 (against 13 — rotation saves one step). Stable at 4, 16 and
   64 collision substeps. Rotation is never necessary; the cognitive core is absent from the whole
   ladder.
2. **Categorical observation.** Under the 8-token cap (C1) at hard, **10/10** failures are the
   literal action sequence `E,E,E,E,E,E` — the cap deletes A's rotate instruction and B falls back to
   pushing east. At easy the same deletion *raises* success to 7/10, and the seeds C1-easy succeeds
   on — {1, 3, 5, 6, 7, 8, 9} — are **exactly** the seeds a pure push-east policy solves in
   simulation. No test is involved in either statement.

**Why the inversion follows.** A's message is grounded (G3 = 0.999) and *inferentially wrong*: it
reports the true pose and concludes the load must be rotated before it can thread. B complies and
spends its budget hunting an orientation that the coarse rotation quantum cannot hit and that was
never required. Degrading the **message** (C1, C4) deletes the erroneous instruction and improves
outcomes; degrading the **observation** (C3) removes true state and worsens them (1/10, stuck rate
0.466). Same nominal degradation, opposite signs by channel — which is what identifies the mechanism
as instruction content rather than noise.

**What this finding does *not* claim.**

- It does **not** claim the agents cannot rotate, or that this model tier is incapable. The task did
  not ask for rotation, so the run contains no evidence either way.
- It does **not** rest on the handoff-level statistics. Every episode-clustered contrast in this
  entry crosses zero (see the correction above); the nominal *p* ≈ 10⁻⁷ was a clustering artefact.
- It is a statement about **this simulator at its recorded, versioned settings**, not about
  continuum physics. Feasibility verdicts here are properties of a discrete macro-action model with
  a specified integrator.

**Three distinct degeneracy mechanisms were found, and they generalise beyond this arena.** Each is
a way a task can admit a solution that bypasses its stated cognitive core, and each needs a different
check to detect:

| # | Mechanism | Why a clearance heuristic misses it | Detected by |
|---|---|---|---|
| 1 | **Staged crossing.** Thin walls constrain an instantaneous cross-section, not a swept volume, so a non-convex load crosses one member at a time. | The load's collision-free configuration space through the gap is not characterised by a single bounding y-extent: the bar and stem can occupy different longitudinal positions relative to the wall, admitting paths a whole-outline clearance calculation excludes. | Restricted-action oracle, exhausted |
| 2 | **Integration squeeze.** At coarse collision resolution a macro impulse can drive the load through an aperture narrower than its own outline before contact resolves. | It is not a geometric property at all — the verdict moves with the integrator. | Re-certification at higher `substeps`; verdict must be invariant |
| 3 | **Passive self-alignment.** Macro impulses are applied at the COM, but contact torque at the aperture mouth rotates the load anyway, so it aligns itself without any rotate action. | "Translation-only in the action space" is not "rotation-free in the state space". | Restricted-action oracle *plus* inspection of the realised angle trajectory |

Mechanism 2 is the one most likely to be reproduced elsewhere unnoticed: a candidate geometry was
solvable at the shipped `substeps = 4` and unsolvable at 8. **A feasibility verdict that moves with
the integrator cannot gate an experiment.** Every frozen E3 certificate was re-checked and is stable
from 4 to 64, so no result already recorded is affected — but the default is not a certification
standard, and `CERTIFICATION_STEP_CONFIG` (substeps = 64) now exists for that purpose.

**Reusable prescription.** Before treating task success as evidence about communication or
coordination: (a) run the oracle with the capability's actions removed and require exhaustion, not a
failing hand-written policy; (b) require the verdict to be invariant to collision resolution; and
(c) check the realised state trajectory, not just the action sequence, for the capability appearing
without being commanded.

**A fourth lesson, from getting (c) wrong twice.** The state-trajectory check was implemented, was
pre-registered as a criterion, passed on every seed — and could not have failed on any input. It sat
in the branch where the action-set check had *already* rejected the seed, so it could only refine a
failure's label; and it measured under a configuration that pins the quantity it measures to
identically zero, because the same task change that created the degeneracy also introduced a
structural guard against it. Both defects are invisible from the criterion's wording and both
survived a green test suite, because the test asserted that the check's constants existed rather
than that the check could fire. **A degeneracy check needs its own falsification test: construct an
input it must reject, and confirm it does.** Where the guard is structural, the honest report is not
a per-seed pass but the *counterfactual magnitude* — here, up to 103° of uncommanded rotation with
the guard removed, which is what makes the guard a declared modelling assumption rather than an
implementation detail.

#### What is *not* broken

Worth stating plainly, because the two failed gates make the run look worse than it is.

- **The CPVI estimator works.** C0 +0.153 bits, C4 +0.053, gap +0.100 in the predicted direction,
  control task −0.002, selectivity +0.128, over 1,925 handoffs. It orders the conditions correctly
  and does not fit nuisance structure.
- **The gradient is not an entropy artefact.** C4 carries the *highest* label entropy in the sweep
  (H(Y) = 0.983 bits at a 0.479 base rate, against C0's 0.872 at 0.308) and the *lowest* CPVI — more
  headroom, less of it used. The base-rate objection does not explain the gap.
- **The infrastructure is sound.** Job clean, 2,199 s, manifest complete, substrate labelled,
  dataset re-keyed by `PROMPT_VERSION` so attempt 1 could be neither resumed into nor overwritten.
- **The A\* certification is sound.** `_SEARCH_ACTIONS` includes ROT± and the oracle applies the
  real physics to a freshly placed load, so E0 certified against the same action set the agents use.
  The oracle is what exposed the easy cell's zero-rotation optimum.

#### What it changes

This fires the fallback ladder. Both rungs are taken, in parallel, because they do not compete for
the same resource — see the design-log entry of the same date for the reasoning and
PREREGISTRATION §6 for the declaration.

1. **Rung 1 — RQ3a is elevated and started now.** It needs no GPU (S6), so it costs the arena track
   nothing and it is the insurance that can carry the dissertation alone (DSE-041, DSE-042).
2. **Rung 2 — one declared task-geometry change, then one re-gate.** The acceptance criterion is
   checkable before any GPU time is spent: **the A\* optimum must contain ≥ 1 rotation and finish
   strictly inside budget, for every jittered seed, at every difficulty.** That makes easy a
   coordination test and hard reachable at the same time.
3. **RQ3b is explicitly deferred behind rung 2.** The gate blocks low-information handoffs and is
   calibrated against realised outcomes; on a task where degrading the message *improves* outcomes,
   that calibration inherits the inversion wholesale.

**Null result recorded as such.** If the rung-2 re-gate also fails, the headline is rung 3 — and the
methodological findings below stand on their own regardless of which rung carries the thesis.

#### Methodological findings, which do not depend on the gate ever passing

**1. V-usable information is sign-blind.** CPVI answers *"how much does the message reduce a
V-bounded predictor's uncertainty about Y?"* It does not answer *"does acting on the message improve
Y?"* This run separates them cleanly: the clean channel carries **+0.100 bits more usable
information** about the outcome label than the degraded one **and produces less than half the
success rate**. The message is genuinely more informative *and* genuinely harmful, and the two facts
are measured on the same 1,925 handoffs by the same estimator. The literature routinely treats
message informativeness as a proxy for message quality; this is a mechanistically-explained
counterexample with an internal control (C3) fixing the direction.

**2. That gives the pre-registered circularity guard empirical teeth.** PREREGISTRATION §6 already
bans calibrating the runtime gate against CPVI, on principle. This run shows what the ban buys: a
gate calibrated on CPVI would have scored C0's messages *highest* and **promoted precisely the
messages that caused the failures**. The outcome-only calibration rule and `CosineStatistic`'s
probe-independence are now motivated by data rather than by argument — which is a stronger position
than the pre-registration could claim on its own.

**3. Groundedness is not correctness, and G3 cannot tell them apart.** G3 scored **0.999** on a
message corpus whose modal inference was wrong. A check that verifies message numbers against true
state is *by construction* blind to a false conclusion drawn from true numbers. This is a construct
gap in a gate this project pre-registered and then ran — and it generalises: hallucination-style
faithfulness checks on inter-agent messages will pass confidently-wrong-but-faithful reasoning. Any
future G3 needs a correctness limb (e.g. agreement with the oracle's next action) alongside the
grounding limb.

**4. Instruction-following dominates self-observation in a two-agent loop.** v5 gave B the last four
actions and the geodesic each gained — including explicit evidence that rotating gained nothing —
and B kept rotating while A kept telling it to. Cycling fell (0.667 → 0.143 on C0-easy) and the
outcome did not. A receiver that can see its own failed history still weights the sender's
instruction above it. That is a measured statement, not a speculation, and it bears directly on any
gate that hopes to change behaviour by re-prompting.

**5. Negative results survive the instrument being wrong.** Findings 1–4 are established by the
run's *internal* contrasts — C3 against C1/C4, CPVI against outcome, oracle against observed plan —
so they do not depend on G1 or G2 ever passing, and they are not invalidated by the task-validity
defect that rung 2 exists to fix. That is the property that makes them safe to write up now.

---

### E3 — Pilot gate run, Myriad bf16 — **the rung-2 re-gate on the successor task; verdict `fallback`** — 2026-08-29

- **Run id / manifest:** `af50c7c12d65540f` · `runs/rq1/af50c7c12d65540f/manifest.json` · frozen
  reading `runs/rq1/af50c7c12d65540f/README.md` · analysis `runs/af50c7c12d65540f-report/` ·
  RQ2 `runs/af50c7c12d65540f-rq2/` · logs `runs/myriad/af50c7c12d65540f-run/`
- **Substrate:** `myriad-nvidia-a100-pcie-40gb` (job 238085) · **Model + revision:**
  `Qwen/Qwen3-14B`@`40c069824f4251a91eefaf281ebe4c544efd3e18` (bf16, vLLM 0.18.1, torch 2.10.0+cu128,
  guided JSON, `temperature=0`, `seed=0`)
- **Encoder + revision:** `BAAI/bge-base-en-v1.5`@`a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`
- **Code:** git `10283b0` — the commit PR #76 merged, so the run is at a real merged revision ·
  sweep `f2b7bc42a511a735`, matching the dry run declared before submission · prompt v9 ·
  simulation digest `92b0c63b141ab074` · load shape **bar** (the rung-2 successor task)
- **Cell:** C0/C1/C3/C4 × easy/hard × seeds 0–9 — the pre-registered E3 cell (§6), 80 episodes,
  3,419 handoffs, 47 min. `--max-steps 50`, the deviation logged in §6 and D29 before submission.

**Verdict: `fallback`.** G1 **PASS** 0.800 (easy C0 8/10). G2 **FAIL** −0.250 (CPVI gap −0.015).
G3 groundedness **PASS** 1.000. G3 correctness **FAIL** 0.242 against a null 95th percentile of
0.257, *p* = 0.980. Attempt 2, so a still-failing gate invokes the ladder. **The E3 ledger is now
closed: no attempt 3, and no second rung-2 attempt.**

**The pre-registered directional prediction came out half confirmed.** §6 fixed, before the first
successor model call, that degrading the message (C1, C4) should reduce success relative to C0.
C1 confirms it and harder than predicted — **0/20**, 44.3 wall collisions per episode. C4 inverts it
again and by more than the T-load attempt did — **14/20 against C0's 9/20**. The instruction account
is therefore not sufficient, and by the ladder's own terms **rung 3 stands as the finding**.

| cond | chars | success/20 | CPVI | CPVI 95 % CI | PVI − CPVI | steps | collisions |
|---|---|---|---|---|---|---|---|
| C0 | 373 | 0.450 | 0.222 | [0.165, 0.278] | 0.084 | 40.2 | 3.8 |
| C1 | 46 | 0.000 | −0.023 | [−0.157, 0.060] | −0.004 | 50.0 | 44.3 |
| C3 | 377 | 0.150 | 0.143 | [0.104, 0.186] | 0.119 | 46.3 | 2.1 |
| C4 | 221 | 0.700 | 0.237 | [0.200, 0.278] | 0.065 | 34.5 | 8.6 |

## Findings

**1. The length-matched control earns its place, and it reframes C4.** C1's and C3's deficits
survive stratification on delivered message length (−0.450 → −0.500; −0.300 → −0.277). C4's
*advantage does not*: within the single overlapping stratum the success delta is **−0.071** against
+0.250 unrestricted, and the CPVI delta −0.069 against +0.030. On the pre-declared confound control
**C4 is not better than C0, it is shorter than C0.** The instrument was built for exactly this
challenge and it fired on its first real cell. The caveat travels with the number: 13 of 40 episodes
in one bin is a thin stratum, and this is a sensitivity analysis, not the headline.

**2. Only the lossy channel produces a receiver that acts on the pose.** Run within each condition,
G3's correctness instrument separates them: C0 agreement 0.330 (*p* = 0.070), C1 0.046 (*p* = 0.095),
C3 0.261 (*p* = 1.000), **C4 0.399 (*p* = 0.010)** — the only condition beating its own
within-episode null, surviving Bonferroni across the four tests. Every supporting measure agrees:
C4 turns the right way 61.4 % of the time against C0's 52.8 %, oscillates least (flip rate 0.445 vs
0.559), rotates least (381 vs 436), and leaves least of its own signal unused (0.154 vs 0.268). **The clean, fully grounded, numerically exact message produces a
receiver indistinguishable from its own shuffled actions.** This is the mechanism for the absent
gradient: a channel cannot be degraded informatively when the receiver was never reading the state.

**3. The gate's own pooled reading hid finding 2.** `g3_correctness` pools the cell and returns
0.242 — a clean fail with no indication that one condition passes. The per-condition limb
(`rq1.action_agreement`, added with this run) is what makes the mechanism visible, and it is
deliberately a separate function: the gate produced a verdict of record and must not be re-shaped
after being read.

**4. The condition ordinal is not the severity ordinal, and G2 inherits that.** C1 delivers 46
characters against C4's 221; C1 is by far the harsher manipulation and C4 is the middle of the
range. G2 contrasts C0 against C4 because §6 declares the ladder as C0 → C4 and states the C0−C4
contrast in its own worked example, so the gate is read as declared and the verdict stands. Recorded
because a degradation axis that is not monotone in its own nominal severity is a design fact worth
stating: against C1 the same gate would read success +0.450 and CPVI +0.245. **That is a diagnostic,
never a re-read** — swapping the contrast after seeing which one passes is the forbidden move.

**5. RQ2's estimands pass on the corpus where RQ1's hypothesis fails.** The G1 confirmation was
C0-only, so H4 had no contrast and returned +0.052 [−0.100, 0.201]. On four conditions all three
runtime statistics separate from their shuffled nulls — `info` +0.275 [0.150, 0.384], `fail` −0.275
[−0.379, −0.155], `cosine` +0.269 [0.147, 0.371] — and `fail` separates realised failure at **AUROC
0.906**, up from 0.754. CPVI rank-orders success across all four conditions; H2's indirect effects
exclude zero for C1 (−0.197 [−0.557, −0.081], 43.7 % mediated) and C3 (−0.058 [−0.234, −0.001]); the
RD-15 audit reads *p* = 0.00498 against a null max of 0.0047; control-task CPVI is 0.0018. **The
negative result and the positive result are the same instrument read from two ends.**

**6. The seed sensitivity forbids a magnitude claim.** The per-seed condition gap has sd 0.354 and
spread 1.0 across ten seeds, with six seeds at exactly 0.0 and one at −1.0. The inversion is
reported as a direction with a named mechanism, never as a stable effect size.

**7. What is not settled, and by what.** That the *numbers specifically* are the un-actionable
content is not shown — C4 removes them randomly *and* shortens the message. `PREREGISTRATION.md`
§8b **A2** declares the arm that holds length and swaps which content survives, with its decision
rule fixed before the `qsub`; **A4** would complete the 2×2 and is blocked on a design decision, not
on compute. Whether the receiver's blindness is a *serialisation* artefact or a *capability* limit is
also open, and both are readable with the same agreement limb and no new code.
