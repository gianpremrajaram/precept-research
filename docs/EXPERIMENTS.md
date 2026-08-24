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
> **Last revised:** 24 August 2026.

---

## 1. Status board

The task is certified and the first real model calls have been made. E0 passed on 24 August 2026 and
S1 is under way on the local substrate; **no headline data exists** — every episode recorded so far is
interim `local-lmstudio` data, permanently labelled as such and never pooled with cluster data. E3 has
run once locally and returned `retune_once`; the one-retune ledger is **not** open, because a 4-bit local
verdict is indicative by pre-registration and the ledger starts at the Myriad bf16 re-gate.

| ID | Experiment | Stage | Needs | Status |
|---|---|---|---|---|
| E0 | Task certification (CPU only, no model) | S0 | Nothing | **Passed 24 Aug 2026** |
| E1 | Serving smoke test and transcript read | S1 | A served model | **Run 24 Aug 2026 (v2, then v3)** |
| E2 | Model-ladder benchmark (DSE-005) | S1 / S2 | A served model | **Local row 24 Aug 2026; cluster rows open** |
| E3 | Formal pilot gate run — G1/G2/G3 (DSE-019) | S1 → re-gate S2 | E1, the driver | **Run twice locally 24 Aug 2026: `retune_once` both times (G1 and G2's success half fail; G3 passes 0.977). CPVI gradient C0>C1>C3>C4. Indicative — bf16 re-gate is the verdict of record** |
| E4 | RQ1 information-gradient main sweep (DSE-020) | S3 | E3 verdict = proceed; the freeze | **Not run** |
| E5 | RQ1 robustness cells (DSE-021) | S3 | E4; per-role client refactor | **Not run** |
| E6 | RQ2 measurement primitive (DSE-022) | S4 | E4 episodes (no new compute) | **Not run** |
| E7 | Gate calibration (DSE-017 run) | S4 | E4 episodes | **Not run** |
| E8 | RQ3b causal gate + controls (DSE-025) | S5 | E7 threshold; DSE-018, DSE-045 | **Not run** |
| E9 | RQ3a corpus spike and counts (DSE-041) | S6, parallel | Nothing but network | **Run 24 Aug 2026.** Conditioning state confirmed (220 traces, 5,960 steps, 2,488 handoffs, recorded input contexts); **trace outcome falsified — 0 non-failures on the primary corpus**. Promotes DSE-042 to load-bearing |
| E10 | RQ3a replay labelling (DSE-042) | S6 | E9; a spend cap | **Not run — now the only route to a two-class step-level *Y*** (E9) |
| E11 | RQ3a localisation and baselines (DSE-024, rescoped) | S6 | E9, E10, E4 probes | **Not run** |

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

**Design.** C0, C1, **C3** and C4 crossed with easy and hard, at **three or more seeds**; the bf16
re-gate runs **seeds 0–4** (40 episodes), widened from 0–2 on 2026-08-24 because G1 — the one gate
bf16 can plausibly flip, and the one that has already failed — was resting on three easy-C0
episodes, where a true success rate of 0.67 fails a 0.5 threshold about a third of the time
(PREREGISTRATION §6, dated pre-freeze). C3 is in the cell because it is the only condition carrying a genuine observation asymmetry and it
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
`qsub -P <project> scripts/myriad/pilot.sh` (DSE-050). It serves, waits for the endpoint, warms the
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
| S3 | Control-task selectivity and repeated cross-fits (DSE-043, DSE-044) | M | Open — **blocks F0** |
| S3 | Per-role clients on the runner (`client_a`, optional `client_b`) (DSE-049) | S | **Built 24 Aug 2026** |
| S5 | Gate integration and controls (DSE-018) and the retry feedback template (DSE-045) | M | Open |
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
| F0 Pre-registration | *Y* and *k*, conditioning semantics, *V* and its selection rule, encoder + revision, serialisation, C1/C3/C4 parameters, jitter and seed count, budgets, analysis protocol, gate feedback template, G1/G2/G3 thresholds, *R* repeats, the control-task expectation, the length controls | E3 verdict = proceed | **Open** |
| F1 RQ1 | Headline gradient, mediation, selectivity, absent-versus-unused | E4 complete | **Open** |
| F2 RQ2 | Twin agreement, proxy tracking, encoder sensitivity, operating point | E6, E7 complete | **Open** |
| F3 RQ3b | Causal contrast against both controls | E8 complete | **Open** |
| F4 RQ3a | Localisation under both regimes, baselines table with dates | E11 complete | **Open** |

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
