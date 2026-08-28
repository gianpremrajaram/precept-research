# Experiment Design Log

A dated, reverse-chronological record of **experiment- and research-design changes** — decisions
that alter what the study measures or how the task/measurement is constructed, not routine code
edits. It is deliberately separate from `CHANGELOG.md`: the changelog tracks *code and behaviour* at
the component level; this log tracks *design and research-question* decisions (task geometry, outcome
and probe definitions, difficulty semantics, conditioning choices, gate design) with the evidence and
the risk each one closed. The two do not replace each other — a single change often appears in both,
the changelog saying *what the code now does* and this log saying *why the design changed and what it
de-risked*. Update this log whenever a change would affect the interpretation of a result or a
reviewer's read of the method.

Each entry: **date · area · trigger · finding · impact · risk reduced · correction path · the fix ·
result of the fix · so-what/takeaways.** Keep entries roughly one page.

---

## 2026-08-28 (latest) — H4 as written could be confirmed by a proxy that tracks nothing but the channel label

- **Area:** the RQ2 measurement analysis (§9.7) — what "the runtime proxy tracks CPVI" is allowed to
  mean, and what the four-label comparison is a decision about.
- **Status:** written and merged **before any of job 227886's output is read**. Declared as **D24**
  in `docs/methodology.md` §10.5. The ranking rule lives in `experiments/rq2.py`'s module constants,
  so it is fixed in code rather than in prose that could be edited once numbers land.

### Trigger

DSE-022 (opened 25 June 2026) asks RQ2 for "a documented recommendation: the headline Y (from the
four options) and the encoder". `PREREGISTRATION.md` §4 subsequently froze *Y* at
`y_binary_progress`, and the fallback ladder records in terms that a re-reading of the ticket cannot
soften: re-choosing *Y* "would rescue the gate without touching the defect, which is the forbidden
move". The ticket and the register disagree, and the disagreement had to be settled before the
module was written rather than discovered at review.

### Finding

**1. The Y conflict is real, and resolves by re-scoping rather than by picking a winner.** The
ticket predates the freeze. What it actually needs a decision about — which label a *gate* should
threshold — is not the thing §4 froze, and no gate run exists to have contaminated it. So the
four-label comparison ships in full, reported as a pre-declared robustness check on the frozen label
**plus** the selection input for the RQ3b gate target. RQ1 is untouched. The same narrowing applies
to the encoder: §5 pins the primary and names `all-mpnet-base-v2` a *sensitivity* encoder, not a
candidate, so the implemented rule can only flag a re-freeze and never perform one.

**2. The larger finding: H4 as specified is not falsifiable, given D23.** The acceptance criterion is
"rank correlation of each statistic with CPVI". D23 established that **78.0% / 96.4% / 92.1%** of the
reported cross-condition CPVI contrast on the three existing datasets is the *identity component* —
what survives destroying the message, because condition-level message style plus differing
per-condition progress base rates make the condition tag alone predictive. A runtime statistic that
picks up nothing but that tag will therefore post a healthy ρ against CPVI and satisfy the criterion
exactly as written, while tracking no message content whatsoever. Nothing in the AC separates the
two. `InfoStatistic` and `FailStatistic` are probe-backed on the same embeddings the CPVI probes see,
so this is not a hypothetical failure mode for them — it is the expected one.

**3. RQ1 applies the RD-15 null to the pooled mean only, and its null is not reusable here.**
`shuffled_message_cpvi` returns one mean per permutation. A correlation needs the null paired back to
the handoff it belongs to, which per-permutation means cannot supply.

### Impact

Unaddressed, H4 could have been written up as confirmed — "the cheap runtime statistic tracks the
expensive offline one" — on a dataset where no statistic tracks any message content at all. That
claim is what makes the gate defensible at all: a gate thresholding a statistic that only reads the
channel label is a gate that has learned which arm it is in.

### Risk reduced

The measurement-primitive claim becomes falsifiable. Every H4 number now has a null it must beat,
and the null is one the same run produces rather than an argument made afterwards.

### Correction path considered and rejected

**Partialling condition out of the correlation** (partial Spearman against a condition indicator, the
machinery `analysis/stats.py` already has for message length). Rejected: condition is not a nuisance
covariate here, it is *the manipulation*. Residualising on it removes genuine channel-driven message
content along with the tag, and the residual answers a question nobody asked. The within-condition
permutation removes exactly and only what survives destroying the message, which is the identity
component and nothing else — and it leaves each handoff its own state, its own Y and its own
condition, so the comparison stays like for like.

### The fix

- `_null_cpvi` rebuilds the within-condition permutation to return **per-handoff** null scores,
  averaged over `n_shuffle` permutations. Cross-fit repeats are dropped inside it: averaging over
  permutations already damps the fold noise `n_repeats` exists to damp.
- Every ρ and every mean CPVI is reported three ways — real, null, corrected — with the corrected
  interval from a **paired** episode-cluster bootstrap (`_cluster_resamples`: episodes resampled with
  replacement, both quantities recomputed inside each draw). This is the same estimator the D23
  re-scoring used.
- `DECLARED_ORIENTATION` fixes each statistic's expected sign in advance and both AUROCs are taken
  under it. A value below 0.5 is reported as-is. The gate's calibration derives an orientation from
  the failure label because a gate must act; an analysis that chose the sign making its own AUROC
  exceed 0.5 would be fitting the direction to the data.
- The label ranking is lexicographic with **corrected effect size last**: encoder-invariance of the
  per-handoff CPVI ordering, then the label's own twin agreement, then effect size only as a
  tie-break inside 0.05. The two criteria ahead of it both ask whether the measurement is *the same
  measurement* under a different encoder and without the realised outcome — which is what a gate
  target has to be. `recommended_y = None` is a reportable outcome, not a fallback.

### Result of the fix

The known-answer case is exact rather than approximate. When messages are constant within a
condition, the permutation is a literal no-op, so the corrected mean CPVI, the corrected ρ and both
their intervals come out at `0.0` to machine precision and **no label is admissible** — the fixture
is in `tests/unit/experiments/test_rq2.py` and is the case the module exists to get right. A second
test pins `_null_cpvi`'s mean against `shuffled_message_cpvi`'s, so RQ1's null and RQ2's cannot
drift apart. The module is at 100% line coverage; the suite is 472 passed, 2 skipped.

### So-what / takeaways

1. **A null applied to the pooled mean does not transfer to a contrast or a correlation.** D23 caught
   the first case, this entry catches the second. Anywhere CPVI enters a *comparison*, the null has to
   enter the same comparison, or the identity component rides along unlabelled.
2. **An acceptance criterion can be satisfiable by the failure it was written to detect.** "The proxy
   correlates with CPVI" was a reasonable criterion in June and became an unfalsifiable one the day
   the identity component was measured. Tickets are not self-updating; the register is what catches
   this, and only if the conflict is raised rather than silently reconciled.
3. **A decision rule fixed in code before the data is a different object from one written in prose.**
   The ranking constants are executable, tested, and in the diff that predates the run — which is the
   only form of "pre-declared" that survives someone asking whether it was edited afterwards.

---

## 2026-08-28 — The cross-condition CPVI gap is mostly a condition tag, and what the next run has to establish

- **Area:** what the shuffled-message audit is applied to (§8.5), the reported estimand for every
  cross-condition CPVI contrast, and the cell the post-gate characterisation run covers (§9.10).
- **Status:** post-verdict, pre-run. Declared as **D23** in `docs/methodology.md` §10.5 before the
  job is submitted. Parts are confirmatory rather than blind, and are marked as such throughout —
  this is the first register entry written with episodes on disk.

### Trigger

Job 227048 — the DSE-057/058 successor re-gate, and the second and final permitted E3 attempt —
returned `fallback`: G1 0.200 against 0.500, G2's success half −0.050 against +0.100, G3 0.997.
The one limb that *passed* was G2's CPVI half, reading a `+0.146`-bit C0−C4 gap with the control task
at −0.004 and selectivity +0.088. An audit was asked to decide whether the verdict was a valid
scientific result, a reporting artefact, or a configuration fault. The recovered `manifest.json`
settled configuration immediately: `git_sha 1580259`, `load_shape: bar` at 1.4 × 0.3, slits
1.2/0.8/0.5, `wall_depth 1.5`, prompt surface v6, `--attempt 2`, served bf16 on an A100-40GB. The
run is exactly what it claims to be. That left the reporting question, and it is the whole entry.

### Finding

**1. The apparent progress signal was a per-step statistic read as a cumulative one.** `progress` in
the parquet is `step_progress(pre, post)` — one macro-action's signed geodesic gain, not distance
covered. Against an initial geodesic of ~8.5 units, the largest single shove being 0.906 is 11% of
the journey, not 90% of it. Recomputed as fraction of geodesic closed: easy 0.273 / 0.229 / 0.190 /
0.602 and hard 0.036 / 0.151 / 0.036 / 0.128 across C0/C1/C3/C4. **39 of 40 hard episodes never
leave the starting chamber of three.** The cell is at floor, not on a knife-edge.

**2. The identity component dominates the reported contrast, in every dataset the project has.**
`shuffled_message_cpvi` permutes messages within condition, and §8 already names the resulting null
height as CPVI's *identity* component — condition-level message style plus differing per-condition
progress base rates. That correction had only ever been taken on the **pooled mean**. Applied to the
**gap**, over 250 episode-cluster bootstrap resamples with real and permuted scored in the same
resample:

| dataset | task | eps | reported C0−C4 | shuffled | corrected | identity share |
|---|---|---|---|---|---|---|
| `1c994b87bbca8257` | T | 40 | +0.250 [+0.079, +0.476] | +0.195 | **+0.055** [−0.018, +0.158] | 78.0% |
| `eddd19c654515bb2` | T | 80 | +0.102 [−0.004, +0.211] | +0.099 | **+0.004** [−0.057, +0.055] | 96.4% |
| `a66efd4ea089af6a` | bar | 80 | +0.149 [+0.056, +0.272] | +0.137 | **+0.012** [−0.035, +0.073] | 92.1% |

Two geometries, three runs, two seed counts. `1c994b87bbca8257` is the `+0.243`-bit "healthy
gradient" cited on 2026-08-26 as evidence the instrument discriminated: 78% of it was the tag.

**3. Within difficulty, the cleanest-looking gradient is in the cell where nothing happened.** On
`a66efd4ea089af6a`, hard reports +0.1715 [+0.072, +0.284] — an interval excluding zero — and corrects
to **−0.004** [−0.057, +0.048]. That is the cell whose episodes never left chamber 1, so no mechanism
exists by which real message content could differ. Easy reports +0.140 and corrects to **+0.035**
[−0.037, +0.133], P(>0) = 0.73: unresolved, and the only place a real effect could live.

**4. The raw gap is unstable while the corrected one is not.** Raw swings 2.5× across runs
(+0.102 → +0.250); corrected sits within 0.05 bits of zero in all three. An identity component tracks
whatever the per-condition base rates happened to be that run, and those swing hard — C1's
`y_binary_progress` is 0.81 / 0.93 / 0.70. **C1 is the top-CPVI condition in all three runs**
(+0.263 / +0.254 / +0.157): the 8-token cap reads as the *most* informative channel every time, which
is the artefact's signature and not a gradient.

**5. The pooled-mean excess is real, stable, and the strongest result the project holds.**
Leakage-corrected mean CPVI: **+0.058** [−0.001, +0.140], **+0.048** [+0.025, +0.077], **+0.039**
[+0.009, +0.073]. Roughly 0.04–0.06 bits of genuine per-handoff message content, replicated across
two task geometries and independent of whether the arena is solvable.

**6. n binds the measurement claim, not the capability claim.** The corrected gap's episode-cluster
SE is 0.027 at 80 episodes and projects to 0.013 at 320 — the difference between "unresolved" and a
statement either way. Terminal success is the opposite: hard is 0/40, and bounding the easy contrast
inside ±0.10 needs ~230–305 episodes per arm.

**7. Aside, for the successor-task record.** The bar task produced **fewer** successes than the
T-task it replaced (5/80 against 18/80). It closed the memorylessness defect and cost capability.

### Impact (had it not been caught)

RQ1's headline would have been an information gradient that is 78–96% condition identity, published
with an interval that excludes zero in precisely the cell where the agents were provably inert. The
G2 CPVI limb would have been cited as the surviving evidence that the measurement instrument works —
it is the one limb that passed — and the claim would have rested entirely on the uncorrected number.

### Risk reduced

The forbidden move here is not threshold-shopping; it is reporting a contrast the design cannot
identify. Fixing the estimand *before* the next run, and declaring which half of the fix is
confirmatory, removes the option of discovering the correction after seeing whether the new cell
cooperated.

### Correction path considered and rejected

Re-pointing G2's outcome limb from `y_terminal_success` at `y_binary_progress`, on the argument that
§6 already pre-registers the latter. Rejected twice over: the progress table that motivated it was
the per-step misread of Finding 1, and C1 — the condition driving the apparent progress advantage —
has a 0.905/0.960 collision rate with **0 of 20 episodes** surviving a ≤0.25 contact filter, so the
advantage is ramming. §6 registers terminal success for the *outcome* claim and `y_binary_progress`
for the *CPVI construct*; collapsing the two is a new degree of freedom, not an alignment.

### The fix

1. **The corrected gap becomes the reported estimand** for every cross-condition CPVI contrast, raw
   gap and null shown beside it. Extending `_shuffle_audit` from the pooled mean to the contrast is
   filed as a follow-up; until it lands the correction is computed from the persisted per-handoff
   scores and the parquet.
2. **`medium` (slit 0.80) and `C2` enter the grid** — the two cells with zero recorded episodes, and
   the only genuinely blind parts of D23. C2 is load-bearing rather than decorative: its delivered
   message is surface-identical to C0's, so it is the one arm that can separate a channel effect from
   an identity component instead of arguing about it.
3. **The run is driven by `preceptx-rq1`, never `preceptx-pilot`.** There is no attempt 3; a driver
   that emits a G1/G2/G3 verdict cannot be run without spending one that does not exist. The changed
   grid yields a different `dataset_hash`, so 227048's verdict stands unamended. `scripts/myriad/`
   `pilot.sh` gained a `DRIVER` selector and grid pass-through for exactly this.
4. **No threshold moves.** §5, §6 and §9.10 are untouched by this entry, and the pilot is reported as
   an estimate with intervals plus an explicit statement that G1/G2's thresholds were set
   uncalibrated, rather than as a gate whose verdict carries evidential weight it never had.

### Result of the fix

**Not yet known — the run has not been submitted.** Recorded here so that the estimand, the cells and
the driver were fixed before any of it returned. What is already established is the retrospective
half: the identity share and the corrected gaps in the table above, and the replicated pooled-mean
excess.

### So what

The pilot's one surviving limb was the one that most needed auditing, and the audit that would have
caught it was already in the protocol — pointed at the wrong quantity. A negative control certifies
only the statistic it is actually applied to; "the control task is at zero and selectivity is
positive" says nothing about a contrast the control never touches. The generalisation is not specific
to this arena: **any study that varies a communication channel and measures information at the
boundary inherits this artefact**, because the manipulation and the message surface are the same
edit. That is an RQ2 methods contribution, it rests on data already recorded, and it survives whether
or not the arena ever becomes solvable.

---

## 2026-08-27 — What the RQ3a table can honestly contain, and one blind secondary analysis

- **Area:** the RQ3a localisation comparison — which methods enter it, what an empty cell means, and
  what the MAST arm is allowed to claim (DSE-024); plus a pre-registered RQ1 secondary analysis
  declared before the rung-2 verdict (DSE-046).
- **Status:** pre-freeze, and deliberately pre-verdict. Job 227048 (the successor re-gate) is queued;
  neither piece here reads its result, which is the point of writing both now.

### Trigger

DSE-042 merged, so the RQ3a chain had a loader and an outcome construction but no analysis. Writing
the analysis forced three questions the tickets state as if they were settled: which corpus supports
which regime, what the published Who&When baselines mean when they cannot be run against the
annotator that produced them, and what a "human-agreement audit" is when no human is annotating.

### Finding

**1. Neither CPVI regime can run to completion today, for two different reasons.** The transfer
regime needs a frozen simulator-trained statistic; the successor task was certified on 2026-08-27
and has not yet made a model call, so no such probe exists. The refit regime needs the DSE-042
replay labels, which are a budgeted run that has not happened. Reporting either as a number today
would require inventing an input.

**2. MAST cannot carry CPVI at all — this is structural, not a sample-size problem.** Its traces are
published as one unsegmented transcript, so there is no observation/message split and therefore no
conditioning state. CPVI is *undefined* there rather than small. The earlier reading that MAST is
"the one corpus where refit CPVI runs at zero replay spend" conflates *a genuine two-class outcome*
(which MAST has) with *a per-step conditioning state* (which it does not).

**3. The published Who&When baselines were produced by a hosted frontier annotator.** Every model
call in this project is local or on the Myriad allocation. The procedures can be re-implemented, the
published numbers cannot be reproduced, and quoting the two side by side as though one beat the
other would be a comparison across annotators dressed as a comparison across methods.

**4. The ticket's "small human-agreement audit" has no human in it.** Both label series available —
the judge's selection and the corpus annotation — are already-existing artefacts. Cohen's kappa
between them is a judge-validity check; calling it a human-agreement audit would overstate a number
that is defensible under its true name.

### Impact (had it not been caught)

A results table with `0.0` where the transfer probe does not yet exist, a MAST row implying a CPVI
that cannot be defined, three baseline numbers read as the published ones, and a kappa presented as
newly collected human agreement. Every one of those survives casual review and none survives a
reviewer who checks the substrate.

### Risk reduced

- **Empty cells cannot masquerade as measurements.** Every method carries `status` ∈ {`ok`,
  `unavailable`, `not_applicable`} with a `reason`, and an unavailable method **keeps its row**. A
  dropped row reads as "never considered"; a zero reads as "measured and failed".
- **The transfer regime refuses to guess its sign.** Risk = `orientation × raw`, where orientation
  comes from the calibration that was fitted against realised outcomes. Absent it, the regime is
  `unavailable` rather than defaulted — a wrong sign inverts every localisation number while looking
  entirely plausible.
- **Annotations stay off the scoring path structurally.** `LocalisationStep` has no `annotations`
  field, mirroring `ReplayStep` and `measure.twin`. The annotation enters only through
  `trace_targets`, on the evaluator's side.
- **Judge failures abstain rather than fall back.** A `None` from the backend produces an all-zero,
  unrankable trace and increments an abstention count. No path lets a failed judge borrow the
  annotation it is being scored against.
- **The tie policy is fixed and recorded.** Average ranks over descending risk, top-*k* inclusive:
  a method that scores every step identically cannot win by input order.

### Correction path

MAST is reported as its own arm with its own metric — the cross-fit information in bits that the
trace text carries about the inter-agent-misalignment family — with `cpvi_status = not_applicable`
and the reason carried in the result object, so the limitation travels with the number instead of
living in a caption. The judge is declared an open-weight re-implementation in `JudgeIdentity`, which
reaches both the result JSON and the run manifest. The audit is named
`judge-selected agent vs the corpus's existing annotation`.

### The fix

`experiments/rq3a.py`: annotation-blind scoring view, one `MethodScores` type for every method, both
regimes with explicit statuses, two probe-free baselines, the three Who&When procedures behind a
three-question `JudgeBackend`, per-trace metrics with a trace bootstrap, the MAST arm, the agreement
audit, and a manifest block carrying the judge substitution and the tie policy. 25 unit tests plus an
integration test chaining corpus JSON → loader → replay labels → comparison table.

**And, on the RQ1 side, one analysis written blind (DSE-046).** Per condition, handoffs split at that
condition's own median CPVI crossed with realised progress: the **absent-signal rate** (low CPVI, no
progress — the sender did not encode) and the **unused-signal rate** (high CPVI, no progress — the
receiver did not act), which sum to the condition's no-progress rate. The within-condition split is
what keeps it from restating the condition effect. Declared in PREREGISTRATION §8 and
`ANALYSIS_PROTOCOL` **before** the successor re-gate returns, and declared as *secondary and
descriptive*: at this sweep's episodes per condition the intervals are wide, so it is a mechanism
description, never a significance claim and never a rescue of a null on the primary gradient.

### Result of the fix

The RQ3a table is runnable end to end today and every cell it cannot fill says why. The two inputs
it is waiting on — a frozen simulator statistic and a budgeted replay run — are named in the result
rather than assumed. The RQ1 secondary analysis is fixed in writing while its data is still unseen.

### So what / takeaways

- **"Not applicable" and "unavailable" are different claims and the table must distinguish them.**
  One says the corpus cannot support the method; the other says the input does not exist yet. Only
  the second becomes a number later.
- **A metric that cannot be computed on a substrate is a finding about the substrate.** MAST's
  missing observation/message split is worth a row in the design log, not a silent omission.
- **Re-implementing a baseline under a different annotator produces a different estimator.** Say so
  in the artefact, not in the prose around it.
- **The cheapest defence against researcher degrees of freedom is writing the analysis while the
  data is still in a queue.** Both pieces here were written with job 227048 unfinished, on purpose.

## 2026-08-27 (later) — The RQ3a arm had no outcome variable, and replay is the only one available

- **Area:** the outcome *Y* for the RQ3a (real-log) arm; the budget discipline for any replay run
  (DSE-042).
- **Status:** pre-freeze. No replay has been executed; this entry records the design and the guard,
  not a result.

### Trigger

The DSE-041 counts spike, read back against what CPVI actually needs. The loaders were finished and
the corpora counted; the question left over was which per-step outcome the estimator would condition
on, and the counts answer it in the negative.

### Finding

**Two of the three corpora are single-class at trace level, so the cheap outcome is degenerate.**
TraceElephant is **220 traces, 220 failures**; Who&When is **184, all 184 failures**. A constant
cannot be predicted, so Y1 (trace success) carries no information on either per-step substrate.
MAST is genuinely two-class (**405 non-failures of 1,642**) but is trace-level only and confounded
with system identity, so it cannot supply a *per-step* target at all.

The obvious substitute is the one that is forbidden. Y2 — using the corpus's own `mistake_agent` /
`mistake_step` annotation as the outcome — makes the localisation claim circular: CPVI would be
scored against the very labels it is meant to be an independent signal about.

**So the RQ3a arm, which the fired fallback ladder has already elevated to the headline, had no
usable outcome variable.** That is a stronger statement than "replay would be nice": rung 1 is the
track that can carry the dissertation alone, and without a two-class per-step target it cannot.

### Impact

Counterfactual replay moves from an upgrade path to a **precondition**. It defines the outcome
interventionally — re-run the system from step *t* with that step's output substituted, and record
whether the outcome changes — which is the only construction that yields a within-trace two-class
target on a corpus that also records the per-step conditioning state. It also has the side benefit
of putting the external-validity and causal claims on **one epistemology instead of two**.

### Risk reduced

Three, in descending severity:

1. **Circularity.** The labeller consumes `ReplayStep`, a four-field view on which `annotations`
   does not exist as a field. Not "unused" — absent from the type. A future edit cannot reach the
   annotations by accident, which is the same structural discipline that stops
   `prospective_twin` reaching Y.
2. **An accidental full-corpus replay.** The cap is enforced immediately before every backend
   invocation, not applied to the forecast. A projection is a prediction about a loop; the loop is
   what spends the allocation. A dry run cannot issue a call at all, because `project` takes no
   backend argument.
3. **A sample that quietly reweights the corpus.** Stratification is over the trace outcome with
   `None` as a stratum of its own — it is 176 of TraceElephant's 220 traces, so folding it into
   either class would shift the failure balance the sampling exists to preserve.

### The fix

`experiments/rq3a_replay.py`, plus the discipline around it:

- **Budget in calls and seconds, never currency.** Every model call in this project is local or on
  the Myriad allocation. A dollar figure would be an invented number that reads as measured once it
  reaches a table; GPU seconds appear only as an advisory estimate carrying its calibration source,
  and are `None` when no throughput calibration exists.
- **Refuse on the projected minimum.** If even the cheapest possible execution exceeds the cap, the
  run is refused before anything is sent — rather than started and killed part-way, leaving a
  half-labelled corpus whose base rate has to be explained.
- **Disagreement is reported, not resolved.** Replay is non-deterministic because the systems are
  LLM-driven, so determinism here means what it means for the simulator: low-variance and
  agreement-reported, never bit-exact. Steps below the agreement floor are **flagged and kept**. A
  dropped step changes every count downstream, and the disagreement is itself the finding that the
  step's outcome is not well defined.
- **The cheap label is computed anyway.** `trace_success_labels` runs for every trace regardless of
  budget, so the refit arm has a fallback if replay is cut. Degenerate on the per-step corpora — that
  is the whole reason this entry exists — but a present fallback beats a missing one, and MAST is
  two-class.

### Result of the fix

Code deliverable only; running replay at corpus scale is a budgeted experiment and is deliberately
not part of this change. The path from corpus JSON to labelled steps to a manifest block is
exercisable offline against a stubbed backend, with 100% line coverage on the module and no network
in any test.

### So what

- **A loader that parses is not a substrate that works.** DSE-041 was complete against every
  acceptance criterion and the arm still had no outcome. The counts, not the code, were the
  deliverable — and the counts said the obvious *Y* was unavailable.
- **A spend cap on a forecast is not a spend cap.** The forecast is a statement about the loop you
  think you wrote. The guard has to sit where the money is actually spent, which is the line before
  the call.
- **"Not applicable" is a design answer.** Currency has no place in a budget for local and allocated
  compute; naming the budget in calls and seconds is more precise *and* more honest than converting
  to a price nobody pays.

---

## 2026-08-27 (post-review) — A pre-registered acceptance criterion that no input could fail, and a prompt describing the wrong object

- **Area:** the certification standard's limb 7 (passive self-alignment), the status of
  `hold_orientation` as a modelling assumption, and the successor task's prompt surface (DSE-058).
- **Status:** pre-freeze, pre-compute. Still no model call against the successor task.

### Trigger

An independent multi-agent code review of the DSE-058 branch. Two of its findings were symptoms of
one thing: the successor task was implemented in the physics and the solver, and *not* propagated to
the two surfaces a reviewer reads the method through — the acceptance check and the prompt.

### Finding

**Limb 7 was vacuous in two independent ways, and it is a pre-registered criterion.** It was
declared as one of seven conditions "every declared seed at every difficulty must satisfy", and the
adoption record claims 30/30 seeds passed every limb. Neither half of it could ever fail:

1. **It ran in the wrong branch.** The drift check sat inside the branch where limb 3 had *already*
   found a translation-only path — that is, where the seed was rejected regardless. It could refine
   a failure's label; it could not cause one. The case it was written for — a seed that passes
   limbs 1–3 and is nonetheless degenerate — never reached it.
2. **It measured the wrong world.** It replayed under the shipped `StepConfig`, in which
   `hold_orientation` restores the pre-action angle after every non-rotate action. The measured
   drift was therefore **identically 0.0** on every input it ever saw, on every candidate geometry
   ever tested. The number in the certification record was an artefact of the guard, not evidence
   about the geometry.

**`hold_orientation` is doing far more work than "closes a degeneracy by construction" conveys.**
With the hold disabled, ten straight eastward pushes rotate the load by up to **103° (easy), 98°
(medium), 20° (hard)** across 30 seeds — enough to align the bar with the aperture completely,
with no rotate action issued. **All three rungs exceed limb 7's own 15° limit without the guard**,
so it is load-bearing everywhere, and the ordering is the inverse of the naive one — drift is
largest at the *widest* aperture, which admits the bar and lets contact turn it, while the hard
aperture is too narrow to enter and the load jams instead. The first estimate of this quantity,
taken on seeds 0–2, was **9°**.

**Every prompt surface still described a T.** Not only the `nl` serialiser: both system prompts said
"T-shaped load" and "the T-load", so *all three* serialisation arms told the model it was
manipulating a T while the physics carried a convex bar. Separately, the numeric and NL forms named
one x per wall while each wall spans 1.5 units of it, and the grid drew a one-cell stripe through
1.5 world-units of solid geometry.

### Impact

Had this shipped: the adoption record would have claimed a seven-limb certification that was a
six-limb one; the dependence of the headline rotation-necessity result on a modelling assumption
would have been undisclosed and unquantified; and the successor task's first model call would have
carried the *exact* defect DSE-057 was spent falsifying — a prompt grounded in correct numbers and
wrong about the object — for a second time, on a task adopted specifically to escape it.

### Risk reduced

The dissertation's rung-2 claim is "rotation is operationally necessary on the successor task". The
honest form of that claim is "…given that two grips hold the load's orientation through a push".
The 103° figure is what a reviewer would have asked for and what the record could not have supplied.

### Correction path (and what was rejected)

- **Rejected: delete limb 7 as redundant.** Defensible — the property *is* guaranteed by
  construction — but it would quietly drop a pre-registered criterion and leave the assumption
  undisclosed. The disclosure is the valuable part.
- **Rejected: certify with `hold_orientation` off.** That would certify a world the episodes do not
  run in. Limbs 1–3 must reflect shipped physics or they gate nothing.
- **Adopted: gate on the certifying config, report the counterfactual.** Limb 7 now runs on seeds
  that *pass* limbs 1–3, under the config being certified. `unheld_drift_deg` prints the guard-off
  drift alongside the verdict, so the assumption travels with the certificate rather than being
  discoverable only by reading the source.
- **Prompt surface bumped to v6.** This does **not** consume a fourth retune of the T task: it is
  the prompt surface of a different benchmark, and the arithmetic of "which retune is this" applies
  to the T arena, which is retired. It landed before the first successor model call, so no dataset
  is re-keyed.

### Result of the fix

Certification re-run at `--certify --seeds 30`: **30/30 at every rung**, unchanged, now with a limb
7 that can fail. Regression tests pin that limb 7 can reject an otherwise-clean seed, and that the
drift instrument reads 0 under the hold and non-zero without it — the test that would have caught
this. Full suite 359 passed, 1 skipped.

### So-what

**A pre-registered check that cannot fail is worse than no check.** It converts an untested
assumption into a documented result, and the documentation is what a reader trusts. The failure was
not a wrong threshold; it was a check wired where nothing could reach it, measuring a quantity a
later change had pinned to zero. Neither is visible from the criterion's wording, and both survived
a green test suite, because the test asserted the constants existed rather than that the check
could fire.

**Small samples lied a third time on this task.** 10 seeds hid the aperture-0.48 leak; 3 seeds put
the guard-off drift at 9° when it is 103°. The 30-seed rule was adopted for the first and would have
prevented the third had it been applied to a diagnostic rather than only to the gate.

**A task change is not done when the physics is done.** The bar shipped in the simulator, the
solver, the fingerprint and the difficulty ladder, and stayed a T in the two places that decide what
a model is told and what a reviewer is shown. Those surfaces need to be on the change's checklist
explicitly, because nothing in the type system connects them to the load's geometry.

---

## 2026-08-27 (later still) — The successor task ships, and two certification traps nearly let a broken one through

- **Area:** the successor task's implementation (load shape, wall topology, difficulty ladder, start
  distribution) and, more importantly, two failure modes of the acceptance procedure itself (DSE-058).
- **Status:** pre-freeze, pre-compute. Implemented and certified on CPU; no model call has been made
  against it.

### Trigger

The entry below selected a convex bar in a finite-depth channel and reported apertures 0.55-0.80 as
passing certification. Implementing it for real - on the actual jittered seeds, at the actual frozen
budgets - broke that result twice over.

### Finding

**Trap 1: the certification was budget-dependent, and the budget moved underneath it.** A ladder that
certified 10/10 at step budget 25 leaked as soon as the budget was re-derived to 28. A longer budget
admits longer degenerate paths, so the restricted search finds one where it previously exhausted.
Budget width is part of the acceptance criterion, not a parameter set afterwards; the check must be
re-run after any budget change.

**Trap 2: certifying on the declared ten seeds passed a task that leaks on a quarter of instances.**
Aperture 0.48 gave a clean **10/10 on seeds 0-9** - and **12/20 on seeds 10-29**, for 22/30 overall.
The pilot's own seed set would have certified a manipulation that fails ~27% of the time. Widening
the sample is what caught it, and no amount of care on the declared seeds would have.

**The mechanism behind both is passive self-alignment, and it is chaotic.** Contact at the channel
mouth rotates the load with no rotate action issued, up to 114 deg of it. Its incidence is not a
smooth function of aperture (0.44 gave 6/10 where 0.46 gave 10/10) nor of the y-jitter width, so no
choice of widths tunes it away - any 10/10 found by search is a draw, not a property.

### Impact

The earlier 0.55-0.80 recommendation was wrong, and wrong in the direction that matters: it would
have shipped. Both traps are procedural rather than geometric, so both generalise - any benchmark
certified on its own evaluation seeds, at a budget chosen after the fact, inherits them.

### Correction path (and what was rejected)

- **Rejected - pick the aperture that gives 10/10.** That is fitting the geometry to the seed set,
  and the 30-seed check shows it does not generalise.
- **Rejected - narrow the y-jitter** so off-centreline starts stop being funnelled. Tested at
  (2.4, 3.6) and (2.7, 3.3): results stayed chaotic and no combination reached 10/10 cleanly.
- **Rejected - tune wall friction**, the obvious lever for a contact effect. Aperture 0.45 held
  10/10 at friction 0.2, 0.6 **and** 1.5, so the mechanism is not friction-driven and tuning it
  would have been a constant changed for no reason. **No friction value was altered.**
- **Taken - hold the load's orientation through non-rotate actions** (`StepConfig.hold_orientation`,
  default on). Two grips carrying a rigid load hold its orientation as well as its position; the
  angle changes only when the pair deliberately rotates it. It is the same class of abstraction as
  the existing quasi-static velocity zeroing, and it matches the cooperative-transport referent the
  roadmap names.

### The fix

`hold_orientation` restores the pre-action angle after any action other than ROT+/ROT-. It closes
the degeneracy **by construction rather than by tuning**: every aperture from 0.45 to 1.10 certifies
**30/30**, where before the whole band leaked chaotically.

That reopened the design space the leak had collapsed, so the ladder could spread properly:

| Rung | Aperture | Optimum | Rotations | Budget |
|---|---|---|---|---|
| easy | 1.20 | 8 steps | **1** | 20 |
| medium | 0.80 | 10 steps | 2 | 25 |
| hard | 0.50 | 10 steps | 2 | 25 |

Difficulty now separates in the certificate itself - easy needs one rotation, medium and hard two -
and again in tolerance, since the band of clearing orientations narrows with the aperture.

Also implemented: the convex bar as the active load with the T retained (it is the subject of a
published falsification and deleting it would orphan the finding); channel walls via
`ArenaGeometry.wall_depth = 1.5`, with 0.0 still building the legacy thin segment; a broadside
canonical pose, since at angle 0 the bar is already aligned and would have derived a budget from an
instance no episode ever sees; a start-angle band of 80-100 deg; `x_range` shortened to 2.4 so no
pose starts inside the channel mouth; and fingerprint schema v2, which gains a `load_shape` field
and an `actions` group so a change to `hold_orientation` re-keys the dataset.

### Result of the fix

**30/30 seeds at every rung**, at the frozen budgets, with the translation-only search exhausted and
the full optimum containing an explicit rotation. Full suite green at 354 passed.

### So-what

**The transferable lesson is about acceptance procedures, not geometry.** A benchmark certified on
the seeds it will be evaluated on, at a budget fixed after the certificate, can pass while failing a
quarter of the time - and the failure is invisible precisely where it is measured. Certify on a
strictly larger sample than the study will use, and treat the budget as part of the criterion.

**And the right fix for a degeneracy is usually structural, not a tuned constant.** Friction was the
obvious lever and would have been the wrong one: the effect survived a factor of seven in friction.
Holding orientation removed the mechanism instead of suppressing its symptom, and it is defensible
from the task's own referent rather than from what made the numbers work.

---

## 2026-08-27 (later) — No width and no depth restores rotation for a non-convex load; the successor task is a convex bar

- **Area:** task geometry, the load's shape, the simulator's collision-fidelity contract, and the
  fallback-ladder decision (rung 2 vs rung 3). Follows directly from the DSE-057 entry below.
- **Status:** pre-freeze, pre-compute. The CPU spike lives outside the repository; no arena constant,
  load shape, prompt or macro action has been changed by this entry. It records what the spike
  established and the design decision taken on it.

### Trigger

The entry below established that rotation is unnecessary at every shipped difficulty and closed with
"whatever rung 2 becomes must change what the wall *is* — a channel with depth". Before implementing
that, it was tested on CPU, as the corrected instrument now makes possible in minutes.

### Finding

**Wall depth does not fix it, and the reason is the load, not the wall.** A grid over channel depth
∈ {1.5, 2.5} × aperture ∈ {1.35, 1.40, 1.45} × start angle ∈ {0°, 30°, 60°} contains **no
rotation-required cell**. Every cell is either impossible or translation-only feasible. Two causes:

- The effective aperture is `nominal − 2 × wall_radius`, i.e. 0.1 narrower than the declared width.
- **The T is non-convex, so its collision-free configuration space through a channel is not
  characterised by a single bounding y-extent.** The bar and the stem can occupy different
  longitudinal positions relative to the channel faces, admitting translation-only paths that a
  whole-outline clearance calculation excludes. This is a configuration-space property, not a
  physics-engine artefact: it survives re-certification at 16 collision substeps.

The start-angle exclusion proposed as the rescue does not rescue it. Starts whose orientation is
infeasible for the channel come back **impossible**, not rotation-required: the load cannot reach a
feasible orientation and cannot pass in the one it has.

**Two further mechanisms were found, and both are general.**

- **Integration squeeze.** A candidate tunnel path was solvable at the shipped `substeps = 4` and
  died at 8: at coarse collision resolution a macro impulse drives the load through an aperture
  narrower than its own outline before contact resolves. Every frozen E3 certificate was re-checked
  and is stable from 4 to 64, so nothing recorded is affected — but **a feasibility verdict that
  moves with the integrator cannot gate an experiment**, and the shipped default is therefore not a
  certification standard.
- **Passive self-alignment.** Macro impulses are applied at the COM and carry no torque, yet contact
  at the aperture mouth rotates the load anyway. "Translation-only in the action space" is not
  "rotation-free in the state space": at apertures ≥ 0.90 a convex bar aligns itself against the
  channel and passes with no rotate action ever issued. This is the subtlest of the three, because
  the action log shows no rotation while the state trajectory contains one.

### Impact

Rung 2 cannot be a parameter change of any kind — not width, not depth, not the start distribution.
The T-load arena is falsified as a rotation-necessity manipulation, and the falsification is
structural rather than a mis-set constant.

A **convex** load does work: fit is then governed by extent, with no configuration-space escape. A
1.4 × 0.3 bar in a depth-1.5 channel at aperture 0.80 is certified rotation-required at start angles
80°, 85° and 90° — translation-only search exhausted, full optimum 9–10 steps containing 1–2
rotations, replay stable at 16, 32 and 64 substeps.

### Risk reduced

Implementing a channel for the T — a real change touching the arena, the grid serialiser, the
geodesic and the budgets — that the spike shows would not have worked. And, separately, certifying
any future geometry at a fidelity that had already been demonstrated to give a wrong answer.

### Correction path (and what was rejected)

- **Rejected — tune widths again.** Falsified by the entry below.
- **Rejected — depth alone, or depth plus start-angle exclusion.** Falsified above.
- **Rejected — an L-bend corridor (true piano-movers).** It is the most faithful fix, but it changes
  the geodesic, and `y_binary_progress` is built on a monotone x-geodesic. That is an outcome-variable
  change, which §6 forbids at this stage, and it would not be one coherent package applied once.
- **Rejected — raising the shipped `substeps` default to 16.** Result-affecting for no benefit: the
  frozen certificates are already stable, and it would re-key every dataset.
- **Taken — a convex bar in a finite-depth channel, declared as a successor task, plus a separate
  certification fidelity profile.**

### The fix (decided, not yet implemented)

**Two tracks in parallel, both funded; the arena is not abandoned.**

- **Track C — rung 3 and RQ3a**, starting immediately, needing no GPU and no arena. The methodological
  finding is written up in `docs/EXPERIMENTS.md` ("The headline methodological finding").
- **Track A — the convex-bar successor task.** Labelled honestly: *a successor rotation-control
  benchmark, adopted after the original T-load benchmark was falsified as a rotation-necessity
  manipulation*. It is **not** presented as a repair of the T arena and **not** as the primary
  dissertation result unless it clears certification with time for one clean re-gate. The change of
  embodied object is logged here as a protocol deviation driven by a physics-engine limitation, since
  it alters the load's affordances, the spatial representation agents see, the difficulty mechanism
  and the task fingerprint.
- **`CERTIFICATION_STEP_CONFIG` (substeps = 64)** is added to `sim/feasibility.py`. `StepConfig.substeps`
  is a pure resolution knob — total simulated time is `settle_steps × dt` regardless of it — so the
  shipped default of 4 is left untouched for reproducibility while all new-geometry acceptance runs at 64.

### The certification standard (fixed before any candidate is adopted)

Every declared seed at every difficulty must satisfy **all** of:

1. full-action search solvable at `CERTIFICATION_STEP_CONFIG`;
2. translation-restricted search **exhausted** without reaching the goal, same fidelity;
3. the full-action solution contains ≥ 1 rotation;
4. replay of that solution succeeds at 16, 32 **and** 64 substeps;
5. a strict, pre-declared budget margin — not success on the final permitted action;
6. the verdict is invariant to a conservative collision-margin perturbation, wall radius included;
7. the realised **angle trajectory** under the translation-restricted search shows no passive
   alignment large enough to substitute for a commanded rotation.

Limb 7 is new and exists solely because of the passive-alignment mechanism. The measured rotation
quantum — **exactly 33.7°, deterministic** (min = max over 72 applications) — is used as a candidate
generator and early-rejection screen only. A band half-width of ≥ 17° is **not** an acceptance
criterion: 33.7° does not divide 360°, the band *centre* must be reachable on the lattice from the
declared start, and in-contact rotation differs from open-space rotation. Physics certification is
the criterion; the lattice is the screen.

### So-what

**Three independent degeneracy mechanisms turned up in one task, and each needed a different check
to see.** Staged crossing needs an exhausted restricted search; integration squeeze needs invariance
to collision resolution; passive self-alignment needs the realised state trajectory, because the
action log looks clean. That triple is the transferable content of this project's arena work,
independent of whether the successor task ever runs.

**The arena is not abandoned, and the reason is not sunk cost.** The convex bar is a genuinely
different manipulation with a rotation-required band of **[0.4, 1.5)** in nominal aperture against
the T's empty band — an order of magnitude more design room, certified rather than assumed. It is
worth one properly gated attempt. What has changed is that it is now a *declared successor task*
with a published falsification behind it, not a quiet retune, and it runs alongside a fallback that
no longer depends on it.

---

## 2026-08-27 — Rotation is not necessary anywhere: the task's cognitive core is absent

- **Area:** task geometry and difficulty semantics — whether the arena tests what DSE-006 says it
  tests — plus the acceptance criterion for the fallback ladder's rung 2 (DSE-057).
- **Status:** pre-freeze, pre-compute. No arena constant, prompt or macro action was changed by this
  entry; it repairs the *instrument* that was to decide the rung-2 change, and reports what the
  repaired instrument says.

### Trigger

The 2026-08-26 entry below closed with a rung-2 acceptance criterion and a script,
`scripts/check_rotation_need.py`, said to "decide both halves on CPU in seconds". Before spending
any GPU time behind that gate, the script was reviewed against its own claim. It did not meet it: it
never called the feasibility solver. It decided "is rotation necessary?" by running **one
hand-written policy** — close the y gap, then push east — and reading its failure as necessity.

### Finding

**A policy that fails proves nothing, and this one was hiding the largest task-validity defect in
the project.**

Replacing the policy with the existing A\* oracle restricted to the translation actions `N/S/E/W`,
and exhausting that search, inverts the answer at both difficulties the old script "passed":

| Difficulty | Slit | Full-action optimum | Translation-only optimum |
|---|---|---|---|
| easy | 1.8 | 7 steps, **0 rotations** | 7 steps |
| medium | 1.2 | 13 steps, 2 rotations | **13 steps** — rotation buys nothing |
| hard | 1.1 | 13 steps, 1 rotation | **14 steps** — rotation saves one step |

Over 10 jittered seeds × 3 difficulties, **0/30 meet the necessity criterion**: 28 admit a
translation-only path inside budget and 2 have a full-action optimum with no rotation at all.

The cause is the arena's own geometry, already written down and not followed through. The internal
walls are `pymunk` segments of radius 0.05 — planes with no depth — so the load never has to fit the
gap all at once. The bar crosses at its 0.3 thickness; the stem then crosses at its 1.0 length; each
clears a 1.1 slit alone. `sim/arena.py` says as much ("the TIGHTEST threadable slit is the shorter
member = the stem = 1.0 (rotation cannot beat this)"). What had not been drawn from that sentence is
that the staged crossing is a **translation** manoeuvre — so rotation is redundant at every slit
width, and no choice of widths can restore it.

### Impact

DSE-006 calls rotation through the slits **"the cognitive core of the task"**. That core is absent
from every rung of the shipped ladder. Three earlier readings are superseded:

- *"Easy is the invalid cell"* — easy is invalid, but so are medium and hard, for a deeper reason.
- *"Medium is the untested rung that requires rotation"* — it does not require rotation; running it
  would have bought a third measurement of the same defect at full GPU cost.
- *"Widen the band via `T_THICK`"* — the band `[stem, thick+stem)` describes **head-on single-shot
  clearance**, not necessity. Widening it changes which slits clear in one shot and leaves staged
  translation untouched. It would have been a real change that fixed nothing.

It also explains the failure signature the channel analysis could not: the `ROT+,ROT-,ROT+,ROT-`
oscillation consuming up to 30 of 33 steps in the hard cell is the pair hunting an angle that never
needed finding.

### Risk reduced

The largest available: a rung-2 geometry change, and a GPU re-gate behind it, chosen against a
criterion that could not detect the defect it was written to detect. Under the old script the
`T_THICK` widening would have returned ACCEPTED, the re-gate would have run, and it would have failed
for a reason the instrument was blind to — spending the last rung of the ladder on a non-fix. The
same script would also have returned ACCEPTED for a **geometrically impossible** arena, since it
never checked solvability at all.

### Correction path (and what was rejected)

- **Rejected — soften the wording and proceed.** The criterion was not imprecise, it was invalid.
- **Rejected — keep the policy and add more policies.** Any finite set of policies still cannot
  establish necessity; only exhausting a restricted search can.
- **Rejected — carry the monotone-rotation probe** proposed as a way to separate "action space
  inadequate" from "agents will not commit". A rotation-free path is trivially monotone, so limb 3
  subsumes it; building it would have measured nothing new.
- **Taken — make the criterion a proof.** Three limbs, all per jittered seed: the full-action optimum
  is solvable inside the certified budget; it contains ≥ 1 rotation; and the same search restricted
  to translations is exhausted without reaching the goal.

### The fix

`solve()` gains two keyword arguments, `actions` and `scenario`, both defaulting to the frozen
behaviour so `certify()` and the budget certificate are bit-for-bit unchanged. Restricting the action
set turns the oracle into a necessity proof; supplying a scenario runs it per jittered seed rather
than only on the canonical pose. `scripts/check_rotation_need.py` is rewritten around the three
limbs and reports a named reason per seed (`unsolvable`, `over_budget`, `zero_rotation_optimum`,
`translation_only_feasible`). `tests/unit/scripts/test_rotation_instrument.py` pins the falsifying
facts, including the false-accept on an unsolvable arena.

### Result of the fix

REJECTED at **every** difficulty — easy, medium and hard — where the previous version rejected easy
alone. Two claims elsewhere were falsified by the same pass and corrected in place:

- **The handoff-level statistics do not survive episode clustering.** `docs/EXPERIMENTS.md` asserted
  that the C0-vs-C4 stuck-rate effect "survives a cluster correction comfortably but the exact figure
  needs one". Computed: difference +0.145, episode-cluster 95 % CI **[−0.005, +0.294]**, permutation
  *p* = **0.085**. C0−C1 and C0−C3 also cross zero. The nominal *p* ≈ 10⁻⁷ was entirely an artefact
  of treating ~500 clustered handoffs as independent across 20 episodes.
- **`MIN_CYCLE = 4` counted actions, not repetitions**, as the changelog claimed. Renamed
  `MIN_CYCLE_ACTIONS`; no reported fraction changes.

### So-what

**The rung-3 finding is now stronger, and it is a different finding.** It was going to rest on a
handoff-level *p*-value that has just evaporated. What replaces it is not an estimate at all: an
exhaustive search showing that a task built to require a spatial-reasoning manoeuvre does not require
it, that the two-agent pair spends its budget attempting the manoeuvre anyway because the sender
keeps instructing it, and that degrading the channel improves outcomes precisely by deleting that
instruction. The methodological contribution — *a coordination benchmark can look like it exercises
a capability while admitting a degenerate solution, and the information-theoretic read of the channel
inverts when it does* — is a proof plus a categorical observation (10/10 C1-hard failures are the
literal sequence `E,E,E,E,E,E`), not a contested test.

**Two general lessons, both cheap and both nearly missed.** A necessity claim needs an exhaustive
search over a restricted action set, never a policy that failed; and a clustered design needs its
clustered test computed rather than asserted — the correction here did not shrink a *p*-value, it
removed the finding it was attached to.

**The open question is no longer "which slit widths".** No width restores necessity while the walls
are depthless planes. Whatever rung 2 becomes must change what the wall *is* — a channel with depth,
which forces the whole body through a swept volume and makes orientation binding — or the arena track
ends and rung 3 carries the result. That decision is not taken here.

---

## 2026-08-26 (later) — The channel was degrading a wrong instruction, not information

- **Area:** the identification of RQ1 itself — what `apply_channel` actually manipulates — plus the
  construct boundary of G3 and the task validity of both difficulty cells (E3 attempt 2, DSE-019).
- **Status:** pre-freeze. F0 is still open. This is the fallback ladder firing, declared before any
  further compute is spent.

### Trigger

E3 attempt 2 — the re-gate on prompt v5 and seeds 0–9, `eddd19c654515bb2`, 80 episodes on Myriad
bf16 — returned `fallback`. G1 FAIL 0.300, **G2 FAIL −0.200 with the success gap sign-inverted**,
G3 PASS 0.999. The retune was spent; under §6 there is no attempt 3, so the question was not "what
lever next" but "what did this instrument actually measure".

### Finding

**The degraded channel beat the clean one, and the reason is that A's clean message is confidently
wrong.**

The T's y-extent never exceeds 1.553** at any orientation in the circle (minimum 1.300 at 0°,
maximum 1.553 at −146.8°), and the easy slit is **1.8**. The load therefore clears the gap head-on
**from every possible angle**, with at least 0.247 of clearance at its worst orientation — so on easy,
rotation is not merely unused, it is *geometrically incapable of being necessary*. This is the
documented design intent (`arena.py`: "easy 1.8 (head-on, trivial)"), not an accident of one starting
pose. Simulated over the ten jittered pilot seeds, a rotation-free policy — close the y gap, then
push east — solves **10/10 within budget** (`scripts/check_rotation_need.py`). At medium (1.2) and
hard (1.1) the same policy solves **0/10**: those slits are below the extent at every angle and do
require the threading maneuver.

Against that, A's C0 message (95 tokens, mean) reports the
true pose and concludes *"it may not fit through the slit unless rotated … Rotate the load so it is
aligned with the slit before pushing east."* Every number in that message is true — which is exactly
why G3 scores 0.999 — and the inference is false. B complies, and the per-action rotation quantum is
coarse (measured mean 31.3°, modal 33.7°, damped to 11–19° on wall contact), so "align it vertically"
is not a command this action set can execute cleanly. B hunts for an angle until the budget runs out.

Truncate that message to 8 tokens (C1) and the instruction never arrives: **all seven C1-easy
successes are pure `E,E,E,…`, 8–13 consecutive pushes — the oracle optimum.** 7/10 against C0's 3/10.
Corrupt it (C4, 0.4 dropout, 95 → 56 tokens) and the surviving imperative fragments still point east:
7/10 again.

**C3 is the control that fixes the direction.** It is the only condition that degrades what B
*observes* rather than what A *says*, and it is the only degradation that hurts — easy success 1/10,
worst stuck rate (0.466), lowest progress (0.073), lowest `y_binary_progress` (0.275) in the sweep.
The same nominal degradation carries **opposite signs depending on which channel it lands on**. Two
rival explanations die on that contrast: it is not noise-breaks-cycles (C1's truncation is
deterministic and helps as much as C4's stochastic dropout), and it is not terseness (within C0-easy,
length does not predict success — successes average 99.7 tokens against failures' 95.6).

Two further defects surfaced, both prior to the channel question:

- **Easy cannot require rotation**, by the geometry above; a rotation-free policy solves 10/10
  jittered seeds. The cell tests whether B can avoid being talked out of the obvious, not whether the
  pair can coordinate. The seeds C1-easy succeeded on are *exactly* the pure-push-east success set.
- **Medium was never run.** The pilot cell is easy and hard, and medium is the only rung that both
  requires rotation and is less extreme than the hard cell nobody solved.
- **Hard is solved by nobody — 0/60 episodes** across both attempts, all four conditions, both prompt
  versions. The hard half contributes no outcome variance at all.

And the retune's own result is informative: v5 hit its target and missed the outcome. C0-easy
terminal cycling fell **0.667 → 0.143** while success went 0.4 → 0.3. Breaking the fixed point was
necessary and not sufficient, because the erroneous instruction is regenerated fresh every step.

### Impact (had it not been caught)

The main sweep would have run on a task whose easy cell needs no coordination and whose hard cell is
unreachable, and would have reported a **negative** information gradient as the headline RQ1 result.
Every downstream arm inherits it: RQ3b would have calibrated an outcome-thresholded gate against an
inverted outcome, and the gate would have learned to block the *helpful* (degraded) messages. Worse,
the result would have looked clean — G3 at 0.999, a well-behaved estimator, a positive CPVI gap and
a tidy control task all point at a healthy instrument.

### Risk reduced

The largest single threat to this dissertation: publishing a confounded RQ1 whose sign is an artefact
of the task, defended by gate numbers that could not detect the artefact. It also removes the
possibility of calibrating RQ3b on an inverted target, which would have been undetectable after the
fact and would have invalidated the causal arm rather than merely weakening it.

### Correction path

Rejected: a third retune (forbidden by §6, and the mechanism is not a tuning problem). Rejected:
jumping straight to rung 1 alone — RQ3a needs no GPU, so treating it as a *replacement* for the arena
track spends nothing and gains nothing. Rejected: changing *Y* to rescue the gate — the primary label
`y_binary_progress` shows the same inversion (C1 0.932 > C4 0.479 > C0 0.308 > C3 0.275), so the
defect is in the task, not the label.

Taken: **rungs 1 and 2 in parallel**, because they do not compete for a resource.

### The fix

1. **Rung 1 — RQ3a elevated and started now** (DSE-041, DSE-042). CPU and network only, so it costs
   the arena track nothing and is the insurance that can carry the thesis alone.
2. **Rung 2 — one declared task-geometry change, then one re-gate.** The acceptance criterion is
   checkable on CPU before any GPU time: **the A\* optimum must contain ≥ 1 rotation and finish
   strictly inside budget, for every jittered seed, at every difficulty.** That makes easy a genuine
   coordination test and hard reachable in the same change, and it is verifiable by the oracle that
   found the defect.
3. **RQ3b deferred behind rung 2**, explicitly, for the calibration reason above.
4. **G3 gains a correctness limb** at F0, alongside its grounding limb — see takeaway 3.

### Result of the fix

Pending. Rung 2's re-gate has not run; the acceptance criterion is stated in advance and is
falsifiable without a GPU, which is the point.

### So what / takeaways

1. **V-usable information is sign-blind, and this run separates the signs.** CPVI answers *how much
   does the message reduce a V-bounded predictor's uncertainty about Y*, not *does acting on it help*.
   Here the clean channel carries **+0.100 bits more** usable information **and less than half the
   success rate**, measured on the same 1,925 handoffs by the same estimator, with C3 fixing the
   direction. A message can be more informative and more harmful at once. This is the project's
   sharpest result to date and it is on the dissertation's own topic.
2. **It gives the circularity guard empirical teeth.** §6 already bans calibrating the runtime
   statistic against CPVI, on principle. This shows what the ban buys: a CPVI-calibrated gate would
   have scored C0's messages highest and promoted exactly the messages that caused the failures.
   `CosineStatistic`'s probe-independence and outcome-only calibration are now motivated by data.
3. **Groundedness is not correctness, and G3 cannot tell them apart.** 0.999 on a corpus whose modal
   inference was wrong. A check verifying message numbers against true state is by construction blind
   to a false conclusion drawn from true numbers — and this generalises to any faithfulness-style
   check on inter-agent messages.
4. **Instruction-following dominates self-observation.** v5 showed B its own failed rotations and
   their zero gain; B kept rotating while A kept instructing. Any gate that hopes to change behaviour
   by re-prompting has to reckon with that ordering.
5. **The instrument being wrong did not invalidate the findings.** 1–4 rest on the run's *internal*
   contrasts — C3 against C1/C4, CPVI against outcome, oracle against observed plan — so they survive
   the task-validity defect that rung 2 exists to fix. Designing the pilot so its controls are
   internal is what made a failed gate still worth something.

---

## 2026-08-26 — The retune: a greedy fixed point, not a capability ceiling

- **Area:** the task's action loop and prompt surface — the mechanism behind the G1 failure at the
  bf16 re-gate, and the one retune PREREGISTRATION §6 allows (E3 attempt 1, DSE-019/DSE-055).
- **Status:** pre-freeze. F0 is still open; this changes the design before v1 is committed.

### Trigger

E3 attempt 1 — the verdict of record, Myriad bf16, job 214590 — returned `retune_once`: G1 FAIL
0.400, G2 PASS with its success half at exactly the threshold, G3 PASS 0.986. Under §6 exactly one
retune is available and a second failure fires the fallback ladder, so the lever had to be chosen
from the data rather than from the most plausible-sounding story.

### Finding

**Every one of the 34 failed episodes spent its full step budget** — 965 handoffs against a cell
maximum of 1,020, and `graph.py` exits only on success or budget. The obvious inference is that the
budget is too small. **It is wrong, and the data say so unambiguously**: mean geodesic distance still
to run at the end of a failed episode is **7.02** against a goal radius of 0.8 in a task spanning
about 8, and **only 1 of 34 failures ends within 1.5 of the goal**, while every one of the 6
successes finished in 8–12 steps of an 18-step easy budget (oracle optimum 7). The failures are not
short of road. They are stationary.

Reconstructing the action sequences shows what they are doing instead. **18 of 40 episodes — 53 % of
the failures — terminate in a period-1 or period-2 limit cycle**, and the cycle consumes **68 %** of
the steps those episodes spend: `ROT+,ROT-,ROT+,ROT-,…` (5 episodes, up to 27 consecutive steps),
`N,S,N,S,…` (6), and `E` pressed into an impassable wall for the entire budget (7, including three
`E×33`).

**The mechanism is structural, not stochastic.** Decoding is greedy by design and the v4 prompt
surface is a pure function of the current scene — it carries no action history and no record of what
the last action achieved. So the policy is memoryless: any state whose chosen action returns the
system to that state is a **fixed point**, and any two-state orbit is a **period-2 cycle**, and in
both cases escape is impossible because escape would require the prompt to differ and the prompt
depends only on the state. The repository had already made exactly this argument for the runtime
gate — `GATE_FEEDBACK` exists because "under greedy decoding a re-prompt is a fixed point" — and had
never applied it to the base loop.

Two corroborating observations. **This is not model capability**: the same cell scored 0/3 easy-C0
at 4-bit 8B and 2/5 at bf16 14B — real improvement into the same wall. And **C1 (the 8-token cap)
collapses to a single-action policy**: 7 of its 8 failures are pure cycles, 5 of them `E` for the
whole budget. That is why the *outcome* gradient is flat rather than merely noisy — at easy seed 1
the capped channel succeeded in 8 steps while the clean channel spent 15 straight `ROT-`. A degraded
channel can outscore a clean one when truncation happens to encode the near-optimal policy.

### Impact

The G1 failure was being read as "the task is above this tier", which points at the fallback ladder
or at a bigger model. Both readings are wrong and both are expensive. The binding constraint is that
**the agents cannot observe that what they are doing is not working**, and no amount of model
capability or step budget repairs a policy that is a function of a state it keeps returning to.

Left unfixed, the RQ1 outcome axis is unusable: 34 of 40 episodes are decided by an artefact of
memorylessness rather than by the channel, which is the condition under which H1's outcome half
cannot be measured at all — while H1's *information* half is meanwhile perfectly healthy (+0.243-bit
C0−C4 CPVI gradient, control at zero, selectivity +0.136).

### Risk reduced

Three. **(i)** Spending the single retune on the wrong lever and arriving at the fallback ladder with
the actual defect untouched. **(ii)** Reading a structural artefact as a capability result and
escalating the model tier — a compute decision taken on a misdiagnosis. **(iii)** Reporting the flat
C0-to-C4 success gradient as a null when it has a known non-random cause.

### Correction path

The step budget was the first candidate and is **rejected on the evidence above and recorded as
rejected**: at a mean 7.02 remaining, more steps buy more cycling. Raising the model tier is rejected
because the 4-bit-to-bf16 comparison already shows the wall is not capability. Widening the geometry
is rung 2 of the *fallback* ladder, available only after the retune fails, and would in any case not
touch a cycle occurring in open space.

### The fix

**Prompt surface v5** — the state gains one line naming the last four actions, the geodesic distance
each gained, and their net. Four is the smallest window that shows a period-2 cycle twice.

Three properties are load-bearing and each was chosen against an alternative:

- **Fact, not instruction.** The line reports what happened and leaves the inference to the agent. A
  directive ("this is not working, try something else") would convert an observability fix into a
  behavioural intervention, and after the run the two would be indistinguishable in the result.
- **Appended after `apply_channel`, not inside the serialiser.** The channel must degrade one thing
  only. C3 restricts B's view of the *world*; B's memory of its own actions is not the world, so the
  history survives the restriction. Routing it through the serialiser would instead have required a
  per-form whitelist/window rule and would have made C3-plus-grid behave unlike C3-plus-numeric.
- **Shared, not A-only.** Giving the history to A alone would create an information asymmetry
  *outside* the channel and break the standing invariant that `observation == state_str` in
  C0/C1/C2/C4 — a change to the CPVI conditioning semantics, which is a freeze-level decision, not a
  retune. **The cost is accepted and named:** a receiver that can self-correct from its own history
  needs the message less, so this may depress CPVI. G2's CPVI half is the place that will show it,
  and it currently has +0.243 bits of headroom above a directional threshold.

Two changes ride alongside and are **not** the retune. The cell widens to **seeds 0–9** — a precision
change that moves no threshold and no estimator, and that cannot be optional stopping because the
attempt-1 estimate (0.4) sits *below* the 0.5 threshold, so added *n* moves the expected verdict
toward FAIL. And **`detect_stuck` now measures net displacement across a 5-state window rather than
span across 3**: the old form scored `stuck=False` for all 18 handoffs of the `N,S,N,S` episode (the
COM genuinely moves a unit each step and returns) and for the `E×18` wall press (contact jitter
exceeds the 0.02 threshold), so the field that exists to name a trajectory going nowhere was blind to
the failure mode that consumed the run. No gate reads `stuck` and nothing terminates on it, so this
changes what a run records about itself and never what it does.

**A construct boundary v5 forced, found in review before attempt 2 ran.** G3 scores a message's
numbers against "every number the sender was shown", which was geometry-only until v5 put per-action
geodesic gains into the same string. With `g3_abs_tol = 0.5` and gains clustering in 0–1.5, a
fabricated small-magnitude geometric claim then matched a *gain* and scored grounded: on a synthetic
record, a message asserting an offset of 0.85 that no wall, slit, load or goal coordinate supports
reads **0.0** against the geometry and **1.0** against geometry-plus-history. G3's truth set now
excludes the history line.

The decision worth recording is which way to fail. The two available errors are not symmetric.
Including the history credits fabricated geometry — single-sided inflation of exactly the property
the gate certifies, and PREREGISTRATION §6 fixes the construct as "match true geometry". Excluding it
penalises a message that correctly quotes a gain — a stricter gate, and rare, because A's prompt asks
for position and intent rather than for gains. **A gate should fail closed**, so the history is
excluded. This also refines the D19 principle: "the truth set is what the sender was shown" was a
sound rule while everything the sender was shown was geometry, and v5 is the first time that stopped
being true. The rule is now "what the sender was shown, restricted to the world" — and it is keyed on
an exported prefix rather than a literal, so the serialiser and the gate cannot drift apart silently.

### Result of the fix

Not yet known — attempt 2 has not run. What *is* fixed by construction: `PROMPT_VERSION` feeds
`dataset_hash_for`, so v5 resolves to a new dataset (`eddd19c654515bb2`) and attempt 1
(`1c994b87bbca8257`) can be neither resumed into nor overwritten. The plan costs 80 cells and 4,080
upper-bound model calls, roughly 45 minutes of sweep at the measured ~1.5 calls/s.

### So-what

Three things worth carrying into the write-up.

**A deterministic multi-agent loop needs memory in the prompt or it is not ergodic.** Greedy decoding
is chosen here for reproducibility, and reproducibility bought a policy that cannot leave a state it
re-enters. This is a general property of memoryless deterministic agents in deterministic
environments, not a quirk of this task, and it is the same argument DSE-045 already makes for the
gate — the pilot showed it fires without a gate too.

**"Ran out of budget" and "made no progress" look identical in aggregate and are opposite diagnoses.**
The distinction cost one query over the raw records and would have cost a wasted retune otherwise.
Any horizon-limited agent benchmark reporting only success rate and step count cannot tell them
apart.

**Channel degradation is not monotone in outcome when the degraded message can encode a better
policy.** C1's truncation to a bare direction outperformed the clean channel on one seed. Any
information-gradient design that assumes "less channel ⇒ worse outcome" needs to check this
explicitly; here it is the honest mechanism behind a flat success gradient that would otherwise be
reported as noise.

---

## 2026-08-24 — The RQ3a substrate, opened: the conditioning state is there, the outcome is not

- **Area:** RQ3a external validity — the substrate, the outcome *Y* on real logs, and the refit
  arm's confounding structure (methodology §9.8, roadmap §3.4, DSE-041/042/047).
- **Status:** pre-freeze. No RQ3a analysis has been run; this entry records what the corpora
  contain, measured, before any of it is written into a chapter.

### Trigger

The fallback ladder's first rung is "elevate RQ3a to the headline — it can carry the dissertation
alone", and the bf16 re-gate can return `retune_once` a second time. That pivot has to be
executable on measured ground, not on a substrate described only in papers. Phase F is scheduled
parallel from day one for exactly this reason; it had never actually been opened.

### Finding

Five things, of which two change the design.

**1. TraceElephant is the substrate the design assumed — under different field names.** The
ticket and roadmap §3.4 both say it records `input_context` and `output_content` per step. It
records `input` and `output`, and both are structured objects: `input` is an OpenAI-shape
`{messages, model, stream}` and `output` is a full `ChatCompletion`. The substance holds — the
receiver-observed context genuinely exists per step, which is the whole reason the substrate moved
off Who&When — but the mapping needed writing rather than reading. (The HuggingFace repository is
also auto-tagged `format:imagefolder` / `modality:image`, triggered by screenshot PNGs inside some
web-agent traces. Taken at face value that tag says the corpus is images.)

**2. TraceElephant is failure-only, and one of the two stated reasons for the substrate move does
not hold.** Roadmap §3.4 moved RQ3a off Who&When for two reasons: TraceElephant records the
receiver's input context, and it "ships non-failing executions too, so trace-level outcome is
genuinely two-class". The first is confirmed above. The second is false.

The corpus is **220 traces, not 380**, and every one of the 220 carries a populated
`mistake_agent` / `mistake_step` / `mistake_reason`. The "380 executions of which about 220 are
annotated failures" reading treated 220 as the failure subset of a larger pool; 220 is the whole
pool, and it is entirely failures. Only the 44 `swe-agent-runs-swe-bench` traces carry any
annotation-free outcome (`tests_status`, a SWE-bench harness result), and **0 of those 44 pass every
test**. The other 176 traces have no outcome field at all — `captain-*` and `magentic-*` carry a
`ground_truth` string but no recorded final answer to compare it against, and the ground-truth
string appears in the last three steps' output in only 1 of 20 sampled traces, so it cannot be
recovered by matching either.

So on the primary corpus the refit arm is undefined for exactly the reason it is undefined on
Who&When — the reason the substrate moved in the first place.

**3. MAST's non-failure class exists at the assumed size, and is confounded with system identity.**
DSE-047 flagged the non-failure proportion as resting on all-zero annotation rows glimpsed in a
preview. Counted over all 1642 traces: **405 (24.7%)**. The assumption holds. But the class is
distributed wildly unevenly across the seven systems — AG2 52.1%, Magentic 21.5%, HyperAgent 10.0%,
ChatDev 7.6%, MetaGPT 5.1%, AppWorld and OpenManus 3.3% each.

**4. MAST traces are unsegmented strings.** `trace.trajectory` is one formatted transcript per
trace, laid out differently by each system, median ~9.7k characters and up to 2M. There are no
recorded step boundaries.

**5. Who&When is exactly as characterised.** 184 traces, **184 failures, zero non-failures**, 4092
messages, and no per-step input context anywhere.

### Impact

**On *Y* for logs (methodology §9.8), and it is the opposite of convenient.** The four-way table
ranked Y3 (counterfactual replay) as the recommended definition, with Y1 (trace success) as the
cheaper option that needed a mixed-outcome corpus. Finding 2 says no such corpus is in hand at step
level: Y1 is **degenerate** on both step-level corpora, because a constant cannot be predicted.

This promotes DSE-042 from an upgrade path to **the load-bearing route**. Counterfactual replay is
now the only way to obtain a within-trace two-class target on a corpus that also records the
per-step conditioning state. Y2 (annotation-as-Y) remains forbidden for circularity — the pressure
this finding creates is exactly the pressure that makes Y2 tempting, which is worth naming out loud
now rather than discovering as a rationalisation later. MAST is the only two-class corpus available
without replay, and it is trace-level only and confounded (finding 3), so the trace-level secondary
cannot quietly stand in for the step-level primary.

**On the MAST refit arm.** Finding 3 means a probe fitted on pooled MAST traces can reach much of
the available accuracy by recognising *which system produced the trace* — AG2 is near a coin flip,
OpenManus is almost always a failure — without reading the message at all. This is the same defect
shape the simulator-side shuffled-message audit found, where condition identity leaked into CPVI
through message style, and it is caught here by the same reflex. Any MAST arm must stratify by
`system_name` or report the system-identity-only baseline beside it, as the control task does for
the simulator arm.

**On MAST's role.** Finding 4 fixes it as a trace-level secondary. It cannot test step
localisation, because the corpus does not record steps.

### Risk reduced

The largest single risk in the fallback plan: that RQ3a is elevated to the headline and *then*
found to rest on a corpus that cannot support the claim. That question is now answered with counts,
and the answer is mixed rather than clean — the *conditioning state* is there in full (5,960 steps,
2,488 inter-agent handoffs, recorded input contexts), and the *outcome* is not.

Knowing that now costs a paragraph. Knowing it after the bf16 re-gate returned `retune_once` a
second time, with rung 1 of the fallback ladder already taken and a chapter part-written, would have
cost the fallback itself. The financial risk is unchanged and slightly sharper: replay is the one
place this project could accidentally spend money, and it is now on the critical path for RQ3a
rather than optional, so DSE-042's dry-run projection and hard spend cap are load-bearing controls
rather than hygiene.

### The fix

`data/logs.py` (`LogHandoffRecord` / `LogTraceRecord`, versioned separately from the simulator
schema, physics fields absent rather than nullable), `experiments/rq3a_load.py` (three loaders, one
interface, local paths only), `scripts/fetch_rq3a.sh`, and `docs/rq3a_schema_mapping.md` carrying
the field-by-field mapping, the counts and the three mapping decisions that change what a number
means. Twenty offline unit tests against hand-built fixtures mirroring the verified layouts.

### Result

| Corpus | Traces | Steps | Handoffs | Failures | Non-failures | Observations |
|---|---:|---:|---:|---:|---:|---|
| TraceElephant | 220 | 5,960 | 2,488 | 220 | **0** | recorded |
| Who&When | 184 | 4,092 | 3,505 | 184 | **0** | reconstructed |
| MAST-Data | 1,642 | — | — | 1,237 | **405** | trace-level only |

Two of three corpora are single-class at trace level; the third has no steps. Full per-family
breakdown, the field-by-field mapping and the three mapping decisions that change what a number
means are in `docs/rq3a_schema_mapping.md`.

The transfer regime is untouched by any of this: a simulator-fitted probe applied to logs needs no
log labels to *fit*, only to *evaluate*, and the step-level `mistake_step` annotation exists on all
220 TraceElephant traces. It is the refit regime that needs replay.

### So-what

Two takeaways, one methodological and one about process.

The methodological one is that **the observability caveat is now a measured property rather than a
hedge**. `reconstructed_observation` is not a defensive flag on a hypothetical: Who&When
demonstrably has no per-step input context and TraceElephant demonstrably does, so the flag
partitions the corpora on a fact rather than on a worry.

The process one is that this is the third time in this project that a cheap diagnostic run before
the expensive thing found a defect the expensive thing would have buried — after the serialiser
that printed no obstacle geometry, and the dataset hash that ignored the retune lever. The pattern
is consistent enough to be a rule: **open the substrate before writing about it.** Reading the
paper's field names would have produced a loader that silently found no `input_context` anywhere;
reading its abstract produced a roadmap paragraph asserting a non-failure class that does not exist.

Worth being precise about which half of the substrate decision survives. Moving RQ3a to
TraceElephant was still right — the conditional construct is impossible on Who&When and works here,
and that is the reason that mattered. But the move was argued on two grounds and only one of them
was true, which is a reminder that a decision can be correct and still be defended with a claim that
does not check out. The roadmap paragraph is corrected in place rather than quietly rewritten.

---

## 2026-08-24 — Dataset identity now carries the world the episodes were simulated in

- **Area:** reproducibility contract and the pilot retune path — what makes two datasets the same
  experiment (G1/G2 re-gate, PREREGISTRATION §6, RESEARCH_ROADMAP §3.1).
- **Status:** pre-freeze, before the bf16 run of record. Deliberately done now: the fix re-keys
  every existing dataset, which costs nothing today and would cost a re-freeze afterwards.

### Trigger

Reviewing what the first cluster session would actually hit, on the path the pre-registration
prescribes: the E3 re-gate returns `retune_once`, and the named lever is difficulty — "difficulty
and serialisation is retuned once, then re-gated" (roadmap §3.1), with the fallback ladder's second
rung reading "simplify the task (wider slits …)".

### Finding

The retune lever was outside dataset identity. `_DIFFICULTY_SLITS`, `ArenaGeometry`, the T's
dimensions, the goal radius, the load mass and `GridConfig.cell` are module constants;
`ExperimentConfig` carries `difficulty` as a *label*, and `ArenaGeometry()` / `GridConfig()` are
default-constructed at three call sites and never threaded from config. So widening a slit left
`sweep_hash` and `dataset_hash_for` unchanged, and `run_grid` — which reads completed `episode_id`s
out of the dataset directory and skips them — would find 40 of 40 episodes complete, log
`0 pending`, run nothing, and let the driver re-analyse the **pre-retune** dataset and re-report its
verdict as the post-retune result.

`sweep.py` already carried exactly this reasoning for the knobs a caller *can* set — "a silent
change to the jitter region, impulse parameters, or the label horizon k would otherwise relabel a
re-run dataset without changing its hash" (P0-2, P1-6). The world itself had been missed.

### Impact

A passing-looking broken run, on the verdict of record, at precisely the moment the design is most
load-bearing: the one permitted retune. The failure is silent by construction — the sweep reports a
complete grid and a plausible verdict — and the run that produced it would have looked, in the
manifest and the log, exactly like a correct one. Under the repo's own ordering of failure modes
("a passing-looking broken run is the worst outcome") this is the worst available class.

### Risk reduced

Removes the possibility of pooling episodes from two different worlds under one dataset identity,
and with it the possibility of a retune that silently does nothing.

### The fix

`sim/fingerprint.py`: `simulation_fingerprint()` returns a typed `SimulationFingerprint` over the
world constants that are *not* reachable from `SweepConfig` — the slit map, arena dimensions, load
geometry, damping/friction/mass, goal radius, grid resolution — plus an
`ENVIRONMENT_SCHEMA_VERSION` escape hatch for a behavioural change that leaves every constant
identical. Its 16-hex `digest()` joins `PROMPT_VERSION` inside `dataset_hash_for`, so changed
geometry writes to a different directory rather than resuming into the one it was meant to replace.

Two deliberate boundaries. Jitter, step and outcome configs are **not** re-fingerprinted: they are
`SweepConfig` fields already inside `sweep_hash`, and hashing them twice would put one guarantee in
two places to keep in step. Derived values (`HALF_H`, `COG_Y`) are omitted as pure functions of the
dimensions that are hashed.

The manifest records the **payload as well as the digest**: the digest is what prevents an unsafe
resume, but only the payload answers *why* an identity changed, six weeks later, without the source
tree. `SWEEP_MANIFEST_VERSION` is bumped 1 → 2 accordingly.

A resume assertion sits behind the hash as defence in depth: on resume, a recorded fingerprint that
disagrees with this process aborts. It is unreachable through the hash by construction, and exists
for what identity cannot cover — a hand-copied directory, or a future change to how the hash is
composed. It deliberately does **not** fail closed on a *missing* manifest: the manifest is written
when a sweep finishes, so its absence is the ordinary killed-at-wallclock case, and failing closed
there would have broken the resumability the guard sits inside.

### Result

The regression test drives the whole chain through `run_grid` rather than asserting on the digest:
run the grid, widen `easy` from 1.8 to 2.4, re-run, and assert a full fresh grid is scheduled and
the pre-retune dataset is left intact. A digest-level test alone would not have caught it — the
claim is specifically that `run_grid` resolves and consumes the new identity. Cross-process
stability is pinned under two `PYTHONHASHSEED` values, since a fingerprint that leaked iteration
order would make every resume look like a geometry change.

Cost paid now: `runs/local/*` and `runs/bench/smoke/*` are no longer resumable. Their findings
survive in `docs/EXPERIMENTS.md`, the committed `runs/bench/ladder.*` table is append-only, and both
local pilots were indicative by pre-registration in any case.

### So what

Difficulty was treated as a *label* in the config and as a *number* in the simulator, and only the
label was part of experiment identity. The general lesson is that a knob named in the
pre-registration as a thing you are allowed to change once is, for that reason, a knob that must be
in the hash — the retune path is the one place a silent no-op is most expensive and least visible.

---

## 2026-08-24 — The second length control, and why it is a sensitivity analysis rather than an adjusted effect

- **Area:** RQ1 analysis — the pre-registered controls for the length/condition confound (H1, H2).
- **Status:** pre-freeze, no bf16 data. The matching rule is fixed here, before the run of record,
  which is the only point at which choosing it is not a researcher degree of freedom.

### Trigger

PREREGISTRATION §5 pre-registers **two** length controls and states that both are reported. Only
one existed. Length enters the outcome model as a covariate (`path_b_length_controlled`) and is
partialled out of the CPVI-progress correlation (`partial_spearman_length`); the promised
**length-matched subsample comparison** had no implementation and no ticket.

### Finding

C1 caps message length, so length is confounded with condition *by construction* — this is not an
incidental nuisance covariate but a direct consequence of the manipulation. A covariate adjustment
handles that only under a functional-form assumption: it extrapolates a fitted length effect into
regions where one arm supplies no episodes at all. With C1 shifting the whole length distribution
down, that extrapolation is doing real work in the estimate, and nothing in the reported numbers
would show how much.

### Impact

"CPVI is just message length" is the first sceptical question this design invites, and C1 is the
condition that makes it sharpest. A single model-based adjustment is a weaker answer than the
pre-registration promises, and the shortfall would only have surfaced at examination.

### Risk reduced

An H1/H2 result whose only defence against the length confound rests on an extrapolation the
reported numbers do not expose.

### The fix

`overlap_restricted_contrast` in `analysis/stats.py`, wired into `RQ1Result.length_matched` as a
per-condition Ck-vs-C0 contrast on **both** success and CPVI. Episodes are stratified into
equal-count quantile bins of episode-mean delivered-message tokens; differences are taken only
inside bins holding at least `min_per_cell` episodes of *both* conditions, then size-weighted
across bins. Defaults: 3 bins, floor of 2.

Quantile strata rather than nearest-neighbour caliper matching, deliberately. At the E3 cell's six
episodes per condition a caliper has too little support and can silently collapse the comparison to
one or two idiosyncratic pairs. Coarse strata fail *visibly* instead: `n_bins` falls, `n_kept`
falls, and where the distributions do not meet at all the result returns `interpretable=False` and
a NaN delta rather than a confident number. `delta_unrestricted` is reported alongside every
restricted delta so the restriction's effect is always legible.

### Result of the fix

Five unit tests pin the behaviour, including the load-bearing known-answer case: when the outcome is
a deterministic function of length alone and the arms differ only in their length distributions, the
restricted delta is 0.0 while the unrestricted delta is not.

### So what

**This is reported as an overlap-restricted, length-adjusted sensitivity analysis — never as a
clean estimate of the channel effect with length removed.** The restriction is what makes it
informative and equally what makes it partial: the retained episodes are a non-random subset of both
arms, so the contrast generalises only to the lengths both arms actually reach. That framing is
carried in the model docstring, in `ANALYSIS_PROTOCOL["length_control"]`, and in PREREGISTRATION §5,
so it cannot be quietly upgraded to a causal claim in the write-up.

The general lesson: where a manipulation *creates* a confound rather than merely admitting one,
model-based adjustment and overlap restriction answer different questions, and reporting only the
first flatters the result.

---

## 2026-08-24 — The served revision has to come from the file the manifest reads

- **Area:** provenance — what a recorded model revision can be trusted to mean.
- **Status:** pre-freeze, closed before the run of record.

### Trigger

Reading the Myriad jobscripts end to end before the first cluster session. Both took `REVISION` on
the `qsub` line, copied by hand from `configs/model/<tier>.yaml`. The manifest records the revision
from that config file. Nothing compared them.

### Finding

A typo, a stale copy-paste, or a `TIER` change without a matching `REVISION` change would serve one
checkpoint while the manifest recorded another — and **no check in the repo could detect it.** The
client's health check compares the served *model id*, which was enough to catch a wrong tier, but
`/v1/models` carries no revision at all. Every artefact would have been well-formed and the run
would have looked clean.

### Impact

CLAUDE.md's rule is that a result with an unrecorded revision is not a result. A *wrongly* recorded
one is worse: it survives audit. The bf16 re-gate is the verdict of record for the Y/V freeze, so
this is the run where provenance matters most.

### Risk reduced

A frozen result attributing episodes to a checkpoint that did not produce them.

### The fix

`scripts/myriad/_common.sh:resolve_tier` reads `name` and `revision` from `configs/model/<TIER>.yaml`
— the same file the manifest records them from — so the two cannot disagree. `MODEL`/`REVISION`
survive as overrides for the 70B-AWQ tier, which has no config file until DSE-005 pins its repo id,
and an override contradicting the config prints a warning naming both values, because in a job log a
deliberate override and a typo are otherwise indistinguishable. A unit test asserts every tier config
carries a full 40-character SHA, the invariant the shell now depends on.

### Result of the fix

`REVISION=` disappears from the runbook: `qsub -P <project> scripts/myriad/pilot.sh` is the whole
command, and `-v TIER=qwen8b` the whole fallback.

### So what

Two values that must agree should have one source, not a convention for keeping them in step. The
tell was that the *only* thing standing between correct and silently-wrong provenance was a human
copying a 40-character hex string correctly, under queue pressure, at the start of a session.

---

## 2026-08-24 — The gate's retry has to say something new: a versioned feedback template, and why not temperature

- **Area:** RQ3b gate design — the retry path of the causal arm (H6), and what the four arms are
  allowed to differ in.
- **Status:** pre-freeze, no data. This is a design decision recorded *before* any gate result
  exists, which is the only time it can be made honestly.

### Trigger

DSE-018 (runtime gate integration) was written as: score the handoff, block it if the statistic
falls below the calibrated threshold, re-prompt A, retry up to a bound. Reading it against the
serving configuration — greedy decoding, `temperature=0`, fixed seed, pinned revision — the retry
step is a **fixed point**. The same prompt yields the same message, therefore the same statistic,
therefore the same block, for every one of the bounded retries.

### Finding

The arm would have passed its own unit tests. A test that mocks the client and asserts "blocked
handoffs are retried and recorded" passes whether or not the retry changes anything, because the
mock returns what it is told to. Live, the gate would have blocked, retried N times to no effect,
and then either forced the original message through or spent the step — and the causal contrast
would have measured the cost of *stalling*, not the value of *blocking low-information handoffs*.
That is not a null result; it is a result about the wrong quantity, and it would have been very hard
to detect after the fact because every artefact would have looked well-formed.

### Impact

H6 is one of the two hypotheses that make the gate more than a measurement exercise. RQ3b is one of
the two pre-planned arms (with RQ3a) that carry the dissertation if RQ1's gradient is weak. A
vacuous retry does not merely weaken H6, it silently redefines it.

### Risk reduced

A treatment that appears implemented, passes CI, and measures something other than what its
hypothesis names.

### Correction path

The retry prompt must differ in **content**. Two ways to achieve that:

1. **Raise the temperature on the retry.** Escapes the fixed point with a one-line change.
2. **Append a feedback template**, telling A what a usable instruction has to contain.

Option 1 is **rejected, and the rejection is recorded here and in PREREGISTRATION §6 before any gate
data exists**, on two grounds. First, it breaks the determinism story mid-episode: the run would be
greedy everywhere *except* at exactly the handoffs the gate touched, so the arm that gets the
treatment is also the only arm with sampling noise, and the repo's "seed-pinned, revision-pinned"
claim would need a carve-out precisely where the causal claim lives. Second, and worse, it confounds
the treatment: a post-retry improvement could be the feedback, or it could be the plain fact of
having sampled twice. H6's four arms (gate on / random-rate-matched / always-retry / off) are built
to differ in one thing at a time, and the `always-retry` arm exists specifically to price the
"retried at all" effect — which only works if retrying does not also change the decoding regime.

### The fix

`GATE_FEEDBACK` in `agents/prompts.py`, appended to A's user turn on a blocked retry and nowhere
else. It instructs A to state, from the numbers in front of it, the push direction, whether the load
must rotate first and which way, and the direction of the goal — the three things the task's own
failure analysis (design log, the v4 prompt bump) showed A omitting when its messages went generic.

Three properties make it auditable rather than an implementation detail:

- **Versioned separately from `PROMPT_VERSION`.** `GATE_FEEDBACK_VERSION` is its own constant
  because the template is part of the *treatment*, not the base task: it reaches a model only on a
  blocked retry, so a wording change re-shapes the RQ3b arm while leaving every ungated dataset
  byte-identical. One version bumping the other would be wrong in both directions.
- **Recorded in every run manifest**, beside `PROMPT_VERSION`.
- **Deliberately absent from `dataset_hash_for`.** The gate is unbuilt (DSE-018), so today the
  template reaches no model; folding it into the hash would re-key every existing dataset over a
  string nothing reads. It must join the dataset hash when DSE-018 makes retries live — recorded
  here and in the code comment so that step is not forgotten rather than merely deferred.

### Result

`prompt_a(state, gate_feedback=False)` is byte-identical to the previous `prompt_a(state)`, pinned
by a test, so no existing dataset shifts. A blocked retry issues a demonstrably different user turn,
also pinned by a test — which is the assertion that would have caught the vacuous version.

### So-what

The general shape is worth naming: **a treatment whose mechanism depends on the decoding regime has
to be checked against that regime, not only against its own tests.** Greedy decoding is chosen here
for reproducibility, and reproducibility is exactly what made the retry inert. The tests that would
have passed were not bad tests; they tested the plumbing, and the defect was in the physics. Where a
hypothesis says "intervening changes the outcome", at least one test has to assert that the
intervention changes the *input to the model*, not merely that the code path ran.

---

## 2026-08-24 — Freezing the probe: a control task, repeated cross-fits, and a wider re-gate cell

- **Area:** probe family *V* and its audit (control-task selectivity), per-handoff score stability
  (repeated cross-fits), the message-length confound, and the size of the E3 re-gate cell.
- **Status:** pre-freeze. Every number below comes from re-scoring the E3-local v4 dataset
  (`c0bd4d7499f01d97`, 599 handoffs, 24 episodes) with **no new model calls**
  (`runs/local/c0bd4d7499f01d97-report/rescore_frozen_estimator.json`).

### Trigger

The two remaining tickets that block the Y/V freeze — DSE-043 (control tasks) and DSE-044 (repeated
cross-fits and length control) — were scheduled after the Myriad bf16 re-gate on the grounds that
they block F0 and not the gate. That ordering is wrong. The re-gate is the **verdict of record**; if
the estimator gains selectivity and per-handoff stability afterwards, the recorded verdict was
produced by a different estimator from the one the thesis freezes, and the choice is then between
paying a second GPU hour and explaining a mismatch in the methods chapter. Both tickets are pure CPU
work with no allocation and no queue. They were built first.

### Finding

**1. The probe family had no selectivity evidence.** Probe accuracy alone cannot distinguish "the
representation encodes this" from "the probe learned the task" (Hewitt & Liang, EMNLP 2019). At
1,536-dimensional concatenated features on 599 handoffs that is a live objection, and the held-out
AUROC monitor does not answer it: AUROC says the probe generalises on *these* labels, not that it
would fail on labels carrying no signal. Measured with random labels at the observed base rate,
through the same splitter and probe family: **control CPVI = −0.006 bits** pooled, every condition
between −0.011 and −0.003. Negative, as the pre-registered directional prediction said it must be —
`g_cond` carries twice the features, so against noise it overfits harder and scores *worse* held
out. Selectivity is therefore essentially the whole score: **+0.072 bits** pooled, **+0.187** in C0.
The check is not inert: an almost-unregularised probe at n = 30, d = 128 reads **+0.93**, and an
over-capacity MLP **+1.39**, so the capacity ladder has something to fire on when it should.

**2. A single cross-fit carries more per-handoff noise than the smallest effect being claimed.**
Averaging over five fold assignments, the mean across-repeat SD per handoff is **0.042 bits**. C3's
entire mean CPVI is **+0.058**. Those per-instance scores are not decorative — they are the mediator
in H2 and the input to all of RQ2, and an instance whose score moves ±0.042 by fold assignment can
change sign between analyses. Pooled mean CPVI itself moves from +0.078 (canonical assignment) to
**+0.066** (five-assignment mean): fold choice was worth **16%** of the pooled score.

**3. Message length does not explain CPVI.** "CPVI is just message length" is the first sceptical
question the design invites, and C1 manipulates length directly, so the answer belongs in the
protocol rather than in a rebuttal. Spearman(CPVI, delivered token count) = **−0.085** — no monotone
relationship in either direction. The partial Spearman of CPVI with progress given length is −0.006.

**4. G1 rests on three episodes.** The re-gate cell was seeds 0–2, so G1 — the gate that has already
failed once, and the only one bf16 can plausibly flip — is a three-episode read. A design whose true
easy-C0 success rate is 0.67 fails a ≥ 0.5 threshold on three episodes about a third of the time.

### Impact

Without (1) the freeze would have covered a probe family with no evidence that it cannot manufacture
information; the objection is standard and would have been raised at viva with no answer in the
artefact. Without (2) every per-handoff score in the mediation and in RQ2 would carry fold noise of
the same order as the effects being reported. Without (4) the re-gate risks spending its one retune
on sampling noise.

### Risk reduced

R6-adjacent (measurement validity): the probe family now carries a falsifiable selectivity audit
with a numeric firing rule and a pre-specified remedy ladder, and the per-handoff scores carry a
reported stability. The estimator that produces the re-gate verdict is the estimator that gets
frozen at F0 — no post-hoc substitution.

### Correction path

Written in this order, deliberately: the **rules first**, before any control score existed.
PREREGISTRATION §5 gained the operational firing rule (`mean_control_cpvi > 0.02` bits, or its
episode-cluster interval excluding zero from above) and the capacity ladder that fires on it
(tighten ℓ₂ to `C = 0.1` → within-fold PCA to 128 components → the smaller encoder, applied to
*both* probes, never to `g_cond` alone). The 0.02 threshold is fixed as under 10% of the smallest
CPVI difference the design must resolve — the G2 gap, whose cluster interval starts at +0.060. §5
also pinned the repeat semantics and §8 gained the selectivity and length-control reporting. Only
then were the estimators implemented and the data re-scored.

### The fix

`control_labels` + `control_task_cpvi`; `ProbeConfig.n_repeats` with `cpvi_with_sd` returning
per-instance mean and across-repeat SD; `partial_spearman`; `path_b_length_controlled` on the
mediation; `cpvi_sd` and `msg_tokens` persisted in the scores table. R = 5 wired into the pilot and
RQ1 configs, matching what §5 already pre-registered. Repeat 0 is the canonical fold assignment, so
`n_repeats = 1` still reproduces the unrepeated estimator exactly — no recorded score is disturbed
by the mechanism itself. The E3 cell was widened to seeds 0–4 (40 episodes) with the amendment dated
in §6, pre-freeze and pre-run.

### Result of the fix

The frozen estimator on the same data: C0 **+0.181** [+0.050, +0.302], C1 +0.050 [−0.250, +0.317],
C3 **+0.058** [+0.013, +0.108], C4 −0.031 [−0.084, +0.020]. The C0−C4 gap is **+0.212** against
+0.211 under the recorded estimator — the headline is invariant to the change. The permutation test
still passes at R = 5 (real +0.066 against a null of +0.033 ± 0.006, max +0.046, p = 1/21), and the
null's height falls with repeats, as a fold-noise component should. The pilot verdict is unchanged:
`retune_once`, G1 0.000, G3 0.977. The recorded v4 table is **not** overwritten — it stands as the
dated reading of the estimator of the day, with the frozen-estimator reading reported alongside it.

### So what

Two takeaways. First, the ordering rule this closes: *build the estimator that will be frozen before
spending the compute whose verdict it produces.* A measurement change is cheap before the run and
expensive after it, and "it doesn't block the gate" is a dependency fact, not a scheduling argument.
Second, the control task is what lets the thesis say the CPVI numbers are a property of the
messages rather than of the probe — and it says so with a number (+0.072 selectivity against a
−0.006 control) rather than an argument. Combined with the permutation test, the two audits now
bracket the estimator from both sides: permutation asks whether *this* message mattered, the control
task asks whether *any* probe of this capacity could have manufactured the answer.

---

## 2026-08-24 — Last pass before Myriad: cluster-honest intervals, and a permutation null with a structural floor

- **Area:** uncertainty reporting for per-handoff CPVI summaries; the pre-registered
  shuffled-message negative control; dataset identity under resume; provenance on the pilot report.
- **Status:** pre-freeze, zero headline episodes. Every number below is from re-analysis of the
  E3-local v4 dataset (`c0bd4d7499f01d97`, 599 handoffs) with **no new model calls**
  (`runs/local/c0bd4d7499f01d97-report/reanalysis.json`).

### Trigger

A full-stack review before the Myriad re-gate: estimator, gates, runner, serving path, statistics
and documents re-read against standard V-information practice, and the E3-local dataset re-scored
to check what the recorded numbers rest on. Re-scoring reproduced every recorded point estimate to
the digit (grouped folds and the probe fits are deterministic given record order), and the probe
overfit monitor is healthy — held-out AUROC 0.727 for `g_cond` against 0.566 for `g_base` and 0.807
in-sample, so cross-fitting holds at dim 1536 on n = 599. Two findings did not survive the audit.

### Finding

**1. The per-condition CPVI intervals resampled handoffs as if independent.** Handoffs within an
episode share a start pose, a trajectory and overlapping next-k label windows; the episode is the
sampling unit. The H1 mixed model already treats it as one — the descriptive intervals did not. At
six episodes per condition the iid handoff bootstrap read roughly **half the honest width**: C0
[+0.141, +0.243] handoff-level against [+0.059, +0.307] episode-cluster. What survives the honest
interval is exactly what the E3 entry led with: C0's CPVI excludes zero, and the C0−C4 gap
(+0.211 bits) holds a cluster interval of [+0.060, +0.349]. What does not: C1's interval
([−0.197, +0.320]) is uninformative at pilot scale, and C3's only just clears zero
([+0.002, +0.103]).

**2. The pre-registered shuffled-message null cannot reach zero on this design — and the
pre-registration demanded that it does.** §8 said shuffling "must collapse CPVI". Measured: real
pooled mean +0.078 bits; null +0.043 ± 0.006 (max +0.057 over 20 within-condition permutations).
The floor is structural, not an estimator defect: permuting within condition preserves every
condition-level signature the message *style* carries — an 8-token C1 message is recognisably C1, a
dropout-riddled C4 message recognisably C4 — and per-handoff progress base rates differ strongly by
condition (C0 0.255, C1 0.735, C3 0.379, C4 0.575), so a permuted message still tells the probe
which condition its handoff is in, and condition predicts progress. Left frozen as written, a
*valid* estimator would have failed its own manipulation check on the main sweep.

### Impact / risk reduced

Both were interpretation traps armed for the Myriad stage. The interval defect overstated pilot
precision exactly where the thesis quotes intervals; the null criterion would have converted a
structural property of the design into an apparent estimator failure *after* the freeze, when the
only available responses would have been a logged deviation or a false alarm.

### The fix

`analysis/stats.py` gains `cluster_bootstrap_ci` (episodes resampled with replacement, handoffs
pooled; percentile) and `rq1.py`'s per-condition CPVI/PVI intervals use it; the E3 entry's table is
corrected in place with the change named. PREREGISTRATION §8's criterion becomes the permutation
test — the real pooled mean must exceed every permutation — with the null's height read as the
*identity* component of CPVI and the real-minus-null excess as per-handoff message content, the
correction dated and made pre-freeze. Alongside, three smaller closures from the same review:
`sweep_hash` no longer hashes `concurrency` (an execution knob; changing worker count on a resumed
run would have re-keyed the dataset and orphaned every completed episode), `PilotReport` embeds
`AnalysisProvenance` (the re-gate verdict is a result of record and must carry its revisions), and
the last carriers of the falsified "near zero by construction" claim (the CLI cell comment,
RESEARCH_ROADMAP §2.3, methodology §8.3's sender-conditioning overstatement) are corrected.

### Result

Suite green with the new tests (cluster CI degenerate cases, hash invariance under concurrency,
provenance presence). The corrected E3 numbers stand as re-published in `docs/EXPERIMENTS.md` §7;
no verdict changes — `retune_once` on both local runs, the ledger still closed.

### So what

- The pilot's headline observation — a positive, zero-excluding C0 CPVI and a +0.211-bit C0−C4 gap
  — **survives the honest interval**. The full ordering C0 > C1 > C3 > C4 remains a point-estimate
  observation at six episodes per condition; the bf16 re-gate sizes against it.
- The RD-15 audit is stronger after the correction, not weaker: "real exceeds every permutation" is
  a test that can actually fail, and the null's height becomes an interpretable quantity — how much
  of CPVI rides on *which condition you are in* rather than *what this message says*.
- The lesson generalises D20: claims of the form "this quantity is zero (or cannot be zero) by
  construction" about a probe-relative measure keep being wrong in both directions. The last pass
  hunted down the remaining instances before they could shape a Myriad-stage reading.

## 2026-08-24 — Four defects in the pilot gate, and the information gradient they were hiding

- **Area:** the pilot gate — how G2 estimates CPVI, what population G1 scores, what G3 counts as
  ground truth, and whether C3's receiver restriction still restricts anything.
- **Status:** found by two E3 runs, on the **interim local substrate**, at **zero headline episodes**.
  No result is re-frozen; the one-retune ledger remains closed.
- **A correction is recorded inside this entry.** The first reading of the first run was that CPVI is
  near zero *by construction* wherever the receiver already sees the sender's state. That reading is
  **wrong**, it was written into these documents, and the second run falsified it. It is corrected in
  place below rather than deleted, because the mistake is instructive: it is exactly the error the
  V-information framing exists to prevent.

### Trigger

E3-local: C0/C1/C4 × easy/hard × seeds 0–2, 18 episodes and 445 handoffs against
`mlx-community/Qwen3-8B-4bit`. The verdict came back `retune_once` with G1 and G2 failing, and the
G2 numbers were odd enough to audit rather than accept: a **negative** success gap (C4 outscored C0)
and a CPVI gap of +0.007 bits between two quantities that were themselves ≈ 0.03 bits. The audit
found four defects, and the second run — 24 episodes, 599 handoffs, with C3 added — showed that with
all four corrected the same data carry a CPVI gradient of **+0.211 bits**.

### Finding

Four things, in descending order of consequence.

**1. G2 refitted its probe on a two-condition subset, and that is not the estimator the study uses.**
`g2_signal` selected the C0-plus-hardest rows, featurised *those*, and fitted the CPVI probe on them.
Pointwise V-usable information is defined as per-instance scores from **one** fitted probe; refitting
per contrast discards the other conditions' rows and shifts the class balance the probe sees. On the
same 24-episode data the subset fit read the C0-minus-C4 CPVI gap as **+0.012 bits** and the
whole-cell fit reads **+0.211 bits** (C0 +0.192 [+0.141, +0.243], C4 −0.018 [−0.065, +0.028]). The
gate was not measuring a weak gradient; it was measuring a strong one badly.

**The correction, stated plainly.** The first reading of this run — written into the changelog, the
methodology's deviation register, the pre-registration and this log before the second run — was that
`observation` being byte-identical to `state_str` in 445 of 445 records makes CPVI "near zero by
construction", so the C0/C1/C4 cell could never have passed G2. **That is wrong.** CPVI is
*V-usable* information: it asks what a message adds to what a **bounded probe family** can extract
from the receiver's state, not what it adds to what the receiver formally holds. A logistic probe on
a frozen sentence embedding of `load=(2.63, 2.03, -0.70) …` cannot compute the geodesic; a message
that says "push north, you are below the slit" states the answer. So a message can carry real usable
information even when the receiver holds the identical bytes — and it does: **C0 CPVI is +0.192 bits
with an interval excluding zero, in the condition where the receiver sees everything.** The error was
to reason about Shannon-style redundancy in a place the design deliberately uses a model-relative
measure, which is the precise confusion V-information was introduced to resolve.

What survives from the first reading, and what the PVI − CPVI gap actually said: the gap on the v3
run was **+0.0000 bits [−0.0005, +0.0005]**, which says the *state-only baseline extracted nothing*
about next-*k* progress from these embeddings, so conditioning subtracted nothing. That is a fact
about the baseline probe's power, not about the receiver's knowledge, and on the corrected v4 run the
gap is small but structured (C0 −0.036, C1 +0.051, C3 +0.037, C4 +0.012) — i.e. conditioning
*raises* the C0 score and lowers the others.

**2. G1 was scored on the wrong population.** The pre-registration says "C0 self-play episode success
≥ 0.5 **at easy difficulty**"; the implementation averaged C0 across *both* difficulties. On the E3
cell that mixes a solvable geometry with one designed to be hard, so a pair that solved every easy
episode and no hard one would score exactly **0.5** — passing the capability floor by arithmetic
accident, on a task it had half failed.

**3. G3 scored a sender as hallucinating geometry it had been shown.** `_record_grounding` drew its
truth set from `rec.state`, which carries the **load body only**. From v3 the serialiser also printed
the wall abscissae and the slit interval, and from v4 the load's dimensions, so a message correctly
citing "the slit runs 2.1 to 3.9" was counted as fabricating both numbers. G3 read **0.720** on the
v4 run — a FAIL — for messages that fabricated essentially nothing; with the truth set taken from
what the sender was actually shown it reads **0.977**, and the v3 run reads **0.999** rather than
0.811. This is the same rot as defect 4 below, with the same cure: derive the truth from the state,
never from a hand-maintained list of keys.

**4. C3's numeric restriction had rotted against the v3 prompt surface.** `_restrict` blacklisted the
`goal=` line. v3 added `walls_x=` and `slit_y=`, and the blacklist kept delivering both, so C3's
receiver still held the full arena layout and the asymmetry the condition exists to create was
nominal. The docstring claimed "the goal and global layout must come from A's message"; only the
goal did.

### Impact (had it not been caught)

Compounding, and in the direction that produces a confident wrong answer rather than a crash. The
common shape is that **every one of the four made the study look worse than it is**, so each failure
would have been read as evidence about the design rather than about the instrument.

- **RQ1's headline number would have been understated seventeen-fold.** The subset refit reports
  +0.012 bits where the whole-cell fit reports +0.211 on the identical data. A gradient that clears
  any reasonable floor would have been reported as noise.
- **G2 would have been reported as a failed signal gate.** Under the fallback ladder a failing G2
  that survives one retune sends RQ1 to a documented negative and elevates RQ3a to the headline. The
  project would have abandoned its primary research question on an estimator artefact — and, worse,
  the abandonment would have looked principled, because the pre-registration commits to reporting a
  null rather than chasing a positive.
- **G3 would have failed the pilot on honest messages.** 0.720 against a 0.8 floor, for messages that
  fabricated nothing; the sender was penalised for citing geometry the serialiser had printed for it.
- **G1 would have passed the wrong pairs.** The 0.5 floor against a 50/50 easy/hard mix converts the
  capability gate into a difficulty gate at exactly the boundary value.
- **C3 would have been an inert arm** in the one condition carrying a genuine observation asymmetry —
  the same defect class as the sender-conditioning error of §8.3 (D1), recurring because a blacklist
  was left to track a serialiser that changed.

### Risk reduced

Two, and the second is the one that matters. The first is ordinary: three gates now measure what they
are documented to measure. The second is that **an instrument artefact was one run away from being
reported as a research finding.** This project's pre-registration explicitly commits to publishing a
null — which is a virtue, and also a hazard, because a design prepared to accept a negative result
will accept one that its own estimator manufactured. The defence is not scepticism about nulls; it is
that every gate value gets audited against the raw data *before* the verdict is acted on, which is
what happened here.

### Correction path

- **G2** fits the CPVI probe **once over the whole cell** and contrasts the resulting per-instance
  scores, which is the estimator the RQ1 analysis uses. No per-contrast refit.
- **G3** takes its truth set from what the sender was shown — the numeric leaves of `state` *plus*
  every number in `state_str` — so it cannot rot when the serialiser gains a key.
- **G1** now scores easy C0 only, and raises `ConfigError` when no easy C0 episodes are present rather
  than silently averaging whatever is there.
- **C3** now *whitelists* B's own state keys (`load=`, `contact=`) instead of blacklisting global ones,
  so a serialiser that gains a key fails closed rather than leaking it.
- **The E3 cell and the serialisation surface** are design decisions, not defects, and were held open
  for an explicit call rather than changed in passing. Both were recorded here *before* either was
  taken, and both were then taken on 24 August 2026:
  - **C3 joins the pilot cell** (C0/C1/C3/C4 × easy/hard × seeds 0–2, 24 episodes). The reason given
    when the decision was taken — that C3 is the only condition where CPVI can be non-zero — was the
    mistaken one corrected above. The decision survives its own bad argument: C3 is the only
    condition carrying a real observation asymmetry, it is in the headline design, and a pilot that
    never exercises it certifies an instrument the main sweep will not use. Its measured CPVI is
    **+0.051 [+0.026, +0.075] bits** — positive, tightly bounded, and *lower* than C0's, which §8.3's
    "most room to be positive" did not predict and which is flagged for the bf16 re-gate rather than
    explained at six episodes per condition.
  - **G2 gained a third verdict state**, `assessable=False`, pointed at the case that genuinely
    admits no verdict: a cell in which every handoff carries the same progress label, so CPVI has
    nothing to predict. An unassessable gate never yields `proceed` and never spends the retune or
    invokes the fallback. (It was first pointed at the shared-observation case, on the mistaken
    premise above; that trigger is removed.)
  - **The numeric form names the load's own dimensions** (`load_size=(1.4000, 1.3000)`), and A's
    system prompt states that the whole load must fit the slit rather than its centre. Prompt
    surface **v4**, bump two of the three budgeted before E3, with one remaining.
  - The alternative of printing the *derived threading band* (`pass_band_y=(2.75, 3.25)`) was
    considered and rejected: it performs the geometric inference on the agents' behalf and would
    weaken what a C0 success demonstrates about coordination. So was reporting the load's *current
    rotated* y-extent, which is a stronger cue than a constant and is held in reserve as the third
    bump if the re-run shows the alignment error surviving.

### The fix

Four code changes on `infra/DSE-031-033+049-first-run-unblock`, each with a test that fails against
the old behaviour: `g2_signal` featurises and fits over all records rather than the C0-plus-hardest
subset; `_record_grounding` unions `state_str`'s numbers into the truth set; `g1_capability` filters
to `difficulty == "easy"`; `_restrict` uses `_C3_NUMERIC_KEEP = ("load=", "contact=")`. Plus the two
design decisions above: C3 in the pilot cell, and prompt surface v4.

### Result of the fix

Both datasets re-gated with the corrected code (v3: 18 episodes / 445 handoffs; v4: 24 episodes /
599 handoffs, C3 included):

| | v3 cell, corrected gates | v4 cell, corrected gates |
|---|---|---|
| G1 capability (easy C0) | **FAIL** 0.000 (0/3) | **FAIL** 0.000 (0/3) |
| G2 success gap (C0 − C4) | **FAIL** −0.167 | **FAIL** −0.333 |
| G2 **CPVI gap** (C0 − C4) | +0.012 bits | **+0.211 bits** |
| G3 groundedness | **PASS** 0.999 (was 0.811) | **PASS** 0.977 (was 0.720) |
| Verdict | `retune_once` | `retune_once` |

CPVI by condition on the v4 cell, one probe fitted over all 599 handoffs, *Y* = `y_binary_progress`,
2 000-resample percentile intervals:

| | C0 | C1 | C3 | C4 |
|---|---|---|---|---|
| CPVI | **+0.192** [+0.141, +0.243] | +0.084 [−0.032, +0.194] | +0.051 [+0.026, +0.075] | −0.018 [−0.065, +0.028] |
| PVI | +0.157 [+0.115, +0.198] | +0.135 [+0.003, +0.264] | +0.088 [+0.034, +0.136] | −0.007 [−0.047, +0.035] |
| PVI − CPVI | −0.036 | +0.051 | +0.037 | +0.012 |

**The gradient is monotone in the channel and in the predicted direction: C0 > C1 > C3 > C4, with C4
at zero.** That is H1's shape, at 6 episodes per condition, on a 4-bit model that cannot complete the
task. It is not a result — it is the first evidence that the instrument is capable of producing one,
which is exactly what a pilot is for. The verdict remains `retune_once`, and per the pre-registration
a 4-bit local G1 failure is indicative and does **not** spend the retune; the ledger opens at the
Myriad bf16 re-gate.

**The outcome half went the other way and this is not glossed.** Episode success was C0 0/6, C1 1/6,
C3 0/6, C4 2/6 (3 of 24 overall) — the clean channel came last, and G2 fails on its success half in both runs. With G1
at zero there is no headroom for success to fall, so the outcome contrast has nothing to measure
until a tier that can do the task is in the loop. The honest statement is that the information half
of H1 and the outcome half of H1 cannot both be assessed at this tier, and only the second is
blocked.

### The alignment error, recorded before it was decided

The failure mode inside the easy episodes is legible and is not a coordination breakdown. All three
easy C0 episodes reached chamber three (final `com_x` 8.44 / 8.91 / 9.17, goal at x = 10, r = 0.8) and
ran out of the 18-step budget short of the goal, having spent steps on one repeated error (this is the
observation the v4 bump above answers): the model
declares the load "aligned with the slit" when `com_y` is *outside* the printed slit range — 2.0074
called within (2.1, 3.9) — and then pushes east into the wall. Part of that is a 4-bit model doing an
interval comparison badly. Part of it is ours: the state names the **gap's** extent and never the
**load's**, so the band of `com_y` that can actually thread the slit (±0.25 about the centre for the
easy 1.8 gap against a 1.3-tall load, not the full ±0.9) is not derivable from the prompt without a
constant the prompt never supplies. That is the same defect family as the v3 entry above — the state
describing the obstacle but not the thing being manoeuvred — and it is logged here before any decision
is taken, because changing the serialisation after seeing which cells failed is exactly the move the
freeze exists to police.

**What v4 bought, measured.** A now makes the comparison correctly — "The load's center is at y=2.03,
which is below the slit's lower bound of 2.1" is the first message of the v4 run, against v3's
"currently aligned with the slit … since its y-position (2.0074) is within the slit's y-range". The
description is fixed and the **outcome is not**: easy C0 remained 0/3. So the alignment error was a
real defect in the prompt surface and *not* the cause of the G1 failure, which is a capability limit
of the 4-bit tier. That is worth knowing before the bf16 re-gate, and it is why the third prompt bump
stays unspent.

### So what

- **An estimator is part of a gate's definition, not an implementation detail.** G2's threshold was
  right and its arithmetic was right; it fitted the probe on the wrong rows, and reported a strong
  gradient as noise. "Fit once on frozen embeddings" is a reproducibility rule in `CLAUDE.md`; it is
  also, it turns out, a correctness rule.
- **V-usable information is not Shannon redundancy, and the difference is easy to lose.** The
  mistaken reading corrected in this entry — that a receiver holding the state makes CPVI zero — is
  the exact confusion the V-information framing exists to dissolve, and it survived three documents
  before data contradicted it. The measure is defined relative to a bounded probe family; "the
  receiver already has it" is not an argument unless the probe can use it.
- **Blacklists rot; whitelists fail closed.** C3 leaked because the v3 serialiser gained two keys and
  nothing forced the channel to notice; G3 mis-scored for the same reason from the other side. Every
  place the design says "B must not see X", or "X is the truth", is now derived from the state.
- **Audit the gate value against the raw data before acting on the verdict.** Every one of the four
  defects made the study look worse than it is, and the design is pre-registered to accept a null.
  A study willing to report a negative needs its instruments checked hardest, not least.
- **The register keeps its clock property.** All four findings predate any headline episode; none is
  contingent on an outcome, because no outcome of record exists.

---

## 2026-08-24 — The numeric serialisation named no obstacle: prompt surface bumped to v3, and the serialisation axis de-confounded

- **Area:** the prompt surface — what the state serialisers expose, what the two system prompts ask for, and what dataset identity covers.
- **Status:** decided during S1 from the first transcript read (E1), at **zero headline episodes**. One of the three budgeted pre-E3 prompt bumps. Nothing is re-frozen.

### Trigger

The E1 transcript read — the deliberately human step — on the first five real episodes
(C0, numeric, easy, seeds 0-4, local 4-bit tier). 40% episode success, which on its own looks like a
weak-but-alive baseline.

### Finding

The success rate was not the finding. The transcript was.

- **A emitted 7 distinct messages across 75 handoffs**, all variants of "Push the load rightward to
  align it with the slit. Rotate it counterclockwise if needed…" — no coordinate, no offset, no
  reference to anything that changes between steps.
- **B chose `E` 75 times out of 75.** A constant policy: it was not conditioning on the message, and
  barely on the state.
- Tracing why A had nothing to say produced the real defect: **the `numeric` serialisation contained
  no wall or slit geometry at all** — only the load pose, a contact flag and the goal. A could not
  say "you are 1.0 below the slit" because the slit's position was not in A's input.

That last point contradicted the serialiser module's own stated invariant, that the three forms are
*isomorphic in information* and differ only in surface form. The grid **draws** the walls and slits;
the NL form **names** the nearest slit centre and distance; numeric named neither. The serialisation
factor was therefore not a representation A/B — it was partly an information A/B.

### Impact (had it not been caught)

Three compounding failures, and the first is the one that would have ended the project quietly.

- **CPVI would have been ≈0 by construction, in every condition.** Seven near-identical messages
  carry almost no information about anything, so `g_cond` could not beat `g_base` no matter how the
  channel was degraded. RQ1's information gradient would have been flat — and flat *because of the
  prompt*, not because of the channel. That reads exactly like "G2 fails: no measurable signal",
  whose documented response is the one allowed retune and then the fallback ladder.
- **G3 would have been vacuous rather than failed.** Groundedness is scored as the fraction of
  numeric mentions in a message that match the true geometry. Messages containing no numbers have no
  mentions to check, so the gate would have returned a degenerate pass on an empty denominator.
- **The serialisation robustness arm would have measured the wrong thing.** A numeric-vs-grid
  difference would have been reported as a representation effect when part of it was the grid simply
  containing information the numeric form withheld.

### Risk reduced

The class here is **a degenerate channel that still produces a complete, well-formed dataset**. Every
manifest would have been valid, every episode labelled, every gate computed — and the headline
finding would have been an artefact of a missing line in a serialiser.

### Correction path

Considered and rejected: (a) prompt A harder without changing the serialiser — it cannot cite what it
cannot see, so this would have taught it to invent coordinates, which is worse than saying nothing
and would have corrupted G3; (b) switch the headline serialisation to `grid` — that abandons the axis
rather than fixing it, and picks the winner before the experiment; (c) accept the confound and note
it — the serialisation arm is one of the robustness cells the thesis reports, so a known confound in
it is not something to write around.

### The fix

**PROMPT_VERSION v2 → v3**, covering three changes that are one change in effect:

- `_numeric` gains `walls_x=(4.0000, 8.0000)` and `slit_y=(lo, hi)` with the width in the comment.
  The three forms now expose the same information, and the isomorphism claim in the module docstring
  is true rather than aspirational.
- **A's system prompt** states the convention (+x toward the goal, +y north), states that passage
  depends on the load's y matching the slit's y-range, and asks for the load's position *relative to
  the next slit* plus the next move, using the numbers in front of it rather than generic advice.
- **B's system prompt** now says A sees more of the scene, and to follow A's instruction unless its
  own observation plainly contradicts it. The action hint spells out the axis meanings, closing E1's
  third check ("does 'north' mean what the arena means by it?").

**Dataset identity moved with it.** `sweep_hash` covers the sweep config, which carries no prompt
version, so a bump would have resumed into the previous prompt's dataset and pooled two prompt
surfaces into one set of episodes. `experiments.sweep.dataset_hash_for` now folds `PROMPT_VERSION`
into the dataset hash and is the single derivation every reader uses.

### Result of the fix

Re-run of the same five episodes under v3 (see the E1 entry in `docs/EXPERIMENTS.md` §7 for the
numbers). The check that matters is not the success rate but message variety and action variety:
a message that changes with the state is the precondition for any CPVI at all.

### So what / takeaways

**A serialiser is part of the prompt, and an information-isomorphism claim is a testable one.** The
invariant was written down in the module docstring and was false in one of three branches. Nothing
tested it, because the tests checked that each form *round-trips the load pose* — not that the forms
expose the *same* information. The claim the experiment depends on and the property the tests check
were different properties.

**A flat gradient is the most dangerous possible result**, because it is indistinguishable from the
honest negative the design is prepared to report. Everything upstream of the measurement that could
flatten it — an empty message, a constant message, a constant action — has to be checked before the
gradient is believed. E1 exists for precisely this, and it earned its place twice in one session.

**Read the transcript, not the score.** 40% success on C0/easy looked like a live baseline worth
proceeding on. The same run, read line by line, showed one action repeated 75 times. The score was
not wrong; it was answering a different question from the one that mattered.

## 2026-08-24 — Local pilot substrate: the same non-thinking regime as the cluster, and an empty message that would have looked like success

- **Area:** the pre-cluster pilot substrate — how the local runtime is made comparable to the cluster, and what the runner does when a message never arrives.
- **Status:** decided during S1, at **zero recorded episodes**. Nothing is re-run and no result is re-frozen.

### Trigger

The first live model call of the project. LM Studio serving `mlx-community/Qwen3-8B-4bit` at
`localhost:1234/v1`; a two-line probe issuing one `chat` and one `structured` call through the real
client, before any episode was attempted.

### Finding

Two things, one expected and one not.

- **The runtime ignores `chat_template_kwargs`.** The repo disables Qwen3's hybrid thinking per
  request via `chat_template_kwargs={"enable_thinking": False}`, which vLLM honours. LM Studio's MLX
  runtime silently drops it, and so does `reasoning_effort`. Qwen3 therefore reasons, and LM Studio
  routes the reasoning into a **non-standard `reasoning_content` field** rather than into `content`.
- **The failure mode is silent.** With the reasoning in its own field, `content` came back as the
  empty string with HTTP 200. The client's existing guard only rejected `None` and a literal
  `<think>` block, so an empty A-message passed straight through into the handoff record.

Qwen3's in-band `/no_think` switch selects the same non-thinking branch the cluster selects via the
template kwarg, and it works on this runtime: content returns, `reasoning_content` empties.

### Impact (had it not been caught)

An empty A-message is not a degraded message — it is **no channel at all**. Every condition C0-C4
would have delivered the same thing (nothing, or nothing-truncated, or nothing-dropped), so the
information gradient RQ1 exists to measure would have been flattened to zero *by the serving
substrate*, not by the channel. The runs would have completed, written valid manifests, and produced
a clean, publishable-looking null. Under the pilot gates that null reads as "G2 fails, the task
carries no signal" — and the documented response to a G2 failure is to spend the one allowed retune
and then invoke the fallback ladder. The project could have abandoned its headline research question
on the strength of a serialisation bug in a laptop runtime.

The second-order cost is comparability: had thinking simply been left on, the local pilot would have
been iterating prompts against a *different generation regime* from the one the cluster runs, so
every prompt fix validated locally would have been unvalidated where it mattered.

### Risk reduced

The class of failure this closes is **fail-open serving**: an endpoint that returns success while
returning nothing usable. That is the single most dangerous failure shape in this repository,
because its output is indistinguishable from a real negative result.

### Correction path

Considered and rejected: (a) leave thinking on and raise `max_tokens` — burns the budget, and
measures a regime the cluster does not run; (b) switch the local tier to a non-thinking instruct
variant — changes the model rather than the mode, and breaks the local-to-cluster correspondence
this stage depends on; (c) edit the model's chat template on disk — undocumented, unauditable, and
invisible in the manifest.

### The fix

- `ServingConfig.thinking_switch`, empty by default so the cluster path is untouched, appended to the
  **final user turn only**. Both substrates now select the same non-thinking template branch by the
  mechanism each supports, and which mechanism was used is recorded per run.
- `LLMClient.chat` raises on empty or whitespace-only content, not merely on `None`.
- `health_check` asks its ping for 16 tokens rather than 1, so a runtime stuck in thinking mode fails
  **before** a sweep starts rather than at handoff one.

### Result of the fix

The local endpoint returns grounded prose for A and schema-valid actions for B; the E1 smoke runs the
loop end to end. A misconfigured endpoint now fails at the health check with a message naming the
cause.

### So what / takeaways

Two, both about where defects are found rather than what they were.

**The defect was invisible to review and obvious to one live call.** It is not a logic error — the
code does what it says — it is a disagreement between two runtimes about where generated text goes.
No amount of reading finds that. This is the concrete argument for S1 existing at all: the free local
pilot is not a rehearsal for the cluster run, it is the stage that catches the class of defect that
only appears when a real model answers.

**Guard the empty case, not just the absent one.** `None` was already rejected; `""` was not, and
`""` is what a 200 response carries when a runtime writes its output somewhere unexpected. In a
fail-loud repository the invariant to encode is *usable output*, not *present output* — the same
reasoning that already forbids catching a `ServingError` around agent B's action.

## 2026-08-23 — RQ3a re-founded: substrate migrated, and Y on logs defined by intervention

- **Area:** external validity — the RQ3a substrate, the conditioning state on real logs, and the definition of the outcome variable Y outside the simulator.
- **Status:** decided pre-build. DSE-023 superseded by DSE-041; DSE-024 rescoped; DSE-042 created. No RQ3a code exists yet, so nothing is re-run and no result is re-frozen.

### Trigger

Cross-referencing the August architecture review against the roadmap and the methodology text, while writing the Y-on-logs design note that DSE-023 had been blocked on since July. The note could not be written, and the reason it could not be written turned out to be the finding.

### Finding

Three distinct problems with Who&When as the primary RQ3a substrate, which the methodology text had been treating as one.

- **The conditioning state is not in the data.** CPVI is the log-loss reduction of a probe on state-plus-message over a probe on state alone, and the pre-registered semantics fix that state as *the state observable to the receiver*. Who&When records agent **outputs**. It does not record the input context each agent received — the system-constructed prompt, the retrieved documents, the orchestrator's framing. Reconstructing it by concatenating preceding outputs yields a different quantity from what the receiver actually saw, and the entire conditional construct rests on that quantity being right. TraceElephant's authors demonstrate the gap directly by re-running a Who&When failure to restore the missing inputs, and report large attribution-accuracy degradation when inputs are removed.
- **The outcome is single-class.** All 184 Who&When instances are failures. A probe cannot be fitted against a constant label, so the "refit probes on a held-out portion of the logs" arm was not merely difficult, it was **undefined as written**. This is a property of the corpus, not of the method.
- **Using the annotation as Y is circular.** The obvious per-step label — "is this the decisive error step, per the human annotation" — trains the probe on the very label the localisation claim is evaluated against.

Separately, a benchmark now exists that fixes the first two: TraceElephant (ACL 2026, CC-BY-4.0) records `input_context` **and** `output_content` per step, ships roughly 380 executions of which about 220 are annotated failures, and includes runnable environments.

### Impact (had it not been caught)

RQ3a is the pre-planned fallback that can carry the dissertation alone if the pilot gates fail. Built as specified, it would have produced a CPVI computed against an approximated conditioning state, with a refit arm that could not be fitted at all, tested against a circular label. A null would have been uninterpretable — bad construct, or genuine absence of signal, with no way to tell. That is the worst possible state for a fallback, because it is the one that looks like a result.

### Risk reduced

The fallback is now verifiable rather than assumed, and the failure mode it was exposed to — an uninterpretable null on the arm that has to carry the dissertation if the headline fails — is closed before any loader was written. Maps to roadmap risks R1 (capability floor, whose branch is elevating RQ3a) and the new R9 (replay non-determinism).

### Correction path

Read the conditional construct back against what each corpus actually records → separate the three problems → evaluate four substrate strategies and four candidate definitions of Y on cost and on the probability each yields an *interpretable* rather than merely *reportable* result → adopt, and schedule a counts spike ahead of any chapter text.

### The fix

- **Substrate.** TraceElephant becomes primary. Who&When is retained as a **transfer-only comparability anchor** so the familiar published numbers appear in the same table, with every row it emits carrying an explicit `reconstructed_observation` flag so approximated and true conditioning state can never be silently pooled. MAST-Data stays as a cheap trace-level secondary; it cannot test step localisation.
- **Outcome.** Y is defined by **counterfactual replay** — re-run from step *t* with the step's output substituted, and record whether the outcome changes — with majority vote over *n* replays, a reported agreement rate, an agreement floor, stratified step sampling recorded in the manifest, a hard spend cap and a dry-run projection. Trace success is computed for every trace regardless, so the refit arm survives if replay is cut. The annotation is named in the methodology as *considered and rejected*, because that sentence does real work at viva.
- **Schema.** A separate `LogHandoffRecord` with its own version and **absent, not nullable** physics fields. `HandoffRecord` is the stable simulator contract and is not widened; a log record is a different contract. The cross-fit group key becomes `trace_id`, the exact analogue of `episode_id` — the leakage discipline is unchanged and only the name of the group changes.

### Result of the fix

The construct and its substrate now agree: the only corpus that records what the measure conditions on is the one the measure is estimated on. Defining Y by intervention also makes the external-validity claim and the causal claim rest on **one epistemology instead of two** — the whole thesis argues, after Lowe et al., that correlation between a message statistic and an outcome is not evidence the message mattered and that only intervention settles it, and replay applies that same intervention to the label rather than to the gate. The technique is precedented at scale (AgenTracer built a 2,000-trace corpus this way; Causal Agent Replay formalises step-level intervention as a structural causal model; Jaques et al. established the counterfactual logic for influence in MARL in 2019). What is novel is its use to define the *target of an information-theoretic boundary measure*.

### So what / takeaways

The substrate switch reads as a retreat only if it is presented as one. Presented correctly it is an argument in this dissertation's favour: it is the only work in this space that *requires* full observability, because it is the only one conditioning on it. Two honest caveats are carried forward and must be closed by the counts spike before either number reaches the thesis: the claim that MAST-Data contains a usable non-failure class rests on all-zero annotation rows in a dataset preview rather than a count, and the AgenTracer accuracies quoted in the review come from secondary summaries rather than the paper's own result table.

---

## 2026-08-23 — Probe validity hardened before the freeze: control tasks, repeated cross-fits, length control

- **Area:** the measurement construct — what the probe family is permitted to claim, and what gets frozen at the Phase-2 gate.
- **Status:** decided pre-freeze. DSE-043 and DSE-044 created and marked as blocking the Y/V freeze. No probe result exists yet, so nothing is re-run.

### Trigger

Auditing the estimator's existing checks against the standard objections in the probing literature, while deciding what the pre-registration document has to enumerate.

### Finding

The design carried exactly one null — the shuffled-message audit, which permutes messages within condition and tests whether the message carries signal. That answers one objection and was being treated as though it answered three.

- **It does not answer the capacity objection.** Hewitt and Liang (EMNLP 2019) show that probe accuracy cannot distinguish "the representation encodes this" from "the probe learned the task", and that the remedy is a **control task** whose labels are random by construction, with a reported **selectivity** gap. With 1,536-dimensional concatenated features against a few hundred pilot handoffs, a probe manufacturing apparent information from its own capacity is a live risk, and `auroc_train_cond` does not settle it — a probe can generalise and still be reading its capacity rather than the message.
- **It does not address per-instance noise.** A single cross-fit gives each handoff its score from exactly one fitted model, and at pilot N the *sign* of an individual score can flip under nothing more than probe reseeding. Those per-handoff scores are the mediator in H2 and the ground truth in all of RQ2, so that noise propagates into two of the four research questions unquantified.
- **It does not pre-empt the length confound.** "CPVI is just a fancy word count" is the first sceptical question the construct will attract, and C1 makes it *structural* rather than incidental, because C1 manipulates message length directly.

### Impact (had it not been caught)

All three would have surfaced after the freeze, when the honest responses are expensive. A control-task check added post hoc invites the question of why it was added; a capacity reduction chosen after seeing results is a researcher degree of freedom whatever its merits; and answering the length objection at viva rather than in the pre-registration converts a designed dissociation into a defensive rebuttal.

### Risk reduced

Closes new risks R8 (probe selectivity) and R5 (length confound), and quantifies the measurement error on the scores feeding H2 and RQ2. All three are cheap now and forbidden later — which is precisely the category of change the freeze exists to force forward.

### Correction path

Enumerate the objections separately rather than as one → map each to the null that actually answers it → confirm none of the three is already covered → ticket each as blocking the freeze rather than as a nice-to-have.

### The fix

- **Control task and selectivity (DSE-043).** Refit both probes against random labels stratified to the observed base rate; report `control_mean_cpvi` and `selectivity = mean_cpvi − control_mean_cpvi` alongside **every** CPVI figure, captions included. The pre-registered expectation is that control CPVI is approximately zero. Critically, the **capacity-reduction rule fires from a schedule written down before any pilot data is inspected**, so that the answer to "how did you choose the regularisation?" is something other than "it worked".
- **Repeated cross-fits (DSE-044).** Average per-instance scores over *R* cross-fits with different fold seeds and persist the across-repeat standard deviation per handoff as that score's measurement error. Logistic refits are milliseconds at this scale, so the repetition is nearly free; *R* is pre-registered.
- **Length control (DSE-044).** Pre-register a partial Spearman of CPVI with outcome given message token length, and length as a covariate on the mediation's second path, both reported alongside their uncontrolled versions.

### Result of the fix

Three objections, three distinct nulls, each stated in the methodology with its pre-registered expectation. The estimator now reports what it can and cannot claim rather than only what it found.

### So what / takeaways

The length control is the one that pays a dividend beyond defence. If CPVI were a proxy for length, then within C1 — where length is clamped to a narrow band — it would have little variance and no relationship to the outcome. If it continues to discriminate outcomes among messages of near-identical length, the construct and the confound have **dissociated**, and C1 flips from the cell that threatens the measure to the cell that demonstrates it. That inversion is only available if the analysis is pre-registered; run afterwards it is indistinguishable from a search for a favourable framing.

---

## 2026-08-23 — Contribution re-framed against a field that moved, and one arm cut on evidence

- **Area:** positioning and scope — what the dissertation claims relative to the current state of failure attribution, and which optional arm survives.
- **Status:** decided. DSE-047 created (docs); DSE-027 cut; DSE-046 created as its replacement.

### Trigger

Verifying the architecture review's citations before adopting them, and sweeping the same searches for anything the July literature review missed.

### Finding

Two things, one anticipated and one not.

**Anticipated: the headroom argument is stale.** The review positioned H5 against Who&When's 53.5% agent and 14.2% step accuracy. Both have been beaten — AgenTracer-8B reports roughly an 18% relative margin over frontier reasoning models, and TraceElephant's own agentic methods reach 66.7% agent and 33.3% step on full traces. The general claim that survives is narrower: *generic* frontier reasoning models remain below roughly 10% on step attribution.

**Not anticipated: the prospective framing is now contested.** The architecture review's proposed reframing was that CPVI is prospective where "every method above is a retrospective analysis of a completed failed trace with the outcome already known". That is no longer true as stated. **AgentForesight** (arXiv:2605.08715) reframes attribution as *online auditing*: at each step of an unfolding trajectory an auditor sees only the prefix and must continue or alarm at the earliest decisive error, explicitly to recover the intervention window that post-hoc attribution forfeits. It predates the July review and was missed by it. **Causal Agent Replay** (arXiv:2606.08275) was also missed and formalises step-level counterfactual intervention as a structural causal model.

Separately, the SocialJax arm was being carried on a scheduling justification that does not survive contact with the suite: SocialJax ships sequential social dilemmas in which agents interact through spatial actions and reward structure, and its shipped algorithms are not communication algorithms. **There is no message to score.**

### Impact (had it not been caught)

Writing the RQ3a chapter to 14.2% invites the single most predictable hostile viva question. Worse, claiming novelty on prospectivity alone would have been contestable by one citation, and contested at viva rather than in the text. And the SocialJax arm would have been cut late, on time grounds, producing a weak limitations paragraph in place of a strong one.

### Risk reduced

Closes R7 (stale baseline framing) and opens and immediately answers R10 (online-auditing adjacency) rather than leaving it for a reader to raise. Removes an arm that would have competed for the same GPU allocation as the headline sweep.

### Correction path

Verify each cited number and dataset independently rather than adopting the review wholesale → sweep for adjacent 2026 work in the same searches → where an adjacency exists, state it in the text on the axes where the contribution actually survives → re-derive the cut on the arm's own properties rather than on the calendar.

### The fix

- **Baselines re-anchored.** The 2025 figures are kept as a **dated floor**, with every method tabulated alongside its substrate and its date.
- **Contribution restated on three axes rather than one.** Against AgentForesight specifically: this is a **measure with units, not a detector with a verdict** (a log-loss difference between two logistic probes, against a 7B RL-trained model); it scores a **single boundary**, not a trajectory prefix; and it **blocks and rewrites the message before the receiver acts, validated against a matched-firing-rate control**, which an alarm does not do and which no online-auditing result has yet met. The comparison to report is the **operating characteristic** — localisation per unit of compute, and whether any is available before the outcome exists — not raw accuracy.
- **SocialJax cut on evidence.** The sentence for the limitations chapter is that the learned-message setting is a *different measurement regime*, not a harder version of the same one: IMAC and the emergent-communication line optimise a message encoder end to end and presuppose gradient access, whereas this dissertation scores arbitrary frozen natural language at inference with no access to the sender. A comparison between them would confound the training regime with the message space.
- **Replacement (DSE-046).** The Eccles et al. absent-versus-unused decomposition on RQ1's own handoffs: a two-by-two on CPVI against realised progress, split within condition, reporting the absent-signal and unused-signal rates with intervals.

### Result of the fix

Causal Agent Replay turns out to *strengthen* the position rather than threaten it — it makes replay-defined Y a well-precedented choice rather than an idiosyncratic one, and it does not do information theory, so the specific novelty (replay used to define the target of an information-theoretic boundary measure) is intact. The SocialJax replacement costs a day of analysis and no new compute, engages the same literature, and converts a possible RQ1 null into a reportable finding about which half of the channel failed.

### So what / takeaways

Adopting a research review without verifying it is the same error as trusting an experiment without a control. Two of the three most consequential findings here came from checking the review's own citations rather than from the review, and one of them changes a novelty claim. The general rule to carry forward: any positioning claim of the form "nobody does X" has a shelf life measured in weeks in this field, so it should be written on the axis that is *structurally* hard to occupy — here, intervention with matched controls — rather than on the axis that is merely currently unoccupied.

---

## 2026-07-25 — Difficulty ladder was partly infeasible; retuned pre-freeze

- **Area:** task geometry — difficulty ladder (slit widths) and the rotation action.
- **Status:** corrected pre-freeze (no results frozen, no datasets recorded; nothing to re-freeze).
- **Trigger:** building the P1-4 feasibility tool (an A\* search over macro-actions on the
  deterministic simulator) while preparing the pre-Myriad pilot tooling. Running it across the ladder
  immediately returned *easy solvable, medium/hard not*.

### Finding

The task asks a rigid **T-load** (a bar of length 1.4 perpendicular to a stem of length 1.0, both
0.3 thick) to pass a gap of height *w* in a thin internal wall, moving along +x. The design intent
(DSE-006) was *"the T must rotate to pass a narrow slit."* That intent is geometrically false, and
the search proved it:

- To cross a thin wall, at the instant each member of the T crosses the wall plane its vertical
  (y) slice must fit inside the gap. The bar and stem are **perpendicular**, so whichever member is
  aligned with travel crosses at its 0.3 thickness while the *other* member is then perpendicular to
  travel and presents its **full length** across the gap.
- Rotation only swaps which member jams. The tightest slit the T can thread is therefore the
  **shorter member's length — the stem, 1.0** — and it is **invariant to rotation**. A slit ≥ the
  full y-extent (bar thickness + stem = 1.3) clears head-on with no maneuver at all.
- The shipped ladder was **1.8 / 1.0 / 0.7**. So **easy (1.8)** was trivial, **medium (1.0)** sat at
  exactly zero clearance (stem length == slit), and **hard (0.7) was impossible** — below the 1.0
  threshold at every orientation. The A\* search confirmed it: easy solved in 7 pushes; medium only
  via a fragile ~17-step thread found at fine resolution; hard exhausted its reachable state space
  unsolved.
- Secondary finding: the rotation action was **uncontrollable** — one `ROT+` turned the T ~135°
  (angular impulse 2.0 against the T's small moment ~0.29), leaving only 45°-multiple orientations
  reachable, far too coarse to aim the T for a threading maneuver.

### Impact (had it not been caught)

`rq1_sweep` defaults its headline difficulty to **hard**, and the pilot runs easy + hard. On the
shipped geometry the pilot's **G1 (self-play solves the task)** and **G2 (a measurable C0→hard gap)**
would have failed on hard — *not because the models cannot coordinate, but because the task is
unsolvable.* That is precisely the "nobody can do it" misdiagnosis P1-4 was written to prevent: a
confusing null read as "no usable information transfer," triggering the fallback ladder for the wrong
reason and burning the (imminent) Myriad allocation on an uninterpretable result.

### Risk reduced

This is the concrete pay-off of the front-loaded pilot and of **early testing enabling a pivot during
architecture design**. Because the feasibility check was built and run *before* any recorded run
(CPU-only, no cluster, no LLM), a task-design flaw in the headline difficulty was caught and fixed
while it was still free to fix — no frozen result, no committed dataset, no re-run. Post-freeze or
post-Myriad, the same finding would have cost a re-specification of the difficulty axis and the
regeneration of every dependent result. (Maps to roadmap risks R1/R2: capability floor and
measurable-signal gates.)

### Correction path

Feasibility search flags infeasibility → first-principles slice geometry confirms it is real, not a
search artifact → candidate ladders and rotation magnitudes tested empirically against the search
until all three difficulties solve with a graded, sensible path length → budgets frozen from the
verified optima → wired per-difficulty.

### The fix

- **Slit ladder 1.8 / 1.0 / 0.7 → 1.8 / 1.2 / 1.1.** Easy unchanged (head-on trivial); medium 1.2
  and hard 1.1 both sit in the *threading* regime above the 1.0 stem threshold, graded by clearance
  (difficulty now increases via how precise the threading maneuver must be, a cleaner information
  gradient than the broken "rotate to pass").
- **Rotation `angular_impulse` 2.0 → 0.5** (~34° per action), so an agent can aim the T.
- **Per-difficulty step budgets** certified at ~2.5× the oracle optimum and wired through
  `SweepConfig` / `EpisodeRunner` / `rq1_sweep` (previously a single budget of 12 for all
  difficulties, which under-fed hard).

### Result of the fix

All three difficulties are now solvable with a clean gradient — **easy 7 steps, medium 13, hard 13**
(oracle optima); frozen budgets **18 / 33 / 33**. Rotation is controllable at ~34°/action. A
regression test re-derives and certifies the tight difficulties stay feasible within budget, so a
future physics change that breaks feasibility fails loudly rather than silently. The full suite,
type-check and lint stay green.

### So what / takeaways

Caught at the best possible moment: pre-freeze, pre-Myriad, at ~zero compute cost, exactly as the
front-loaded pilot was designed to enable. It converts the (now imminent, ~21 July) Myriad window
into headline sweeps rather than debugging, and it validates building the cheap CPU-side
feasibility/diagnostic tooling *before* the cluster arrives. The difficulty axis is now a defensible,
graded threading-clearance construct — worth stating plainly in the methodology, since a reviewer
would otherwise ask why "rotation" is the difficulty knob when rotation does not change the threading
cross-section.
