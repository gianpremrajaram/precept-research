# Pre-registration — CPVI at the LLM coordination boundary

**Status: v0 — DRAFT, NOT FROZEN.**
**Drafted:** 24 August 2026, during stage S1 (free local pilot), before any episode has been
recorded.
**Freezes as:** v1, committed the day the E3 gate verdict is `proceed` (freeze **F0** in
`docs/EXPERIMENTS.md` §6). Until that commit, every value below is revisable without penalty.

---

## 0. Why this document exists, and why its date matters

The forbidden researcher degree of freedom in this project is choosing the outcome *Y*, the probe
family *V*, or a gate threshold **after** seeing how the conditions came out. This file exists to
make that impossible to do quietly: it fixes those choices in a dated, version-controlled artefact
while the number of recorded episodes is **zero**.

That zero is the load-bearing fact. No result exists yet, so no choice recorded here can be
contingent on one. Everything in v0 is derived from the pilot design, the feasibility oracle and the
implemented defaults — not from any outcome. The same syllogism licenses the deviation register in
`docs/methodology.md` §10.5, and this document is its forward continuation: **the register closes at
F0, and every deviation after F0 is logged in §9 below, prospectively, with its reason.**

What v0 → v1 may change: values that the pilot is *designed* to reveal — chiefly the CPVI bit-scale
(§5), which cannot be honestly pre-specified before a single CPVI number exists, and the prompt
version (§2), which has a budgeted iteration allowance that closes at F0. Everything else should
survive unchanged; each change that does occur is listed in §9.

---

## 1. Hypotheses

| ID | Hypothesis | Test |
|---|---|---|
| H1 | Degrading the A→B channel degrades task outcome monotonically across C0 → C4. | Mixed model of per-handoff progress on condition, seed as random intercept, episode as a variance component; Holm-corrected condition contrasts. |
| H2 | The outcome effect of the channel is mediated by the conditional usable information in the message. | Episode-level Baron–Kenny on episode success with per-episode mean CPVI as mediator; indirect effect *a·b* per condition with a percentile bootstrap CI. |
| H3 | Apparent message value overstates conditional value: PVI > CPVI, and the gap widens as the shared state becomes more informative. | PVI − CPVI reported for every condition; never a message-value number without its state-only baseline. |
| H4 | A target-free runtime statistic computable at the handoff tracks CPVI closely enough to gate on. | E7 calibration against **realised outcomes only**; agreement with CPVI is a *reported correlation*, never a calibration target. |
| H5 | The measurement transfers off this task: handoffs that carry little conditional information localise failures in real multi-agent traces. | RQ3a — TraceElephant primary, transfer and refit regimes reported separately and never pooled. |
| H6 | Blocking low-information handoffs improves outcomes relative to matched controls. | E8 four-arm causal contrast (gate on / random-rate-matched / always-retry / off). |

---

## 2. The task, and what is fixed about it

- **Arena.** Three chambers, two internal walls, `chamber_w = 4.0`, `chamber_h = 6.0`,
  `wall_radius = 0.05`, slit centred at `slit_y = 3.0`. Goal at `(2.5 × chamber_w, slit_y)`.
- **Load.** One rigid T (bar + stem), y-extent 1.3, symmetric about the body so it centres on a slit.
- **Difficulty = slit width only.** easy 1.8 / medium 1.2 / hard 1.1. All three stay above the 1.0
  stem-length threading threshold; nothing else varies with difficulty.
- **Step budgets** (certified by the A\* oracle, ≈2.5× the optimum): easy 18, medium 33, hard 33.
  Oracle optima at the time of drafting: **easy 7, medium 13, hard 13** (E0, re-certified
  24 Aug 2026). Hard is therefore not starved relative to easy.
- **Macro-actions.** Seven, applied as a world-frame impulse at the COM with quasi-static settling
  (`linear_impulse = 3.0`, `angular_impulse = 0.5`, `dt = 1/60`, `substeps = 4`, `settle_steps = 30`).
- **Start-pose jitter** — the replication mechanism. `x ∈ [1.2, 2.8]`, `y ∈ [1.5, 4.5]`,
  `θ ∈ [−π/2, π/2]`, drawn from a seed-keyed RNG, so **different seeds are different problem
  instances, not identical greedy replays**.
- **Serialisation.** `numeric` for the headline sweep; `grid` and `nl` are the robustness axis.
- **Prompts.** `PROMPT_VERSION = v4` in force. Iteration allowance: **at most three version bumps in
  total**, all before E3, all logged; **two are spent** — v2 → v3 after the E1 transcript read showed
  A emitting 7 distinct messages across 75 handoffs, and v3 → v4 after E3-local showed A calling
  `com_y = 2.0074` aligned with a `(2.1, 3.9)` slit and pushing into the wall (both in
  `docs/experiment_design_log.md`, 2026-08-24). **One bump remains.** The version in force at F0 is
  the version of record.
- **Serialised state is part of the prompt surface**, so a serialiser change is a prompt bump. As of
  v3 the three forms are information-isomorphic: each exposes the load pose, the goal **and** the
  wall/slit geometry, differing only in surface form. v4 adds the load's own dimensions (`load_size`,
  bar 1.4 × height 1.3) to the numeric form: naming the gap without naming the object left "aligned
  with the slit" underdetermined. The dimensions are constants of the load, **not** a derived pass
  band — computing the threading band from them remains the agent's inference.

## 3. Conditions — the channel degrades one thing only

`apply_channel` touches the A→B message and nothing else; no condition may reach into physics state
or the action path. Any C0–C4 outcome difference is therefore attributable to the channel.

| Condition | Degradation | Parameter (v0) |
|---|---|---|
| C0 | pass-through | — |
| C1 | length cap | `c1_max_tokens = 8` whitespace tokens |
| C2 | one-step delay | B reads the previous step's message; step 0 reads `(no message yet)` |
| C3 | receiver-side window restriction | `c3_window_rows = 2` grid rows either side of the load |
| C4 | seeded token dropout | `c4_dropout = 0.4` per token |
| C5 | supervisor arm | **disabled**; optional, first to cut |

## 4. Outcome *Y*, horizon *k*, and conditioning semantics

**Horizon.** `k = 3` handoffs.

**The four labels, all computed post-episode from the full trajectory:**

| Label | Definition | Role |
|---|---|---|
| `y_binary_progress` | net geodesic progress over the next *k* steps is positive | **primary** *Y* |
| `y_continuous_displacement` | that same signed net progress, unthresholded | secondary |
| `y_discrete_config` | the chamber the load occupies at the **end** of the forward window | secondary |
| `y_terminal_success` | the episode reaches the goal at this or any later step | headline **episode-level** outcome (H2) |

`y_discrete_config` is deliberately the chamber at the *end* of the window, not at the handoff:
labelling the chamber at the handoff would make it a state feature `g_base` can predict from
`e_s` alone, collapsing its CPVI toward zero (toward, not to — usable information is
probe-relative; §6).

**Truncation.** `y_window_truncated` marks handoffs whose forward window was clamped at episode end.
**Pre-registered decision: truncated handoffs are retained in probe fits and reported with a
sensitivity check excluding them.** The flag is always visible, never silently dropped.

**Conditioning semantics — the one that decides what CPVI means.** The conditioning state *s* is the
state observable to the **receiver** at the handoff, i.e. `observation`, not A's full `state_str`.
The two coincide except under C3, where the restricted window is precisely the point: conditioning
on A's full view would hand `g_base` the global information the C3 message uniquely carries and
floor its CPVI near zero. Receiver-conditioning is applied at one choke point (`Featuriser.featurise`)
so every downstream consumer inherits it.

**CPVI is always the conditioned quantity.** CPVI = log₂ g_cond(y) − log₂ g_base(y), where `g_cond`
sees state **and** message and `g_base` sees state alone. **The PVI − CPVI gap is reported for every
condition**; a message-value number without its state-only baseline is not reportable.

## 5. Probe family *V* and its selection rule

- **Family.** ℓ₂-regularised logistic probes on frozen embeddings, `C = 1.0`, `max_iter = 1000`,
  homoscedastic variance model, probe seed 0.
- **Cross-fitting.** `n_splits = 5`. **R repeats: 5 repeated cross-fits** (DSE-044), with the
  across-repeat spread reported alongside the point estimate. Repeat 0 uses the canonical fold
  assignment and repeats 1…R−1 reshuffle the grouped folds under distinct seeds, so `n_repeats = 1`
  reproduces the unrepeated estimator exactly. The per-handoff across-repeat SD is persisted as
  `cpvi_sd`. R is fixed at 5 here and is **not** re-chosen after results are seen.
- **Selection rule.** *V* is fixed to the family above; the alternative (an MLP head, `mlp_hidden =
  64`) is a **pre-registered sensitivity check**, not a selection candidate. No probe family is
  chosen by looking at which yields a larger gradient.
- **Control task (DSE-043).** CPVI is re-estimated against **random labels** drawn i.i.d. at the
  observed base rate, through the same cross-fit splitter and the same probe family, so the
  comparison is like for like (Hewitt & Liang, EMNLP 2019). **Selectivity** = mean CPVI − mean
  control CPVI is reported alongside every CPVI summary.
  - *Directional prediction, fixed here:* control CPVI is expected to be **≤ 0**. Against random
    labels neither probe generalises out of fold, and `g_cond` carries twice the features, so it
    overfits the noise harder and scores *worse* held out. A negative control CPVI is the expected
    reading and is **not** a failure.
  - *Firing rule:* the capacity ladder fires when `mean_control_cpvi > 0.02` bits **or** its
    episode-cluster 95% interval excludes zero from above. 0.02 bits is fixed as under 10% of the
    smallest CPVI difference this design must resolve — the G2 gap, whose E3-local episode-cluster
    interval starts at +0.060 bits.
  - *Capacity ladder, in order; take the first rung that clears the rule:* (1) tighten the ℓ₂ probe
    from `C = 1.0` to `C = 0.1`; (2) reduce dimensionality with a within-fold PCA to 128 components;
    (3) fall back to the smaller encoder. Every rung is applied **before F0**, and to *both* probes
    — never to `g_cond` alone, which would manufacture the selectivity it is meant to test.
- **Length controls (DSE-044).** C1 shortens messages, so message length is confounded with
  condition by construction. Two controls are pre-registered and both are reported:
  - *(i) Length as a covariate.* Episode-mean delivered-message token length enters the H2 outcome
    model alongside condition and CPVI (`path_b_length_controlled`), and is partialled out of the
    CPVI-vs-progress rank correlation (`partial_spearman_length`).
  - *(ii) Overlap-restricted comparison.* Episodes are stratified into **3 equal-count quantile
    bins** of episode-mean delivered-message tokens; each Ck-vs-C0 difference — on success and on
    CPVI alike — is taken only within bins holding **at least 2 episodes of each condition**, then
    weighted by bin size. Bins are fixed before the data: quantile strata rather than a
    nearest-neighbour caliper, because at six episodes per condition a caliper has too little
    support and can collapse the comparison to one or two idiosyncratic pairs, whereas coarse
    strata degrade visibly. Where no bin holds both conditions the result reports
    `interpretable = false` and no delta rather than extrapolating one.
  - **Reported as a sensitivity analysis, not an adjusted effect.** Control (ii) is an
    *overlap-restricted, length-adjusted sensitivity analysis*; it is **not** a clean estimate of
    the channel effect with length removed. C1 shifts the length distribution by construction, so
    the overlap region is a non-random subset of both arms and the restricted contrast generalises
    only to lengths both arms reach. The unrestricted difference is reported beside every
    restricted one so the restriction's effect on the estimate is always visible.

**Encoder — pinned, frozen, computed once.**

| Role | Model | Revision |
|---|---|---|
| Primary | `BAAI/bge-base-en-v1.5` | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` |
| Sensitivity (DSE-022) | `sentence-transformers/all-mpnet-base-v2` | `e8c3b32edf5434bc2275fc9bab85f82640a19130` |

Embeddings are computed once, cached by content hash keyed on `(name, revision, text)`, and probes
fit on frozen vectors. An unpinned revision raises rather than warns, so it cannot reach a recorded
run.

## 6. Gates and thresholds

**Phase-1 go/no-go (E3), fixed here before the run:**

| Gate | Threshold (v0) |
|---|---|
| G1 capability | C0 self-play episode success ≥ **0.5** at easy difficulty |
| G2 signal | C0-minus-hardest success gap ≥ **0.1**, **and** the CPVI gradient points the same way (`g2_min_cpvi_gap = 0.0`, directional) |
| G3 groundedness | ≥ **0.8** of numeric mentions in messages match true geometry (abs tol 0.5, rel tol 0.05) |
| Minimum seeds for a `proceed` | **3** |

**The E3 cell is C0, C1, C3 and C4 crossed with easy and hard, over seeds 0–4** (40 episodes).
*Amended 2026-08-24, before the bf16 re-gate and before F0: v0 specified seeds 0–2 (24 episodes).
G1 then rests on three easy-C0 episodes, where a design whose true success rate is 0.67 fails the
≥ 0.5 threshold about a third of the time. The widening buys a five-episode read on the one gate
that has already failed once and stays inside the declared compute budget. The amendment changes
the cell, not any threshold.*
C3 is in the cell because it is the only condition carrying a genuine observation asymmetry and it
is in the headline design: a pilot that never exercises it certifies an instrument the main sweep
will not use.

**G2's CPVI estimator is fixed here.** The probe is fitted **once over the whole cell** and the
C0-minus-hardest contrast is taken between the resulting per-instance scores. Pointwise V-usable
information is defined as per-instance scores from one fitted probe; refitting per contrast is a
different estimator from the one the RQ1 analysis uses, and on the E3-local data it read the C0−C4
gap as +0.012 bits where the whole-cell fit reads +0.211.

**A note on what CPVI being conditional does and does not imply.** CPVI is *V-usable* information:
it asks what a message adds to what a **bounded probe family** can extract from the receiver's
state, not what it adds to what the receiver formally holds. A receiver that holds the sender's
state verbatim does **not** force CPVI to zero — E3-local measures **+0.192 bits** in C0, where
`observation` is byte-identical to `state_str` in every record. What the `PVI − CPVI` gap measures
is how much of the message's apparent value the baseline probe could already have extracted; that
gap is reported everywhere CPVI is, per the standing rule.

G2 reports a third state, **unassessable**, for the case that admits no verdict: every handoff
carrying the same progress label, so CPVI has nothing to predict. An unassessable gate never yields
`proceed`, and never spends the retune or invokes the fallback ladder, because an absence of data is
not evidence about the design.

**G2 is directional-only in v0 by design.** CPVI is in bits and its scale is unknown until the pilot
produces one. Pre-specifying a magnitude floor now would be a number invented rather than reasoned.
The 0.1 success gap carries the magnitude claim. **v1 replaces `g2_min_cpvi_gap = 0.0` with a
positive floor set from the pilot's observed bit-scale — this is the single change v0 anticipates,
and it is set from the pilot, never from the main sweep.**

**One retune.** The pilot is allowed exactly one retune (`attempt = 2`). A gate still failing after
it triggers the documented fallback ladder — elevate RQ3a to the headline — not a scramble.
**The retune ledger starts at the Myriad bf16 re-gate.** Local 4-bit results are indicative:
a local G1 pass is not the verdict of record, and a local G1 *failure* does not spend the retune
until it is reproduced at bf16.

**Runtime gate (RQ3b).** The gated statistic is computable **at the handoff** and is calibrated
against **realised outcomes only**. Calibrating it against CPVI is the circularity error and is
banned; `CosineStatistic` is probe-independent and exists to answer that objection directly.
Firing-rate budget **0.2**; calibration is flagged unreliable below **N = 200**. The threshold is
chosen on the calibration split before the causal arm is run, never after.
Retry feedback uses the DSE-045 template: under greedy decoding an unchanged re-prompt is a fixed
point, so a blocked handoff must be re-prompted with new content or the gate is vacuous. The
template is therefore **part of the treatment**, not an implementation detail: it is versioned
(`GATE_FEEDBACK_VERSION`, v1 at freeze) and recorded in every run manifest beside `PROMPT_VERSION`,
and it instructs A to state the push direction, the rotation decision and the goal direction
explicitly. **The alternative of raising the temperature on the retry is rejected here, before any
gate result exists**, on two grounds: it breaks the determinism story mid-episode — the run would be
greedy everywhere except at exactly the handoffs the gate touched — and it confounds the gate's
effect with an increase in sampling entropy, so a post-retry improvement could not be attributed to
the feedback rather than to having sampled a second time. H6's four arms must differ in one thing;
retries stay greedy and the content is what changes.

## 7. Design of the main sweep

- **Factorial.** condition (C0–C4) × serialisation × difficulty × seed; one episode per cell, seed
  carrying replication.
- **Scale.** **50 episodes per condition**, ≈250 episodes total, ≈6,000–12,000 model calls. Power
  basis: a gap of 0.4 needs 25–30 episodes/condition, 0.3 needs 45–50, 0.2 needs 90–100. The design
  is powered for a gap of ~0.3.
- **Replication is the seed axis, and only the seed axis.** The jitter is *drawn from* the seed (the
  RNG is keyed on `[seed, salt]`), so there is no separate jitter factor to cross: 50 episodes per
  condition means **50 seeds**, each a different start pose. Earlier planning notes described this as
  "10 jitter draws × 5 seeds"; that phrasing does not correspond to anything the grid can express and
  is corrected here before the freeze.
- **Substrate.** Headline data is generated at bf16 on Myriad. Local `local-lmstudio` data is
  permanently labelled in the manifest and **never pooled with cluster data**.

## 8. Analysis protocol

- **H1.** Linear-probability mixed model of per-handoff progress on condition; seed as random
  intercept, episode as a variance component within it. Holm correction across condition contrasts.
- **H2.** Episode-level mediation on `y_terminal_success` with per-episode mean CPVI: path *a*
  (condition → CPVI), paths *b* and *c′*, total *c*, and indirect *a·b* per condition with a
  percentile bootstrap CI (`n_boot_mediation = 400` refits over episodes).
- **Uncertainty.** `n_boot = 2000` for condition summaries; α = 0.05; **effect sizes and intervals
  are reported, never bare significance**; seed sensitivity is always reported. Intervals on
  per-handoff quantities (CPVI, PVI) are **episode-cluster** bootstrap — episodes resampled with
  replacement, their handoffs pooled — because handoffs within an episode share a trajectory and
  overlapping label windows; the iid handoff resample read roughly half the honest width on
  E3-local. Episode-level quantities (success) use the plain episode bootstrap.
- **Negative controls.** A shuffled-message audit (`n_shuffle = 20`): messages permuted **within
  condition**, decoupling each message from its handoff. Criterion: the real pooled mean CPVI must
  exceed every permutation's (a permutation test, p ≈ 1/(n_shuffle + 1)). The null is **not**
  expected to reach zero — v0 first said shuffling "must collapse CPVI", corrected here pre-freeze
  when E3-local showed the floor is structural: within-condition permutation preserves the
  condition-level signatures message *style* carries (an 8-token C1 message stays recognisably C1),
  and per-handoff progress base rates differ by condition (E3-local: C0 0.255, C1 0.735, C3 0.379,
  C4 0.575), so condition identity alone predicts progress. The null's height is the *identity*
  component of CPVI; the real-minus-null excess is the *per-handoff message content* (E3-local:
  real +0.078 bits against a null of +0.043 ± 0.006, max +0.057 over 20 permutations).
- **Probe selectivity and length.** Every CPVI summary carries `mean_control_cpvi` and
  `selectivity` (§5). Message length is confounded with condition by construction — C1 caps it — so
  the protocol also reports the **partial Spearman** of per-handoff CPVI with progress given
  delivered-message token length, and the episode-level mediation reports path *b* both uncontrolled
  and with episode-mean message length as a covariate.
- **Reproducibility claim.** Seed-pinned, revision-pinned, low-variance. **Exact reproducibility of
  LLM runs is never claimed.**

## 9. Deviation log (opens at F0)

Every deviation from v1 after the freeze is recorded here — prospectively, with its reason, before
the affected analysis is re-run. The pre-F0 register lives in `docs/methodology.md` §10.5 (D1–D18)
and closes when v1 is committed.

| # | Date | What changed | Why | Effect on results |
|---|---|---|---|---|
| — | — | *(no deviations; the register opens at F0)* | — | — |

---

*v0 drafted at zero recorded episodes. Freeze this file as v1 on the E3 `proceed` verdict, add the
freeze date to the header, and record the CPVI floor set for G2.*
