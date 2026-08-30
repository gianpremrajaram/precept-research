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

## 2026-08-30 (latest) — one calibration substrate, and a mechanism number that stops depending on its grid

- **Area:** the RQ3a transfer arm's calibration substrate (amendment **A5**, executed); the
  permutation null behind `rq1.action_agreement`; the instrument declaration for RQ3b (§8c).
- **Trigger:** an external review of the re-freeze branch. The re-freeze whose stated purpose was
  clearing a stale `git_dirty` had silently changed its `--transfer` input to the **A2 pilot**
  calibration (`8902072e1f47b6de`) — neither the substrate of record (`86ecbbdf35322dc3`) nor the one
  **A5** declares (`af50c7c12d65540f`). Three substrates were live at once and the committed
  `rq3a.json` disagreed with the `docs/EXPERIMENTS.md` row sitting beside it. Separately, the review
  noticed that `action_agreement` drew its whole permutation null from one RNG stream opened outside
  the condition loop.

**Finding 1 — a re-freeze that changes an input is a new result, not a provenance repair.** The two
are only distinguishable by diffing the output, which nobody did: TraceElephant's `cpvi_transfer`
agent accuracy moved 0.525 → 0.576 under the swapped calibration, restoring numbers that the 29
August session had already superseded. The lesson is procedural and cheap to bank: **a re-freeze
undertaken for provenance reasons must reproduce its predecessor's numbers bit-for-bit**, and if it
does not, the input moved and it is a result-affecting change owing a declaration.

**The fix.** Rather than restore the of-record numbers and leave **A5** pending — which would have
left two substrates live and forced a second re-freeze before R6 — **A5 is executed here**, so one
calibration (`af50c7c12d65540f`, the E3 re-gate corpus) backs the RQ3a transfer arm on the laptop and
R6's judge arm on the cluster. A5's selection rule was declared before these numbers existed and does
not read them: calibrate on the corpus with the most conditions and the widest realised-outcome
variance. The E3 corpus wins on both, and is measurably the better instrument (`fail` AUROC **0.906**
against 0.754; `cosine` **0.766** against 0.569).

**Result — A5 delivers a mixed reading, and the declaration obliges reporting it.**

| corpus | metric | confirmation (`86ecbbdf`) | **E3 re-gate (`af50c7c1`)** |
|---|---|---|---|
| TraceElephant | agent acc. | 0.525 [0.432, 0.610] | **0.492 [0.407, 0.585]** |
| | step acc. | 0.110 | **0.127** |
| | top-k acc. | 0.407 | **0.432** |
| | MRR | 0.315 | **0.331** |
| Who&When | agent acc. | 0.333 | 0.287 [0.220, 0.367] |
| | step acc. | 0.207 | 0.153 |
| | MRR | 0.415 | 0.329 |

**The headline survives and is stated at its new value.** On TraceElephant the transfer statistic
still beats both zero-call baselines with non-overlapping intervals — 0.492 [0.407, 0.585] against
schema validity 0.263 [0.186, 0.347] and mean cosine 0.254 [0.186, 0.339] — and it localises *more*
precisely by every rank-sensitive measure (step accuracy, top-k, MRR all up); the agent-level number
is the one that fell. On Who&When the arm remains a tie it does not win: mean cosine now leads
nominally (0.367 vs 0.287) on heavily overlapping intervals, where before it led 0.367 vs 0.333.
A5 said in advance that a worse transfer arm under the better calibration would be **reported, not
reverted**; this is that report.

**Finding 2 — a permutation null keyed to the whole dataset makes a per-condition number depend on
the grid it was computed on.** `action_agreement` opened `default_rng(seed)` once for the condition
loop, so each condition consumed whatever stream position the preceding conditions left. The same arm
scored on a four-condition grid and on a two-condition grid gets different nulls — and **A2's decision
rule reads R1's two-condition agreement limb directly against E3's four-condition C4 value**, so the
comparison the amendment turns on was the one the defect broke. Fixed by seeding inside the loop on
the condition's fixed index in `CONDITION_ORDER`, which is invariant to which conditions are present.

**Finding 3 — the same exposure showed the *p*-value was Monte-Carlo-limited, not data-limited.**
Re-deriving E3 under per-condition seeding moved C4's *p* 0.010 → 0.015, across the Bonferroni line
(0.05/4 = 0.0125). At `n_perm = 200` the estimator's standard error at that *p* is ≈ 0.005, so the
corrected claim was never resolvable at this permutation count in either direction. Raising the count
converges it: *p* = 0.0040 at 2,000 permutations and 0.0060 at 20,000, against 0.0149 at 200. The
default is now **2,000**, matching the `n_boot = 2,000` used elsewhere. **This is a precision change,
not a re-analysis:** the estimand is the exact permutation *p*, unchanged; only the Monte-Carlo error
in estimating it moves, and the converged value **supports the reading already on the record** rather
than rescuing one that had failed — which is precisely why it is safe to make. Agreement, rotation
and flip-rate figures are deterministic and did not move at all. E3's frozen artefacts
(`runs/af50c7c12d65540f-report/`) and its README are re-derived; the C4 row now reads *p* = 0.004.

**Risk reduced.** Three: an artefact set that contradicted its own changelog; a mechanism number that
would have been incomparable between E3 and the R1 arm about to be submitted against it; and a
headline significance claim resting on 200 draws. All three were reachable before the `qsub`, which
is the only place they were cheap.

**So what.** The transferable rule is the first one: *provenance repairs must be diffed*. A re-freeze
is the one operation whose success criterion is that nothing changed, so it is the one where changing
something is hardest to notice. The second is narrower and worth the same vigilance — when a
statistic is going to be compared across runs, every source of its variation that is not the data
(RNG keying, permutation count) has to be pinned before the first of those runs, not after.

---

## 2026-08-30 — the gradient is absent, the mechanism is named, and the register learns to bend without breaking

- **Area:** the E3 verdict of record and the close of the E3 ledger; the per-condition form of the
  correctness limb; what the pre-registration locks and what it may amend; the headline order.
- **Trigger:** the rung-2 re-gate returned. `af50c7c12d65540f` — 80 episodes, 3,419 handoffs, job
  238085 at git `10283b0` — read `fallback`, and §6's pre-registered directional prediction came out
  **half confirmed**: C1 collapsed to 0/20 exactly as predicted, and C4 inverted again at 14/20
  against C0's 9/20. Under the ladder's own terms that ends the arena track and makes rung 3 the
  finding. The question this entry settles is what "rung 3" now *says*, given that the run also
  produced the project's clearest positive result.

**Finding 1 — the absence has a mechanism, and only a per-condition instrument could see it.**
`g3_correctness` pools the cell and returns 0.242 against a null of 0.257: a clean fail carrying no
information about *which* condition failed. Run within each condition (`rq1.action_agreement`, 2,000
within-episode permutations each, seeded per condition), the picture inverts: **C4 is the only
condition whose receiver beats its own within-episode null** — agreement 0.399 against a 95th
percentile of 0.388, *p* = 0.004, surviving Bonferroni across four tests — while C0 sits at 0.330
against 0.331 (*p* = 0.069), C3 at 0.261 against 0.342, C1 at 0.046. Every supporting measure agrees: C4 turns the
right way 61.4 % of the time against C0's 52.8 % and C3's 43.0 %, oscillates least (flip rate 0.445
against 0.559 and 0.570), rotates least (381 against 436 and 563), and leaves least of its own signal
unused (0.154 against 0.268 and 0.323). **A channel cannot be degraded informatively when the
receiver was never reading the state.** That is why no C0→C4 gradient exists to measure, and it is a
substantive claim about when communication-value measurement is meaningful — not a failed experiment.

**Finding 2 — the measurement is vindicated on the corpus that failed the hypothesis.** CPVI
rank-orders realised success across all four conditions (C4 0.237 > C0 0.222 > C3 0.143 > C1 −0.023
against success 0.700 > 0.450 > 0.150 > 0.000). C1's CPVI interval spans zero, which is the estimator
correctly reporting that a 46-character message carries nothing. H2's indirect effects exclude zero
for C1 (−0.197 [−0.557, −0.081], 43.7 % mediated) and C3 (−0.058 [−0.234, −0.001]); path *b* is
+0.766 and +0.652 with length controlled. The RD-15 audit reads *p* = 0.00498 against a null max of
0.0047. Control-task CPVI is 0.0018. And RQ2, re-run here, flips H4 from the C0-only corpus's
+0.052 [−0.100, 0.201] to three statistics all excluding zero — `info` +0.275 [0.150, 0.384], `fail`
−0.275 [−0.379, −0.155], `cosine` +0.269 [0.147, 0.371] — with `fail` separating realised failure at
**AUROC 0.906**, up from 0.754. **The negative result and the positive result are the same instrument
read from two ends.** That sentence is now the spine of the write-up.

**Finding 3 — the length-matched control fired on its first real cell, and it reframes C4.** Within
the single overlapping delivered-length stratum, C4's success advantage goes from +0.250 to
**−0.071** and its CPVI delta from +0.030 to −0.069, on 13 of 40 episodes. C1's (−0.450 → −0.500)
and C3's (−0.300 → −0.277) deficits survive. On the pre-declared confound control **C4 is not better
than C0; it is shorter than C0** — so the success claim is withdrawn to a length effect while the
agreement and CPVI results, which are not length-matched, stand as open. This is the instrument
DSE-044 was built for doing the one job that matters: stopping a pretty number being over-read.

**Finding 4 — the condition ordinal is not the severity ordinal, and that is a result.** C1 delivers
46 characters, C4 delivers 221, C0 delivers 373. C1 is by far the harsher manipulation while sitting
lower on the nominal ladder. G2 contrasts C0 against C4 because §6 declares the ladder C0 → C4 and
states the C0−C4 contrast in its own worked example, so the gate is read as declared and the verdict
stands — but against C1 the same gate would read success +0.450 and CPVI +0.245. **Recorded as a
diagnostic and never as a re-read**: swapping the contrast after seeing which one passes is precisely
the forbidden move, and it is named here so nobody is tempted later. A degradation axis that is not
monotone in its own nominal severity is the kind of thing the measurement literature assumes away.

**Finding 5 — a register that forbids everything forbids following its own evidence.** The
pre-registration had been read as uniformly prohibitive, which would have blocked arbitrating a
mechanism the instrument itself uncovered. That is not integrity; it is paralysis, and it has a cost
in unexploited evidence. **§8a now states the boundary explicitly**: immutable are evaluated
thresholds (including "hardest" = C4), *Y* and *V* for confirmatory contrasts, the spent attempt
ledger, and the circularity guard; amendable are headline order, cells that arbitrate a mechanism
rather than rescue a hypothesis, secondary analyses on recorded data, and framing. The protocol has
four parts and all four bind: declared in writing here and dated **before** the run it governs; its
decision rule — including how it could disappoint — stated in advance; labelled post-hoc wherever
reported and never pooled into a gate, a contrast table or a family-wise correction; recorded twice,
in §8b and here.

- **Impact.** F0 will not fire and E4 will not run. RQ1 is written up as rung 3 with a named
  mechanism rather than an unexplained null. **A1** re-orders the headline to lead with the
  measurement primitive and its mechanism, with RQ3a as the external-validity chapter — the ladder
  pre-committed which results *survive* a failed re-gate, never their billing. **A2** declares one
  post-hoc arm and **A3** declares RQ3b's direction in advance (§8c), so a predicted null becomes
  evidence rather than an absence.
- **Risk reduced.** Three. A pretty inversion being written up as a compression effect when the
  pre-declared control says it is a length effect. A null being reported without a cause, which a
  reviewer reads as a broken instrument rather than a finding about receivers. And a register so
  rigid that the honest move — testing the mechanism you found — becomes indistinguishable from the
  dishonest one, with no protocol to separate them.
- **Correction path.** A2 arbitrates C4's two readings by holding length at C4's measured median
  (40 whitespace tokens, 217 characters against C4's 220) and swapping which content survives: a
  prefix cap keeps every number and severs the directive, the exact mirror of dropout. A4 would
  complete the 2×2 with targeted numeric redaction and is blocked on a design decision, not compute.
- **The fix.** `rq1.action_agreement` and `ActionAgreement` added, emitting `action_agreement.csv`
  beside the analysis; deliberately a separate function from `pilot.g3_correctness`, which produced a
  verdict of record and must not be re-shaped after being read. `ChannelConfig` wired to the CLI
  (`--c1-max-tokens`, `--c3-window-rows`, `--c4-dropout`) — it had never been reachable from a shell,
  so every dataset to date ran at its defaults and A2 had no knob to turn. `channel` is inside
  `sweep_hash`, so a changed parameter keys its own dataset and cannot append into the run it is a
  control for.
- **Result.** The verdict is frozen at `runs/rq1/af50c7c12d65540f/` with its manifest, summary, gate
  report and full reading; the lineage row is closed at `gate-verdict`; §6 records the verdict against
  the prediction; §8a/§8b/§8c open the amendment register with three declarations and one proposal.
- **So-what.** The dissertation's claim is no longer "we looked for an information gradient and did
  not find one". It is: *you cannot measure the value of communication in a pair whose receiver does
  not act on the state — and CPVI is the instrument that detects exactly that.* The absent gradient
  is the evidence for the claim rather than a hole in it, and the same instrument that fails the
  hypothesis passes its own validity checks on the same corpus and transfers to two real failure
  corpora. That is one story, not three.

---

## 2026-08-29 — the gate learns to tell "grounded" from "right", and immediately says the pair is not right

- **Area:** G3's second limb (declared at F0, previously unimplemented), the permutation criterion
  behind it, the RD-15 audit's resolution, and the step budget the E3 verdict of record runs at.
- **Trigger:** the confirmation passed G1 at 0.500 with grounding at 0.9998, and neither number can
  distinguish this corpus from attempt 2's — which scored 0.999 grounding while reasoning wrongly.
  Submitting E3 without closing that would have bought a verdict nobody could interpret.

**Finding 1 — the reference policy was already in the repo, one function away.** `certify()` requires
a scripted *rotate onto the lattice point nearest flat, then push east* policy to solve 16/16
jittered starts at every shipped difficulty, or the task does not certify. That makes it a
**sufficient** reference plan for every pose, and its per-pose form is four lines. No A* per handoff,
no new physics: `alignment_rotations` was extracted from `scripted_policy_solves` so the two cannot
drift, and `oracle_action` wraps it.

**Finding 2 — the threshold must be a null, not a number, and the two failure modes prove it.**
A fixed floor would have been chosen after seeing a corpus, which is the forbidden move. The limb
instead thresholds on a **within-episode permutation null of B's own actions**: shuffling actions
inside an episode preserves that episode's action habits exactly and destroys only their link to the
pose. The two failures the limb exists to catch are *invariant* under that permutation — always-push-
east (projection blindness) and always-rotate (attempt 2's defect) score exactly the null by
construction — so neither can pass by being lucky about a base rate. The level is the null's
one-sided 95th percentile, **not** RD-15's beat-every-permutation criterion: at 200 permutations that
is *p* ≤ 0.005, an order of magnitude stricter than the hypothesis tests the gate exists to license.

**Finding 3 — on first application the limb fails the corpus that just passed G1, decisively.**

| read | value |
|---|---|
| oracle-action agreement | **0.285** |
| null 95th percentile (the gate) | 0.322 |
| null mean (state-blind rate) | 0.315 |
| permutation *p* | **1.000** |
| grounding limb, same corpus | 0.9998 |

The pair matches the certified plan **less** often than its own actions do when shuffled. Three
independent supporting reads say the same thing and do not depend on the limb's tie-breaking near
90°: only **4 of 60** episodes ever bring the load within 6° of alignment (0.65 % of 2,597 handoffs);
**rotation-direction agreement is 0.519** on 1,400 rotations, a coin flip at SE 0.013; and mean
|misalignment| runs 84.5° at the first handoff to 36.3° at the last, which is what a bounded random
walk does from a broadside start, not what convergence looks like. ROT+ and ROT- are issued 723 and
678 times — near-perfect balance, the signature of oscillation rather than correction.

**Finding 4 — the verdict of record carries an unlogged step-budget deviation.** The confirmation ran
`--max-steps 50` broadcast to all difficulties against certified budgets of 30/35/35. That was never
logged. It is logged now, in PREREGISTRATION §6, and E3 runs at 50 as well — so that one gate does
not describe two task parameterisations. Finding 3 is the independent reason the extra steps bought
nothing: more budget buys more random walk.

**Finding 5 — the RD-15 audit was quoting a floor as if it were a result.** At 20 permutations the
smallest expressible *p* is 1/21 = 0.0476, so "the observed CPVI exceeded all 20 permutations" — the
strongest statement the test can make — printed as a marginal pass. `n_shuffle` is now 200 in both
RQ1 and RQ2 (the null costs ~2 min on a 2.6k-handoff cell). In RQ2 the null is *averaged* rather than
thresholded, so there the gain is variance, not resolution: a noisy null attenuates every
shuffle-corrected correlation, which is the H4 headline.

- **Impact.** G3 can now separate *says what is true* from *concludes what is right*, and the first
  thing it says is that the v9 corpus does the first and not the second. The `proceed` path is
  blocked on evidence rather than on missing code. Both non-GPU blockers on E3 are closed — the
  length-matched contrast turned out to need nothing built, only checking: it is wired, stratifies on
  *delivered* length (the right covariate, since C1's cap acts on delivery), and populates at the E3
  cell's 20 episodes per condition.
- **Risk reduced.** The one this project has already been burned by: certifying a run healthy on a
  fidelity check while its reasoning is wrong. Attempt 2 shipped on exactly that, and the fallback
  ladder fired because of it. A second occurrence is now detectable *before* the GPU spend.
- **Correction path if the limb is wrong rather than the corpus.** The oracle is a sufficient policy,
  not a unique optimum — position never enters it, and near 90° both rotation directions are near
  ties, so a disagreement means "not the certified plan", not "wrong". Direction agreement at 0.519
  is the check that does not depend on either caveat, and it agrees. If E3 returns a real gradient
  while this limb still reads at chance, the limb is the thing to revisit, not the result.
- **The fix.** `sim/feasibility.oracle_action` + `alignment_rotations`; `pilot.g3_correctness` with
  the permutation null; the limb wired into `run_pilot`'s gate list beside the grounding limb, never
  replacing it; `n_shuffle` 20 → 200.
- **Result.** 570 tests pass. The limb reads FAIL on `86ecbbdf35322dc3` and its criterion, its
  reference policy and its level are all recorded in PREREGISTRATION before the E3 cell exists.
- **So-what.** This is a **prospective prediction about E3**, and that is the point of writing it
  down today. If C0→C4 comes back flat or sign-inverted, the mechanism is already on the record and
  named: B is not acting on the pose, so degrading the message about the pose cannot change much. If
  the gradient shows up anyway, then message content is doing work through some channel other than
  the rotation decision — which would be a more interesting result than a clean pass, and one this
  limb is what makes visible.

---

## 2026-08-29 (earlier) — G1 passes by nothing, the measurement primitive replicates, and the transfer arm gets worse when its calibration gets better

- **Area:** the E3 gate (G1 verdict of record, G2's assessability, G3's missing limb), the RQ2
  measurement-primitive claim, and the RQ3a transfer arm's calibration source.
- **Trigger:** the G1 confirmation read out (job 236653, `86ecbbdf35322dc3`, 60 episodes, 2,597
  handoffs, 37 min on one A100). Every downstream question was waiting on it.

**Finding 1 — G1 passes at exactly the threshold, and the two live statements of that threshold
disagreed.** Easy C0 came in at **10/20 = 0.500** against a pre-registered *≥ 0.5*: a pass on `>=`,
Wilson 95% **[0.299, 0.701]**. A design whose true rate is 0.5 passes this gate about half the time,
so it is reported as a pass on the letter of the declaration, never as demonstrated capability.
Separately, `docs/myriad.md` §9a had paraphrased the gate as *easy ≥ 8/20 and medium ≥ 3/20* — 0.40
easy plus a medium clause that the register does not contain. Both readings are satisfied here
(easy 10, medium 3), so the verdict is not in dispute; at easy 8/20 or 9/20 the project would have
had two documents disagreeing about a gate outcome, with the more permissive one in the runbook the
submitter actually reads. §9a now quotes PREREGISTRATION §6 instead of paraphrasing it.

**Finding 2 — the measurement primitive replicates on disjoint seeds while the outcome does not.**
Seeds 12–31 never informed anything about seeds 0–11's estimates, and yet:

| quantity | A2 pilot (24 ep, seeds 0–11) | confirmation (60 ep, seeds 12–31) |
|---|---|---|
| mean CPVI | 0.1876 [0.115, 0.249] | **0.1853 [0.152, 0.217]** |
| mean PVI | 0.2728 | 0.2695 |
| **PVI − CPVI gap** | 0.0851 | **0.0842** |
| control CPVI | −0.0024 | −0.0016 |
| selectivity | 0.1900 | 0.1869 |

Every line agrees to about 1%, with the interval roughly halving. Over the same change the matched
outcome rate moved 0.417 → 0.325 (easy 0.583 → 0.500, medium 0.250 → 0.150). **The outcome is
seed-sensitive; the measurement is not.** That is the property a measurement primitive has to have,
and it is now measured on held-out seeds rather than assumed. The shuffled-message audit puts the
observed CPVI above **all** permutations (null mean −0.0010, max 0.0020, ***p* = 0.00498** — the
1/201 floor, not a marginal result; the audit was regenerated at 200 permutations later the same
day, see the entry above, and at 20 it could only express this as *p* = 0.0476), and selectivity
0.187 against a control task at −0.002 carries the Hewitt–Liang separation.

**Finding 3 — G2 is unassessed and G3's declared second limb does not exist.** The confirmation is
C0 only, so there is no condition contrast: `contrasts` is empty, the mixed model refuses to fit
("only condition C0 is present, so C(condition) is rank-deficient"), and G2 is *unassessable* rather
than failed — it spends no retune. More seriously, G3's grounding limb scored **0.9998** here, and
that number is worth almost nothing on its own: attempt 2 scored 0.999 on a corpus whose modal
*inference* was wrong, which is exactly why PREREGISTRATION declares a second limb — agreement with
the oracle's next action — due at F0. It is declared and **not implemented**; `pilot.py` carries the
grounding limb only, while an oracle already exists in `sim/feasibility.py`. So the check that would
tell us whether the v9 clearance line fixed what it was built to fix is the one not built. This is
the fourth instance of the reachability pattern in this project, and the first where the unreachable
thing is a *gate limb* rather than a helper. *(Closed the same day — see the entry above, which is
where the limb's first reading lives; this paragraph stands as written at the time.)*

**Finding 4 — a better-calibrated statistic transferred slightly worse, and the pre-declaration is
what settles which one is frozen.** Re-keying the RQ3a transfer arm to the confirmation was declared
unconditionally in `runs/rq1/8902072e1f47b6de/README.md` before this run existed. At home the
re-keyed statistic is unambiguously better: `fail` AUROC **0.593 → 0.754**, episode-cluster CI
[0.444, 0.737] → **[0.638, 0.870]** with no resample at or below 0.5, and `ece_reliable` true for the
first time (n = 2,597). On the log corpora it is not:

| corpus | metric | A2 calibration | confirmation calibration |
|---|---|---|---|
| TraceElephant | agent acc. | 0.576 [0.492, 0.661] | **0.525 [0.432, 0.610]** |
| | step acc. | 0.169 | 0.110 |
| | MRR | 0.376 | 0.315 |
| Who&When | agent acc. | 0.367 | 0.333 |
| | step acc. | 0.180 | **0.207** |
| | MRR | 0.381 | **0.415** |

The frozen result is now the confirmation calibration, because the re-key was declared before the
numbers existed and adopting the weaker headline is what the declaration obliges. Two readings, both
kept: **(a) the headline claim is robust** — `cpvi_transfer` beats both surface baselines on
TraceElephant agent accuracy under *two independent calibrations*, with non-overlapping intervals
both times against 0.263 and 0.254, which is a stronger statement than the single-calibration
version; **(b) home AUROC does not predict transfer quality.** Last session's argument for why a weak
home AUROC did not invalidate the transfer result was that the two are different estimands. That
argument now cuts the other way and is reported doing so.

**Finding 5 — H4 is falsified in its literal form, and the gate works anyway.** `preceptx-rq2` had
never been run on 60 episodes; it is offline and free, so it was. H4 states that *a target-free
runtime statistic tracks CPVI closely enough to gate on*. On this dataset the tracking is absent:

| statistic | Spearman vs CPVI (shuffle-corrected) | AUROC for low CPVI | AUROC for realised failure |
|---|---|---|---|
| `fail` | **+0.052 [−0.100, 0.201]** | 0.355 | **0.754** |
| `info` | −0.052 [−0.201, 0.100] | 0.645 | 0.246 |
| `cosine` | −0.007 [−0.123, 0.124] | 0.602 | 0.431 |

*(Correlations regenerated at 200 permutations later the same day — see the entry above. At 20 the
reading was +0.025 [−0.113, 0.155] / −0.025 / −0.007; every interval still spans zero and the
AUROCs do not depend on the null, so the conclusion is unchanged.)*

`fail` predicts realised failure at 0.754 while correlating with CPVI at essentially zero, and
`info` is its exact rank-inverse (the DSE-061 identity, visible here as 0.246 = 1 − 0.754). So the
statistic that gates well is **not** a CPVI proxy, and the statistic that tracks CPVI does not
predict failure. The pre-registration anticipated this shape precisely — agreement with CPVI is *"a
reported correlation, never a calibration target"* — which is why the result is reportable rather
than fatal. What it changes is the **claim RQ3b is allowed to make**: the gate cannot be justified as
an approximation to CPVI. It must be justified as a statistic calibrated directly against realised
outcomes, with CPVI as the offline construct it is measured *beside*, not *by*. That is a narrower
and more defensible contribution, and it is the circularity guard (R5) paying for itself — a project
that had calibrated against CPVI would have manufactured a correlation here and never seen this.

**Finding 6 — the Phase-2 freeze now has its evidence, on the pre-declared rule.** Y is confirmed as
`y_binary_progress` by the selection rule declared in `experiments/rq2.py` before any RQ1 outcome was
read (admissibility, then encoder-invariance, then twin agreement, with corrected effect size only a
tie-break inside 0.05). Corrected CPVI by label: `y_binary_progress` 0.1869 [0.156, 0.218],
`y_continuous_displacement` 0.2447 [0.168, 0.321], `y_terminal_success` 0.0585 [0.027, 0.098],
`y_discrete_config` 0.1167 but **inadmissible** (minority share 0.027). The encoder-sensitivity check
ran: primary CPVI 0.1858 against the second encoder's 0.1519, ρ = 0.842, and
`label_ranking_invariant` is **true** — the ranking of candidate outcomes is not an artefact of
`bge-base-en-v1.5`. H3's twins rank-agree at Spearman **0.612** but agree loosely pointwise
(Bland–Altman bias 0.071, limits [−0.844, 0.986] against a mean CPVI of 0.19); the rank claim is the
one the data supports and the numeric-equivalence claim is not made.

**Impact.** G1 is settled and RQ1 is unblocked to the extent one gate can unblock it. RQ2's central
claim — that CPVI is a stable per-handoff quantity — has a held-out replication instead of a single
in-sample estimate. RQ3a's transfer arm rests on a statistic that is now demonstrably informative at
home, and on two calibrations rather than one.

**Risk reduced.** (a) A gate verdict that could have been contested between two internal documents is
now single-sourced to the register. (b) The transfer arm's biggest stated caveat — "the transferred
statistic is not established above chance at home" — is closed by measurement, not by argument.
(c) The RQ3a headline is no longer a one-calibration result.

**Correction paths considered and rejected.** *Keeping the A2 calibration because its TraceElephant
number is higher* — rejected: that is selecting a calibration set on the test outcome, the exact
researcher degree of freedom the pre-declaration exists to remove. *Reporting only the confirmation
and dropping the A2 row* — rejected: the comparison is the robustness evidence, and hiding a
sensitivity because it is unflattering is the same error in quieter form. *Treating G3 0.9998 as a
pass* — rejected on the register's own reasoning; the limb is missing, so the gate is reported as
grounding-limb only. *Re-running G1 on more seeds because 0.500 is uncomfortable* — rejected: the
gate was declared once and evaluated once, and re-reading it after seeing 0.500 is optional stopping.

**The fix.** `runs/rq1/86ecbbdf35322dc3/` is frozen with the manifest, summary, calibration and the
full reading. `docs/myriad.md` §9a is rewritten: G1 marked done, the paraphrase corrected to quote
the register, and the **E3 re-gate** (C0/C1/C3/C4 × easy/hard × seeds 0–9, `preceptx-pilot`'s own
cell) promoted to the next submission with both candidate hashes costed. The RQ3a artefacts are
re-frozen on `86ecbbdf35322dc3` and their manifests say so in `transfer_train_dataset_hash`.

**Result of the fix.** G1 PASS (0.500, [0.299, 0.701]); G2 unassessed pending its cell; G3
grounding-limb 0.9998, second limb outstanding. RQ3a re-frozen: TraceElephant `cpvi_transfer` agent
0.525 [0.432, 0.610] against schema validity 0.263 and mean cosine 0.254, at zero model calls.

**So what.**
1. **The gate held on unseen seeds, and only just.** Every number in this project that depends on
   G1 should carry [0.299, 0.701] with it. The honest framing is that the task is *marginally*
   solvable by this pair at easy difficulty, which is precisely enough for a gradient study and not
   enough for a capability claim.
2. **The strongest RQ2 evidence in the project arrived as a side effect.** A held-out replication of
   CPVI, PVI, the gap, the control and selectivity to within 1% is worth more to the measurement
   argument than any single-run point estimate, and it came free with a capability gate.
3. **The next GPU spend is the E3 re-gate, not the main sweep.** G2 is the only unassessed gate with
   a declared cell, and running the RQ1 condition sweep before it would be out of protocol. Two
   things must land first and neither needs a GPU: G3's oracle-agreement limb, and the length-matched
   contrast (CPVI's partial Spearman against message length is 0.439 here, and `length_matched` is
   empty only because one condition has nothing to match across).

---

## 2026-08-29 — RQ3a's headline method finally reported, and the refit arm reclassified from unrun to undefined

- **Area:** the RQ3a external-validity design (§9.8) — which methods the H5 comparison actually
  contains, what defines the refit regime's outcome, and what the published baselines are as of
  today. Also the arena→logs seam: nothing in the repo persisted a calibrated statistic.
- **Status:** transfer arm **reported and frozen** on both corpora; refit arm **undefined pending
  replay**, with counts; baselines **re-anchored** and verified against primary sources.

**Trigger.** RQ3a's frozen result (28 Aug) tabulated seven methods and carried numbers for two:
`schema_validity` and `mean_cosine`. Both CPVI regimes — the methods H5 is *about* — read
`unavailable`, and the judge replications had never run. The external-validity track, which the
fallback ladder designates as able to carry the dissertation alone, was reporting only its controls.

**Finding 1 — the transfer arm was blocked on an artefact nothing wrote.** `transfer_scores` needs a
persisted `Statistic` and the orientation its calibration measured. `save_statistic` and
`write_report` had shipped in DSE-017/DSE-018 and were called nowhere outside a unit test; no
calibration existed on disk. `RQ3aConfig.transfer_key` additionally defaulted to `"s_info"` — the
paper's notation, not a `Statistic.key`, and pointing at a statistic DSE-061 had retired. It could
never have loaded. Neither fault was visible because `transfer_dir` was always `None`, so the arm
short-circuited before either mattered. This is the same reachability failure as D27's unreachable
RQ3a driver and the illusory `statistic_key`: **code that exists, is tested, and is not reachable
from any entry point.** Third instance; it is now worth auditing for deliberately.

**Finding 2 — the refit arm's stated fallback is degenerate, and this was measurable.** DSE-042
provides that "trace-success (the cheap label) is computed for every trace regardless, so the refit
arm has a fallback if replay is cut". Counted on 29 Aug:

| corpus | traces | failed | succeeded | unlabelled |
|---|---|---|---|---|
| TraceElephant | 220 | 44 | **0** | 176 |
| Who&When | 184 | 184 | **0** | 0 |

Only TraceElephant's 44 SWE-bench traces carry a `tests_status`, and all 44 failed. The other 176
carry a `ground_truth`, but the corpus ships each trace **truncated at the mistake step** — the
final step's completion is a `tool_calls` continuation, not an answer — so ground-truth matching
cannot recover an outcome. D22 established the corpus is failure-only; this establishes the stronger
consequence: **no annotation-free outcome can be constructed from the shipped data at all**, so
counterfactual replay is not the preferred route to the refit arm, it is the only one. The reason
string said "DSE-042 has not been run on them", which reads as pending and understated this.

**Finding 3 — the baselines had moved twice more.** Verified against each paper's own abstract or
results table (`docs/rq3a_baselines.md`). AgenTracer's margin is "up to 18.18% over Gemini-2.5-Pro
and Claude-4-Sonnet", not the "roughly 18% relative" the review carried. Who&When Pro (Jul 2026)
reports **73.9%** step accuracy — but on 12,326 trajectories whose failures were *injected after
replaying a successful prefix*, so the decisive step is placed by construction. The same
all-at-once protocol scores 14.2% on Who&When and 73.9% on Who&When Pro — but on different models
fourteen months apart, and the later paper reports no original-Who&When row to match them on, so
this is *suggestive* of a substrate effect rather than a controlled demonstration of one. What makes
model capability an implausible sole explanation is that generic frontier models were still below
10% on Who&When step attribution in September 2025.
AgentForesight (May 2026) had already been treated in §7; what had not been stated is that
prospectivity *alone* is no longer a novelty claim.

**Impact.** Without the transfer arm, H5 was answerable only in the negative, and the fallback track
could not have carried the dissertation. Without the census, the write-up would have described a
pending run rather than a corpus property. Without the re-anchoring, the thesis would have been
measured against a floor beaten three times inside fifteen months.

**Risk reduced.** (a) The external-validity fallback now produces a result on its primary substrate
independent of the Myriad queue and of G1. (b) The refit arm's absence is now a *stated corpus
finding* rather than an unexplained gap in a table. (c) The baselines cannot be challenged in the
viva as superseded, and the contribution no longer rests on an axis a 2026 paper has occupied.

**Correction paths considered and rejected.** *Fitting the refit probe on the corpus annotation* —
rejected: the annotation is what localisation is scored against, so fitting on it is the exact
circularity this design exists to avoid. *Treating the 176 unlabelled traces as successes* —
rejected: it invents the class the arm is short of, which is the same error in the opposite
direction. *Waiting for the G1 confirmation before calibrating* — rejected: the plan is unchanged by
either gate outcome (below), so waiting bought nothing and cost the whole queue window.

**The fix.** `preceptx-calibrate` fits and persists the statistics (`fit_statistics` on the whole
set, distinct from `calibrate`'s discarded cross-fit folds, which supply the honest held-out
diagnostics). `preceptx-rq3a --transfer <dir>` wires the arm and **reads the orientation from the
report rather than accepting it as a flag** — a hand-entered sign would silently invert every number
in the table. The threshold is deliberately *not* transferred: it is an operating point on the
arena's score distribution, and the log corpora are a different distribution, so every metric stays
rank-based and the arm makes no pass/fail claim. `OutcomeCensus` records the counts above in the
result and the manifest. The baselines table is `docs/rq3a_baselines.md`.

**Result of the fix.** Statistic `fail`, calibrated on the v9 A2 pilot (`8902072e1f47b6de`, n=1013
handoffs over 24 episodes, held-out AUROC 0.593 with an episode-cluster 95% CI of [0.444, 0.737],
orientation +1), transferred unchanged:

| corpus | method | agent acc. | step acc. | MRR | model calls |
|---|---|---|---|---|---|
| TraceElephant (118 eval.) | schema validity | 0.263 | 0.017 | 0.244 | 0 |
| | mean cosine | 0.254 | 0.093 | 0.255 | 0 |
| | **cpvi_transfer** | **0.576 [0.492, 0.661]** | **0.169 [0.110, 0.246]** | **0.376** | **0** |
| Who&When (150 eval.) | schema validity | 0.327 | 0.007 | 0.225 | 0 |
| | **mean cosine** | 0.367 | **0.273** | **0.448** | 0 |
| | cpvi_transfer | 0.367 | 0.180 | 0.381 | 0 |

**The split between the two corpora is the finding, not a mixed result.** On TraceElephant the
transferred statistic beats both baselines on every metric with non-overlapping intervals on agent
accuracy. On Who&When it does not beat mean cosine. Who&When records agent *outputs* only, so its
observations are **reconstructed** from preceding messages — a statistic that conditions on the
observation has degraded conditioning to work with, while a statistic that only compares message to
observation (cosine) is unaffected by the same degradation. D14 moved the substrate to TraceElephant
on exactly this observability argument, made a priori; it now has a measurement behind it. The
`reconstructed_observation` flag that DSE-041 required on every Who&When row is what makes the two
rows safe to read side by side.

**The Who&When tie is an aggregation coincidence, not a rank identity.** Both `mean_cosine` and
`cpvi_transfer` score 0.367 agent accuracy there — the same 55 of 150 — which after DSE-061 retired
`InfoStatistic` for being rank-identical to `FailStatistic` is a pattern worth checking rather than
assuming. It is not one. The two methods share only **17** of those 55 traces; each is right on 38
the other is wrong on, and their within-trace Spearman correlation is 0.03. On TraceElephant they
are *anti*-correlated (within-trace ρ = −0.24, median −0.50), and the transfer arm's 68 agent hits
overlap the cosine baseline's 30 on only 10. So the transferred statistic is not a dressed-up
cosine: on the primary corpus it ranks against cosine and wins, and on Who&When the two are close to
orthogonal and happen to land on the same total. Post-hoc diagnostic on the frozen scores — it
changes no reported metric — computed as the Spearman correlation of the two methods' risk columns
within each trace, plus the overlap of their agent-accuracy hit sets.

**Why Who&When reports 182 traces against a 184-trace corpus.** Two traces
(`Algorithm-Generated/0bb3b44a…`, `Algorithm-Generated/71345b0a…`) run a single expert end to end and
contain no inter-agent handoff, so `handoffs_only` drops them before scoring. A boundary measure has
nothing to measure where there is no boundary; the manifest carries both counts (`counts.traces`
184, `metrics.rq3a.n_traces` 182) and they are now reconcilable from this note. Within the scored
set the accounting is closed in the artefact itself: `LocalisationMetrics` gained
`n_traces_not_evaluable`, so `scored = evaluated + off_boundary + not_evaluable` holds on every row
— on TraceElephant 220 = 118 + 100 + 2, the two being traces that carry no decisive-step annotation.

**What this does not yet claim.** The published 53.5% / 14.2% figures come from LLM-judge
procedures, and that arm has not run (`judge` and `agreement` are still `null`). No comparison to
the published methods is stated until it does. DSE-024 stays open on that basis.

**And the transferred statistic is not established as better than chance at home.** Its held-out
AUROC of 0.593 carries an episode-cluster 95% CI of [0.444, 0.737] over the pilot's 24 episodes
(2,000 resamples of episodes, the same clustering `calibrate` cross-fits on), so "weak but real"
overstates it: the interval straddles 0.5, and 10.8% of resamples land at or below it. That is a
statement about the *pilot*, not about the transfer arm, and the two are different estimands —
ranking a handful of handoffs within one trace against an annotation is a materially easier problem
than predicting prospective episode failure from a single handoff, and the transfer result is
measured with its own intervals on its own substrate. It is nonetheless the honest caveat on the
statistic being transferred: it was calibrated on 24 episodes, and the orientation `+1` that the
transfer arm multiplies by is chosen by the sign of that same underpowered AUROC. Re-keying to the
G1 confirmation (60 episodes, unseen seeds) is what tightens it, and it is one CLI re-run.

**So what.**
1. **The fallback track has a positive result.** RQ3a can now report that a statistic fitted on a
   physics simulator and applied *unchanged* to real multi-agent traces localises the responsible
   agent at 57.6% where the surface baselines reach 26%, at zero model calls. Probe transfer across
   substrates had no established result in the V-information literature; §9.8 framed a null there as
   the reportable outcome. It is not a null.
2. **The plan was invariant to G1, and that is its strongest property.** Calibration fits on
   handoffs; G1 counts episode successes against a threshold. Neither reads the other, so the
   confirmation dataset can serve both, and if G1 *fails*, the A2-calibrated statistic is what is
   kept anyway — this branch stops being the external-validity track and becomes the headline one.
3. **A corpus with a non-failure class now exists.** Long-Horizon Agent Trajectory Attribution (Aug
   2026, 1,300+ trajectories, 30.3% task-aligned) is the first tabulated corpus that is not
   failure-only. It is a live correction path for the refit arm that does not require building a
   replay harness, and is recorded here rather than acted on: changing substrate again is a design
   decision, not a fix.

---

## 2026-08-29 — G1 declared against a pilot that moved the failure, and the thinking arm that could never have run

- **Area:** the G1 capability gate (thresholds, seed allocation and consequence map), the RQ1
  confirmation design, the `<think>` contract in `serving/client.py`, and two single-condition
  estimands in `experiments/rq1.py` / `analysis/stats.py`. Prompt **v9** unchanged —
  this entry declares a gate over v9, it does not modify v9.
- **Status:** declared, pre-run. The confirmation sweep is the next submission. The v9 ablation
  (A1 `9f46e0e34fab81cf`, A2 `8902072e1f47b6de`) is hereby designated the **pilot**, not the gate.

**Trigger.** The three D26 ablation arms returned from Myriad on 29 Aug. A1 and A2 completed; A3
raised `ServingError` before its first episode. The v9 serialiser worked, which made the gate live
for the first time — and made the question "what would have counted as passing?" one that had to be
answered in writing before the next run rather than after it.

**Finding — four, in the order they matter.**

1. **The v9 clearance line is the mechanism, and it is a large effect.** Seed-matched against the
   frozen baseline `188a3d556b824e3e` (identical start poses at equal seed, verified), A2 gains
   **6 easy seeds and loses 0** (McNemar exact *p* = 0.031); A1 gains 4 and loses 0. Success goes
   1/12 → 5/12 → 7/12 easy and 0/12 → 2/12 → 3/12 medium. The mechanism reading is unambiguous:
   messages naming an extent go 13.7% → 100%, steps sitting in a pose that actually fits go
   **3.1% → 28.6%**, and contact-limited rotations — the trap the baseline diagnosis attributed to
   premature eastward pushes — collapse from **51.0% → 2.4%** of all rotations. That last number is
   the confirmation that the trap was downstream of the projection error, as the diagnosis
   predicted, and not an independent hazard needing its own fix.
2. **The step budget is not the mechanism, and must not be credited as one.** A1 and A2 differ in
   `max_steps` alone (30/35 vs 50) and are byte-identical in prompt. The isolated budget contrast
   is **not significant**: easy gained 3 lost 1 (*p* = 0.63), medium gained 3 lost 2 (*p* = 1.00).
   Cap 50 is retained for headroom and because the pilot that clears the gate must be the
   configuration the gate is declared over — not because it does work.
3. **Rerun instability at a fixed seed is ~12.5% of episodes — and it is not seed sensitivity.**
   Under deterministic decoding a seed solved at cap 30 cannot fail at cap 50. Easy seed 1 and
   medium seeds 0 and 8 did exactly that, and A1/A2's *medium* solved sets are **disjoint**
   ({0,8} vs {3,4,5}). A1-vs-A2 is therefore an accidental partial replication, and 3/24 episodes
   flipping is what it measures.

   **These are two different objects and the write-up must name them separately, because a viva
   examiner will.** *Rerun instability* is the same arena, re-run: batched-inference
   non-determinism, estimable only from a deliberate (or accidental) replication, ~12.5% here.
   *Across-seed sensitivity* is different arenas in one run: whether the draw makes episodes
   systematically easier or harder. On the pilot the second is **absent** — per-seed success counts
   give a binomial dispersion of 0.64 (A1) and 1.06 (A2), i.e. indistinguishable from coin-flipping
   at a shared rate. So the outcome variation is decoding noise, not arena difficulty, which is why
   12 seeds per rung cannot carry a gate decision at medium and why the gate is declared with a
   no-reruns clause below.
4. **The binding constraint has moved from perception to commitment.** Starts are near-perpendicular
   by construction (80.7–98.8°), so every episode needs ~4 (easy) / ~6 (medium) *consecutive*
   same-direction rotations. At medium the split is decisive: successes reverse direction 0.26 of
   the time with a longest committed run of 6.0 (exactly the requirement); failures reverse 0.55
   with a longest run of 3.0. A second, independent waste appears alongside it — starts are already
   y-aligned within tolerance (com_y 2.90–3.06 against a ±0.20 medium tolerance), yet failing medium
   episodes spend a median of **14** N/S moves and drift to com_y 3.95.

**Impact (had it not been caught).** Two distinct failures. Declaring the gate *after* the
confirmation run, on the seeds already inspected, would have made G1 unfalsifiable — the
alternative threshold considered below (≥25% easy, ≥1 medium success) passes at the pilot rates
with probability 0.996, which is a ceremony, not a gate. Separately, crediting the step budget for
the v9 recovery would have put the next ladder rung on a lever the data says is inert.

**Risk reduced.** A capability gate whose threshold was chosen in view of the data it is evaluated
on; and a v10 prompt iteration aimed at the wrong mechanism.

**Correction paths considered and rejected.**
- *Evaluate G1 on seeds 0–11.* Rejected. The threshold is being set with the pilot in view — which
  CLAUDE.md permits pre-freeze, and which this entry logs — so the evaluation set must be disjoint
  or the gate tests nothing. Contamination is a hazard to *decisions*, not to comparisons.
- *Re-run A1 and A2 on fresh seeds too.* Rejected. A1's question (does budget explain the gain?) is
  answered and its answer is no; re-running it would spend GPU time to re-answer it. The
  confirmation runs exactly one configuration: the one the gate is declared over.
- *Fix the direction thrash before the gate.* Rejected on two grounds. The confirmation must test
  the piloted configuration, not a fifth one; and the obvious fix — emitting a signed
  `rotations_to_clear` — trades the floor for a ceiling and thins the gradient from the top, the
  same objection that retired the `fits` boolean in D26 and again in the entry above. Reversal rate
  and longest-committed-run become **reported covariates**; the N/S waste gets identical treatment.
  If medium fails the gate with thrash as the measured cause, that is an evidence-based v10 trigger
  and a better paragraph than a quiet pre-gate patch.
- *Fold the thinking arm into the confirmation as a fourth arm.* Rejected. Thinking mode is a
  **capability** manipulation, not a channel condition; it belongs with the 32B arm in the
  robustness story, never inside the C0→C4 gradient. It also cannot meet the determinism standard —
  Qwen explicitly discourages greedy decoding in thinking mode (noted at `client.py`'s
  `chat_template_kwargs` default), and finding 3 measures 12.5% outcome flips *without* it.

**The fix — the declaration. Fixed here, before the run, and not revisable by its outcome.**

- **Configuration under test:** v9 / C0 / numeric / `Qwen3-14B` @ `40c069824f4251a91eefaf281ebe4c544efd3e18` / cap 50 — A2's configuration exactly.
- **Evaluation set:** seeds **12–31** (20 per rung), disjoint from the pilot's 0–11.
- **G1 passes iff `easy ≥ 8/20` AND `medium ≥ 3/20`.** Binomial joint pass probability, computed
  before the run: **0.881** if the pilot point estimates are the true rates (0.970 easy × 0.909
  medium), and **0.076** if the true rates sit at the pilot's lower CI bounds (0.32 easy, 0.089
  medium). The rejected alternative — ≥25% easy and ≥1 medium success — scores **0.996** at the
  pilot rates, which is a rubber stamp by arithmetic rather than by argument, and is the reason
  this check is written down rather than left to judgement. The spread 0.88 → 0.08 is the bar
  doing its job: it is comfortable if v9 is as good as the pilot suggests and unforgiving if the
  pilot was optimistic, which is exactly the discrimination a capability gate is for.
- **Hard is descriptive, not gated.** Its usable gap is 0.54, requiring the load within ~10° of
  flat and therefore the same ~6-long committed run medium failures already cannot produce. It is
  run for characterisation; a hard floor is **not** a G1 failure and may not be read back as one.
- **Consequence map, declared now.** *Both rungs pass* → G1 met; RQ1 stays headline and the C0→C4
  sweep is the next submission. *Easy passes, medium fails* → G1 met on the gated rung; the ladder
  proceeds with easy primary and medium descriptive, and — only if the medium failure is
  attributable to the measured thrash covariates — a declared v10 decision point opens. *Easy
  fails* → the fallback ladder fires as written, RQ3a takes the headline, and RQ1 ships as a
  mechanistic negative result with the three named failure modes.
- **No reruns.** With a 12.5% measured flip rate an easy result of 7–8/20 will invite a second look.
  The gate is the realised outcome of this run, once. A rerun is permissible only for an
  infrastructure failure that produces no episodes (A3's class), never for an unwelcome number.

**The fix — two analysis estimands that reported a reassuring number instead of refusing.** Both
were found by reading the reconstructed pilot reports rather than by a test, and both are the same
failure class as the `<think>` collision: a computation that is correct on the grid it was written
for and silently wrong on a grid nobody had run yet.

- **`seed_sensitivity` returned exactly 0.0 for every seed.** It is defined as the per-seed
  C0-minus-hardest gap, so on a single-condition grid `hardest is C0` and it computes C0 minus
  itself. Not a degenerate estimate — a *self-subtraction*, whose all-zero output reads as "the
  C0→C4 ordering is perfectly seed-stable", the most reassuring statement the field can make, at
  the moment no ordering exists. It now switches estimand: the per-seed **success rate**, labelled
  `metric="success_rate"` with the reason in the artefact, and a Pearson **binomial dispersion**
  index carrying the inference, because the spread of a rate over few episodes is wide from
  sampling alone. Dispersion returns `None` rather than 0.0 where the null is degenerate (one seed,
  or a pooled rate at 0 or 1) — emitting 0.0 there would rebuild the same false reassurance one
  level down.
- **The H1/H2 mixed models were fitted to a rank-deficient design.** Both regress on
  `C(condition)`; with one condition there is no contrast to estimate. statsmodels does not refuse
  — it emits `ConvergenceWarning`s and returns boundary coefficients, which land in the artefact
  looking like estimates. They are now refused, with the reason in `mediation_note`.

This is why the guards were worth writing before the confirmation and not after: **the confirmation
is itself a single-condition grid** (C0 across three difficulties), so it would have inherited both
defects at exactly the moment the numbers decide a gate. Re-running the two pilot analyses under the
guards produces no `ConvergenceWarning`s and a result the old code could not express: dispersion
**0.64** (A1) and **1.06** (A2), i.e. across-seed variation indistinguishable from binomial noise at
a shared rate — the arena draw does not make episodes systematically easier or harder, and the
outcome variation is decoding noise. That is the finding the all-zeros had been hiding.

**The fix — the A3 contract bug.** `experiments/cli.py` sets `chat_template_kwargs={"enable_thinking":
True}` under `--thinking`, while `serving/client.py` rejected **any** `<think>` unconditionally: the
flag could never have produced an episode. `chat()` now discriminates. Unsolicited, the trace still
raises — that guard protects every channel condition and is not weakened. Solicited, the trace is
**stripped** at the boundary rather than allowed through, because `apply_channel` degrades the
instruction and a C1 truncation applied to a reasoning dump would measure something else entirely.
A trace with no `</think>` raises rather than falling back to the raw string: at `max_tokens = 2048`
mid-trace truncation is a real outcome, and the fallback would have injected raw reasoning into the
channel as a silent contaminant.

**Result of the fix.** The gate is not yet read out. What is fixed is the rule, and it is fixed in
public before the data exists. Four unit tests pin the `<think>` contract: solicited-and-closed
strips to the message, unsolicited raises, unclosed raises, and closed-with-nothing-after raises.
A1 and A2 have had their **analyses** reconstructed offline (`preceptx-analyse`, no GPU, encoder
served from cache), and both reports come back keyed to the dataset hash the GPU job printed —
which is the evidence that splitting analysis off the GPU job (the change that stopped job 227886
holding an A100 for 2h37m running `statsmodels`) leaves dataset identity intact. Their **run**
manifests are a separate artefact and were not reconstructed: `run_grid` writes them to
`runs/<hash>-run/manifest.json` at sweep end, they exist on the cluster, and the 29 Aug results
bundle simply did not pull them. They must be rsynced before either run is frozen — a reconstructed
analysis carries the encoder revision and probe config, but only the run manifest carries the model
revision, the exact command and the serving environment, and CLAUDE.md counts a run without those
as not audit-usable.

The analysis also surfaced what the floored baseline could not: **CPVI is positive, selective and
above its own null on v9 data.** A2 gives CPVI 0.188 [0.115, 0.249] against PVI 0.273 — a
PVI − CPVI gap of 0.085, which is the share of apparent message value that was an echo of the shared
state — with control CPVI −0.002. On `188a3d556b824e3e` there was no outcome variance for a probe to
find. This is RQ2's measurement primitive working end to end, and it is a pilot observation, not a
frozen result.

**Report the interval, not the permutation p.** The shuffled-message null gives *p* = 0.048, which
is marginal and is bounded below by the permutation count: with `n_shuffle = 20` the smallest
attainable p is 1/21 = 0.048, so the test is *at its floor* and the value says "no permutation beat
the observed statistic", not "the effect is barely significant". The CI is the honest summary — it
clears zero comfortably and does not depend on the permutation budget. Raise `n_shuffle` before the
C0–C4 sweep if a p-value is wanted at all.

**So-what.** Two lessons, one methodological and one about instruments. The methodological one: a
gate declared after its data is not a weak gate, it is not a gate — and the tell is arithmetic, not
judgement. Computing P(pass) at the observed rates is a five-line check that separates a real
threshold (0.88) from a rubber stamp (≈ 1.00), and it should be run on every acceptance criterion
before it is written down. The instrument lesson is that a validity guard and a deliberate
manipulation of the same quantity will collide, silently, at whichever boundary was written first.
The `<think>` guard was correct when the only correct amount of thinking was none; the moment
`--thinking` was added, the guard became a contract violation that no test covered because no test
exercised the two together. Any flag that turns on a behaviour some other layer treats as a fault
needs a test asserting the pair, not each half.

---

## 2026-08-29 — The prompt's own pass rule was wrong by the width of the wall's lip

- **Area:** the state serialiser (`sim/serialise.py`), the arena's aperture (`sim/arena.py`), and
  the C3 restriction (`agents/channel.py`). Prompt **v9**, correcting **v8** before v8 ran.
- **Status:** fixed. No v8 dataset exists, so nothing is superseded and nothing re-freezes; the
  three planned D26 ablation arms re-key from v8 to v9 and the table above carries the new hashes.

**Trigger.** A review of this branch before the ablation was submitted, not a run. v8 was built to
remove an inference error the agents were making; the check was whether the number it hands them is
the number the physics uses.

**Finding — three defects, one of them decisive.**

1. **The prompt named an aperture the walls do not impose.** `build_arena` gives every wall segment
   `wall_radius = 0.05`, so each face stands proud of its authored edge and the free gap is
   `nominal − 2 × wall_radius`. v8 stated the load's span next to the **declared** slit width and
   told the agent "a slit narrower than this cannot admit it", so the rule the prompt invited was
   `extent_y ≤ slit_width` — looser than the truth by 0.1 of clearance at every rung. Measured on
   the simulator by pushing a slit-centred bar east across 0–60° in 0.5° steps, the largest angle
   that actually threads is **38.0 / 17.0 / 10.0°** at 1.20/0.80/0.64. `extent_y ≤ usable_gap`
   reproduces all three **exactly**; `extent_y ≤ slit_width` predicts 44.5 / 21.5 / 14.0°, so
   **15% / 21% / 29%** of the poses the prompt certified as passable jam in the channel. v8 would
   have replaced the agents' trigonometry error with a narrower one of the prompt's own making —
   and a failure under it would have been misread as the agents', since the arithmetic they were
   asked to do would have been done correctly. The design log already recorded the effective
   aperture (2026-08-27); v8 simply did not reconcile against it.
2. **The natural-language form never named an aperture at all.** `_nl` gives the slit's centre and
   the channel depth, never its width. v8 appended "a slit narrower than that cannot admit it" to a
   form that cannot say whether any slit is narrower — an invited comparison with one operand
   missing. The nl arm would have measured trigonometry it had no second number for, which is the
   exact non-isomorphism v8's own rationale said the change existed to avoid.
3. **C3 restricted the grid arm less than the other two.** The numeric whitelist (`load=`,
   `contact=`) and the nl first-sentence rule both drop the clearance, but `_window_grid`
   deliberately re-prepended the whole grid header, clearance included. C3 would then have been a
   materially weaker treatment in one serialisation than in the other two, confounding the
   condition contrast with the serialisation axis it is crossed against.

**Impact.** Unfixed, the D26 ablation would have measured how well agents recover from a state
description that is wrong about the task's binding constraint for up to 29% of poses at the rung
the run floored on — and A1-vs-baseline, the contrast that is supposed to isolate the serialiser,
would have carried that error as part of the treatment. The C3 defect would have propagated into
RQ1 proper, where the condition gradient is the headline.

**Risk reduced.** A headline RQ1 result whose C0→C3 gradient is partly a serialisation artefact,
and an ablation whose "the serialiser did not help" reading could not be separated from "the
serialiser told them the wrong thing".

**Correction paths considered and rejected.**
- *Restate `slit_y` as the free interval.* One interval instead of two, but the grid raster draws
  walls at the **nominal** edges, so the text and the picture would have disagreed in the form
  where they sit side by side.
- *Shrink `wall_radius` towards zero.* It is a collision-fidelity parameter, and moving it re-keys
  the simulation digest and invalidates every certificate and the frozen baseline with it.
- *Emit a `fits` boolean.* Rejected in the D26 entry above for the same reason it is rejected here:
  it trades the floor for a ceiling.
- *Keep the clearance visible under C3 in all three forms.* Symmetric, but B cannot use its own
  span without the gap, which C3 removes as layout — so it buys nothing and weakens the treatment.

**The fix.**
- `sim/arena.py` gains `usable_gap(slit_width, geometry) = slit_width − 2·wall_radius`, the single
  source of truth; `alignment_tolerance()` now derives from it rather than re-deriving the subtraction.
- `clearance_line` states **two** scalars in all three forms — `slit_clearance` and
  `load_extent_y` — and still no verdict: the comparison, the rotate-then-translate ordering, the
  y-alignment and the two-wall repetition remain the agent's inference. The nl form names both in
  prose, so it can perform the comparison for the first time.
- `_window_grid` re-prepends the **legend only**, selected by name so a header line added later is
  dropped from C3 until someone deliberately admits it — the fail-closed rule the numeric whitelist
  already used.
- `PROMPT_VERSION = "v9"`, which re-keys the three planned arms.

**Result of the fix.** Not yet measured; the ablation is still the next submission. What is
measured is the rule: a unit test pins `slit_clearance` to `usable_gap` at all three rungs and
asserts the declared width is *not* what the line offers as the bar to clear.

**So-what.** The generalisable lesson is about what "grounded" has to mean when a prompt is an
instrument. v8 was grounded in the ordinary sense — every number in it was read from the live
simulator, none was stale or invented — and was still wrong, because the *relation* it invited
between two true numbers was not the relation the physics enforces. Groundedness checks that
compare serialised values against simulator state (G3, as specified) would have passed it. The
check that catches it is different in kind: take the decision rule the prompt implies, run it
against the simulator, and confirm the boundary it predicts is the boundary that exists. That check
is cheap — it is the sweep above, a few seconds on a laptop — and it belongs beside G3 for any
state variable the agent is expected to *compare* rather than merely read.

---

## 2026-08-29 — H6's analysis plan, declared before the arms have run

- **Area:** the RQ3b causal-gate contrast (`experiments/rq3b.py`, DSE-025).
- **Status:** declared. No RQ3b arm has been run, so nothing here was chosen with data in view.

**Trigger.** The RQ3b driver landed, which makes the arms runnable and therefore makes every
remaining analysis choice a researcher degree of freedom the moment the first one runs. Fixing the
plan is worth a short entry precisely because it costs nothing now and cannot be done later.

**What is fixed.**
- **Unit of analysis:** the episode. The gate acts per handoff, but the intervention resolves at the
  episode - it either reaches the goal or does not - and handoffs inside one share a start pose and
  a trajectory.
- **Outcomes:** terminal success and steps per episode. Both, not one chosen afterwards.
- **The family:** six contrasts - gate-active against each of `matched_random`, `random_trigger`
  and `off`, on each outcome - corrected together under Holm as **one** family. Correcting
  per-outcome would enter six tests as two families of three and inflate the family-wise rate at
  exactly the point the claim is made.
- **The decision rule:** H6 is supported only if gate-active beats **both** score-blind controls on
  terminal success after correction. Beating `off` alone is not enough and never was: the active
  gate both blocks and buys the sender another turn, and `off` holds neither constant.
- **Uncertainty:** bootstrap deltas with percentile intervals and Cliff's delta on every contrast.
  No bare significance.

**Risk reduced.** The two failure modes this closes are choosing the outcome after seeing which one
moved, and quietly widening the family so a marginal contrast survives correction.

**The one thing that is *not* fixed here.** The gate threshold. It is imported from a persisted
`CalibrationReport` whose target is pinned to `realised_failure` and never CPVI - the R5
circularity guard - and `build_gate` re-fits the statistic on the **calibration** records rather
than on any arm's own episodes. A threshold re-derived from the arms would let the treatment choose
the operating point it is about to be judged at.

**So-what.** Two readings are pre-authorised, and they are different claims. A **null** - the gate
matching its score-blind controls - says the statistic does not localise the handoffs that decide
the episode, and is reported as a finding about the measurement. **Untestable** - every arm
returning identical outcomes - says the grid produced no variance to move, which after job 232980's
1/96 is the live risk, and is a statement about the task rather than about the gate. The driver
distinguishes them in `verdict` rather than leaving the distinction to the write-up.

**Amended the same day, before any arm ran.** "Untestable" was first written as *every arm
returning identical outcomes* and implemented as such, which is a condition that cannot occur: the
gated arms re-prompt where the ungated one does not, so their step counts differ on any real grid
and a wholly floored run would have been reported as "H6 NOT SUPPORTED" - a verdict about the gate
drawn from a grid that never moved. The declaration and the code now key untestability on the
**primary outcome alone**: terminal success constant across every episode of every arm. Nothing
else in the plan changes, and no data existed when this was corrected.

---

## 2026-08-29 — The agents could not see the quantity the task turns on, so they pushed into walls they could not pass

- **Area:** the state serialisation (`sim/serialise.py`, `sim/load.py`), the step budget
  (`sim/feasibility.py` `STEP_BUDGETS`), and decode-time reasoning as a dataset-identity key
  (`experiments/sweep.py`). Prompt surface **v7 → v8**.
- **Status:** shipped; the ablation that tests it is the next Myriad submission.

**Trigger.** Job 232980 — the first 96-episode C0 grid on the DSE-059–063 corrected task
(`188a3d556b824e3e`, prompt v7, budgets 30/35/35) — returned **1/96 successes** (easy 1, medium 0,
hard 0), against 6/96 on C0 of the superseded pre-correction task. The actuator correction was
sound and is not what failed: `ROTATION_STEP_DEG = 12.0` is realised exactly (max |Δθ| measured
11.99987°), and against a 1.4×0.3 bar the passing windows are ±40°/±21°/±14° for slits
1.2/0.8/0.64 — 6.7/3.5/2.3 quanta, matching the certificate's median 4/6/7 rotations. Seed 7 is the
existence proof: eight consecutive `ROT-` from 84.8° to 0.8°, then seven `E`, goal reached at step
28 of 30.

**Finding.** Three mechanisms, in causal order, measured over all 3,198 handoffs.

1. **Projection blindness.** The v4 numeric form gives pose plus `load_size=(1.4, 0.3)` and — by an
   explicit design decision recorded at `serialise.py` — leaves the pass band as "the agent's
   inference to make". It is not made. 99.9% of messages quote the angle; **6.6%** mention a sine,
   cosine or projection at all; 77.3% invoke "thickness". The modal error is in the dataset's very
   first message: it quotes `angle=1.7236` (98.8°, true vertical span 1.43) and then computes the
   span as **0.3**, the thickness, concluding "you can push the load rightward". **97.6% of all `E`
   actions (904 of 926) were issued at poses that could not fit the slit.** Only 17 of 96 episodes
   ever reached a threadable pose.
2. **A one-way trap, entered by mechanism 1.** Free rotation needs `com_x ≤ 2.55` (channel face
   3.25 less the bar's 0.70 half-length). The linear quantum is 1.034 and starts jitter in
   [1.2, 2.4], so one premature `E` lands the bar at [2.23, 3.43] — usually past the bound, where
   rotation is contact-limited to half a quantum (`ROT+` median |Δθ| 6.25° against the 12° cap).
   Median final `com_x` ≈ 2.8: parked against wall 1 with 3,148 of 3,198 handoffs in chamber 1.
   The only escape, `W`, was issued **twice in 96 episodes**.
3. **Budget exhausted by oscillation.** All 96 episodes hit their cap. `ROT+` 1173 against `ROT-`
   878 with a direction-reversal rate of 0.36–0.47; the longest *committed* same-direction run has
   median 6/6/5 (easy/medium/hard) against the certificate's required 4/6/7.

**Impact.** No outcome variance, so RQ1's gradient, RQ2's proxy tracking and RQ3b's calibration all
had nothing to bind to. It also mis-attributes the null: read without the mechanism, 1/96 reads as
"the channel does not matter" or "the model is too weak", when what the run actually shows is that
the observation never carried the quantity the task turns on.

**Risk reduced.** A C0–C4 sweep bought at ~5 A100-hours on a task whose control arm cannot move,
and a headline null that a reviewer could attribute to model capability with no way to rule it out.

**Correction paths considered and rejected.**
- *Re-tune the physics a third time.* The actuator is certified correct and the success trace
  proves the task solvable in budget; the defect is upstream of the physics.
- *Emit a boolean `fits`.* It reduces the task to "rotate until True, then push east" and trades
  the floor for a ceiling, thinning the C0→C4 gradient from the top instead of the bottom.
- *Scale to 32B.* The ceiling is representational, not parametric; a larger model reading the same
  under-determined state would be re-running the same experiment.
- *Shrink the linear quantum* to keep the bar out of the trap. It doubles the `E` count needed to
  cross ~8 units and blows the budget it was meant to protect.

**The fix.** Prompt **v8**, three parts.
- `sim/load.py` gains `extent_y(angle) = 1.4·|sin θ| + 0.3·|cos θ|` — the load's vertical span at
  its pose — `sim/arena.py` gains `usable_gap = slit_width − 2·wall_radius`, and `clearance_line`
  puts **both** in **all three** state forms. (The second half is the v9 correction below; v8
  shipped the span against the declared width and never ran.) It states two scalars, not a
  verdict: the slit comparison, the rotate-then-translate ordering, the y-alignment and the
  two-wall repetition all remain the agent's inference. This is the v4 argument one derivative up —
  v4 named the object's constants because naming the gap without the object was underdetermined,
  and DSE-058 made each wall a channel, so the pass-relevant quantity became the load's
  *projection*, which is state and not a constant. It goes in all three forms because withholding
  it from one would make that form measure trigonometry rather than representation, which is not
  the axis the serialisation A/B is for.
- `--max-steps` broadcasts one budget over the certified `STEP_BUDGETS`. The certificate bounds an
  *optimal* policy; every one of 96 episodes saturated it, and the single success finished at 28 of
  30, so the budget is a live constraint rather than a formality.
- `--thinking` enables the Qwen3 reasoning trace (and raises `max_tokens` 512 → 2048), carried on
  **`SweepConfig`** rather than only on `ServingConfig`. That placement is the load-bearing part:
  `dataset_hash_for` reads `SweepConfig`, so a serving-only flag would have let the thinking and
  non-thinking arms hash alike and the writer's resume path would have appended both into one
  directory — the same pooling failure `prompt_version` and the simulation digest exist to prevent.
  `False` is excluded from the hash payload, so no dataset recorded to date re-keys.

**Result of the fix.** Not yet measured — the ablation is the next submission, and this entry is
written before its result deliberately. Four arms, C0/numeric/14B/easy+medium/12 seeds, with the
v7 baseline free because it is already run:

| arm | prompt | budget | thinking | dataset hash | contrast |
|---|---|---|---|---|---|
| baseline | v7 | 30/35 | off | `188a3d556b824e3e` | job 232980, easy+medium seeds 0–11 |
| A1 | v9 | 30/35 | off | `9f46e0e34fab81cf` | vs baseline → the serialiser alone |
| A2 | v9 | 50 | off | `8902072e1f47b6de` | vs A1 → the budget alone |
| A3 | v9 | 50 | on | `9fe1823c20d33c75` | vs A2 → reasoning alone |

The baseline grid is a superset of the ablation grid (it ran three difficulties at 32 seeds), so
the contrast is drawn on matched cells rather than on aggregates.

**So-what.** Two takeaways worth carrying into the write-up whichever way the ablation lands.
First, **the ablation is a result, not overhead**: "which intervention lifts a two-agent
coordination task off the floor — giving the agent the derived clearance, or giving it reasoning
tokens?" is a finding about where LLM spatial coordination actually breaks, and the four arms
separate representation from capability cleanly. Second, if A3 clears where A2 does not, the honest
claim is that the projection is computable but not *reliably computed under greedy single-pass
decoding* — a claim about deployment conditions, not about the model's competence, and a different
sentence from "the model cannot do it".

---

## 2026-08-28 — The encoder returned different vectors for the same string, and every embedding computed on this laptop was suspect

- **Area:** the embedding featuriser (`measure/featuriser.py`, §5) — which is upstream of PVI,
  CPVI, the twin, all three runtime statistics, the gate calibration and G2. Not an RQ3a entry
  despite being found in one.
- **Status:** fixed (`EncoderConfig.device`, default `cpu`, with a regression test). The local
  embedding cache is purged; the RQ3a result frozen earlier the same day is re-run and superseded.

**Trigger.** Freezing the first RQ3a result, a re-run of the *identical* command on the *identical*
corpus moved a headline number: TraceElephant `mean_cosine` step accuracy read **0.093220** on the
first run and **0.101695** on the second — one trace in 118 flipping its argmin. The scoring path is
pure arithmetic over embeddings with no seed, no probe and no bootstrap in it, so a moved point
estimate meant the embeddings had moved. Two further warm-cache runs then reproduced 0.101695
exactly, which localised the difference to *cold run versus cached run* rather than to run-to-run
noise.

**Finding.** `sentence-transformers` auto-selects a backend and on Apple Silicon that is **MPS**,
which returns **substantively different vectors for the same input string** depending on which
batch it lands in. Measured on `torch 2.10.0` / `sentence-transformers 5.6.0` against the pinned
`BAAI/bge-base-en-v1.5@a5beb1e3`, one text repeated 64 times across a 32-wide batch boundary:

| device | min cosine to row 0 | rows below 0.999 | max elementwise deviation |
|---|---:|---:|---:|
| MPS | **0.542745** | 62 of 64 | 0.159 |
| CPU | 0.999999999999 | 0 of 64 | 1.8e-07 |

CPU's 1.8e-07 is ordinary float32 batch-order jitter and is the real floor. A **0.46 cosine gap is
not jitter — it is a different vector**, returned for a string the encoder had already been given.

Two things made it reachable rather than theoretical. The corpus is duplicate-heavy — TraceElephant's
2,488 handoff messages are only **1,166 unique strings**, one of them repeated **316** times and
another 287 — so duplicates straddle batch boundaries constantly. And `embed_texts` caches by content
hash, writing one vector per unique text: on a cold run the analysis sees up to 316 *different*
vectors for one string, and on every later run it sees the single cached one. That is exactly the
cold-versus-warm discrepancy that surfaced it.

**Impact.** Every embedding ever computed on this machine is suspect, including the 50,975 cached
vectors, which were a mix of MPS-poisoned and correct values. The blast radius is the whole
measurement stack, not RQ3a: the same `Featuriser` produces `e_s`/`e_m` for the CPVI estimator, the
retrospective/prospective twin, `CosineStatistic`, the gate calibration and the G2 pilot gate. Any
number any of those produced from a local analysis on this laptop is unreliable.

**What is *not* implicated, and what is unverified.** Job 227886's analysis ran on Myriad against an
A100, i.e. the **CUDA** backend, which is a different kernel path and is not covered by this
measurement either way. The honest position is that the cluster results are neither implicated nor
cleared: the check is one command and it has not been run there.

**Risk reduced.** The severe one — silent wrong vectors producing plausible-looking results. Nothing
crashed, no interval blew up, no status flipped to `unavailable`; the table simply contained a
different number, and only a gratuitous re-run caught it. A determinism check that runs only against
a warm cache, or only against a stub encoder, cannot see this class of fault at all.

**Correction path — three alternatives rejected.**

- **Deduplicate texts before encoding, *instead of* pinning the device.** Rejected in that form, and
  the distinction matters. Dedup makes the symptom disappear — each unique string encoded once, so
  there is no divergence left to observe — while leaving the encoder free to return wrong vectors
  for non-duplicate inputs whose batch composition differs between runs. That is hiding the fault.
  (Dedup was then adopted *as well*, for a different and much smaller fault; see below.)
- **Auto-select but exclude MPS.** Same effect, implicitly. A pinned value is recorded, greppable
  and overridable; an exclusion list is a rule someone has to know exists.
- **Accept it as float noise and move on.** 0.46 of cosine is not noise, and the number it moved was
  a headline one.

**The fix, in two parts.** First, `EncoderConfig.device`, defaulting to `"cpu"` — the only backend
measured deterministic here — passed explicitly to `SentenceTransformer`. A regression test encodes
one string across two batch widths and asserts min cosine above `1 - 1e-6`, a threshold far above
CPU's float floor and far below the MPS failure, so it separates noise from a different vector
without pinning an exact float.

Pinning the device removed the catastrophic divergence but **not the cold-versus-warm asymmetry
itself**, which turned out to have a second, far smaller cause that only became visible once the
first was gone: two fully warm runs agreed bit-for-bit, while a cold run and a warm run still
disagreed in MRR's fourth decimal (0.254741 against 0.254975). CPU's own 1.8e-07 batch-order jitter
is enough to reorder near-ties in the ranking, because a cold run encodes a repeated string once
*per occurrence* — each landing in a different batch slot — while the content-addressed cache stores
exactly one vector for it. So `embed_texts` now **deduplicates before encoding and fans back out**,
which makes cold and warm identical by construction on any backend, and is markedly cheaper besides.
Indexing the result by text rather than by position also closed a latent misalignment in the same
function: the previous `[v for v in vectors if v is not None]` would have returned *fewer rows than
inputs*, shifting every downstream pairing, had any slot gone unfilled.

**Result.** A cold run and a warm run of the same command now agree on every metric to the last bit
(verified by purging the cache, running, and re-running). TraceElephant `mean_cosine` reads step
accuracy **0.093220** and MRR **0.254975**; the superseded MPS readings were 0.093220/0.257228 cold
and 0.101695 warm — the same headline number by coincidence, from a different and untrustworthy set
of vectors. The entry below carries the corrected table.

**So-what / takeaways.**

1. **The encoder's device is a reproducibility parameter and belongs beside its revision.** It is
   pinned in `EncoderConfig` but is **not** in `AnalysisProvenance`, which records encoder name and
   revision only — so an artefact still cannot say which backend produced its vectors. Adding it is
   a schema change to a model embedded in `RQ1Result` and `CalibrationReport`, and is left as an
   explicit open decision rather than made silently here.
2. **Run the same check on Myriad before trusting a CUDA embedding.** One command, and it closes the
   only remaining unverified backend.
3. **Determinism checks must include a cold path.** Every existing test either injects a stub encoder
   or runs against a warm cache. Both are blind to this, and it was caught by an accident of
   re-running rather than by the suite. Note that the *second* fault only became findable once the
   first was fixed — a 0.46 cosine gap drowns a 1.8e-07 one — which is an argument for fixing the
   loudest fault and then re-measuring rather than assuming one cause per symptom.

---

## 2026-08-28 — RQ3a was fully built and entirely unreachable: an open-weight judge, a decoded abstention, and a fallback that no longer waits on a GPU

- **Area:** the RQ3a substrate and how its result is produced (`experiments/rq3a_run.py`,
  `experiments/cli.py`), the identity of a corpus that is fetched rather than generated, what a
  judge abstention means in the results table (§12), and the dissertation's dependency structure.
- **Status:** implemented (DSE-064–066). Touches no simulator, channel or arena code; the RQ1
  generation created by D26 is unaffected.

**Trigger.** Asked for a lateral piece that would unblock research finalisation without contending
with the imminent corrected-actuator run, an audit of the backlog found RQ3a in an unexpected state:
roughly 1,800 lines of loaders, scorers, both CPVI regimes, the MAST arm and the agreement audit —
all tested — with **no way to run any of it**. No console entry point, no `JudgeBackend` concrete
outside a test stub, no corpus on disk, and no result anywhere in the repo. The cause is on the
record rather than mysterious: DSE-031, which shipped the pilot, RQ1 and RQ2 entry points, lists
*"the gate and RQ3 drivers"* as out of scope. DSE-018 later picked up the gate driver. The RQ3 driver
was never re-ticketed.

**Findings.** Four, and the first is the one that changed the plan.

1. **RQ3a's dependence on compute was much smaller than assumed.** Only three of the seven scored
   methods — the Who&When procedure replications — cost a model call. `schema_validity`,
   `mean_cosine`, both CPVI regimes and the MAST secondary need embeddings and nothing else, and
   `analyse_rq3a` already degrades method-by-method with an explicit `status` and `reason` rather
   than dropping a row. An offline RQ3a run was therefore *already a supported mode*; nothing had
   ever asked for it.
2. **The judge has to be an open-weight replication, and that is a claim about the numbers.** Every
   model call in this project is local or on the Myriad allocation, so the three published
   procedures are re-implemented against the served tier. `JudgeIdentity` already anticipated this
   and carries the caveat; what was missing was the concrete that fills it in.
3. **An abstention and an outage are indistinguishable at the call site and mean opposite things.**
   The `JudgeBackend` contract makes `None` a first-class return meaning *the judge declined*. A
   naive concrete would catch the endpoint's exception and return `None`, which silently converts
   cluster downtime into judge behaviour and inflates a reported abstention rate with minutes of
   network trouble.
4. **A fetched public corpus has no dataset hash.** The simulator side keys every dataset on
   `dataset_hash_for`, which folds in the simulation fingerprint precisely so a geometry retune
   cannot resume into the dataset it replaced (D26). A HuggingFace corpus has no such handle, and a
   silently revised upload would change a frozen result without changing anything that identifies it.

**Impact.** The dissertation's dependency structure, not just its tooling. CLAUDE.md names RQ3a the
pre-planned fallback that **can carry the dissertation alone** if the Phase-1 gates fail — and a
fallback that cannot be run without the compute it is a fallback *for* is not a fallback. Making the
offline arms first-class decouples RQ3a's evidential value from both the Myriad queue and from
whether the corrected-actuator generation clears G1.

**Risk reduced.** Three, in order. (i) The fallback becomes exercisable *now*, so a G1 failure on the
new generation no longer arrives with RQ3a still at zero results. (ii) An abstention rate becomes a
reportable property of the judge rather than a number contaminated by infrastructure. (iii) A frozen
RQ3a result gains an identity that moves if its substrate does, which is the same discipline the
simulator side already has and the log side did not.

**Correction path — four alternatives rejected.**

- **Reuse `RunManifest` by fabricating an `ExperimentConfig`.** Smaller, and wrong in a way the repo
  has already ruled on: DSE-041 kept `LogHandoffRecord` separate from `HandoffRecord` rather than
  widening the simulator schema with nullable physics, and writing invented conditions and seeds into
  a manifest is the same weakening one layer up.
- **A separate `scripts/myriad/rq3a.sh`.** The serve/wait/trap machinery is identical for every
  driver; a second copy is a second thing to keep in step with `serve.sh`. The existing `DRIVER`
  switch took a six-line branch instead.
- **Catch `ServingError` and return `None`.** See finding 3. Abstention is instead made *decodable* —
  the schemas offer `{"step": -1}` and `{"answer": "unsure"}` as reachable answers — so a model that
  cannot tell says so, and a broken endpoint still fails loud.
- **Digest the download rather than the records.** A re-zip with identical contents is the same
  corpus and should keep its identity; a revised upload with the same filename is not and must not.
  The digest is taken over the loaded records, which is the exact surface the analysis sees.

**The fix.** `preceptx-rq3a` (DSE-064), with `run_rq3a` composing load → score → manifest;
`RQ3aManifest` on its own version counter recording corpus identity, counts, encoder revision, judge
identity and substrate; `corpus_digest` as the log substrate's `dataset_hash`; `VLLMJudge` (DSE-065)
over the existing guided-decoding client; and a `preceptx-rq3a` branch in `pilot.sh` plus an opt-in
corpus pull in `prefetch.sh` (DSE-066). The judge, the endpoint, the Hydra model block and the
substrate label are demanded **only under `--judge`**, so the offline path needs no cluster at all.

**Result.** RQ3a runs end to end on a laptop with the judge rows marked `unavailable` and their
reason, and on a GPU node with all seven methods. `--dry-run` prints the corpus counts and the
judge's projected cost — one call per trace for all-at-once, `ceil(log2 n)` for binary search, *n*
for step-by-step — with each term pinned by a test to some judge's real worst case, so the number
that must fit the wall clock is neither exceeded nor inflated. Measured on the fetched corpora:
**3,428 calls for TraceElephant, 4,380 for Who&When** — the first concrete figure the next
allocation window can be planned against. Deliberately *not* converted to a wall-clock estimate
here: the only measured throughput in the repo is a local 8B-4bit tier (`runs/bench/ladder.md`),
and these calls are prefill-dominated — a whole transcript in, a step index out — so a tok/s figure
taken from the episode loop would not transfer. Time the first fifty calls on the node instead.

The offline arms were then run on both corpora (28 Aug 2026, `git_sha 3824de60`, encoder
`BAAI/bge-base-en-v1.5@a5beb1e3` **on CPU** — see the entry above, which supersedes this table's
first reading — corpus digests `ab666509dc934108` and `0aa22b23ee5965c0`, MAST `e46786e3a27fc66b`).
The
DSE-041 counts reproduce exactly from a fresh fetch — 220/5,960/2,488, 184/4,092/3,505, and MAST's
1,642 traces at 405 non-failures — so §6 of the schema mapping is now a reproduced count and not a
single spike. Localisation, `handoffs_only`, step accuracy with a 95% interval:

| corpus | evaluated | `schema_validity` | `mean_cosine` |
|---|---:|---|---|
| TraceElephant | 118 of 220 | 1.7% (0.0–5.9) | **9.3% (5.1–16.1)** |
| Who&When | 150 of 184 | 0.7% (0.0–4.0) | **27.3% (20.7–34.7)** |

Three things follow, and the third is a caution. (i) The cheap embedding statistic beats the cheap
deployable check by a margin whose intervals barely touch on TraceElephant and do not touch at all
on Who&When — so the measurement is earning its keep against the baseline that exists to threaten
it. (ii) It does so at **zero model calls**, which is the cost axis DSE-047 wants the contribution
framed on. (iii) **This is not a like-for-like comparison with the published 14.2% step / 53.5%
agent figures** and must not be tabled as one: those are whole-trace, all-steps numbers from a
hosted frontier annotator, while these are handoff-only over the subset whose annotated step falls
on an inter-agent boundary — 118 of 220 traces on TraceElephant, 150 of 184 on Who&When. Agent
accuracy here is 25.4% and 36.7%, well under the published 53.5%, so whatever is being gained is
gained specifically at step resolution. Reconciling the protocols is DSE-047's job, not this
entry's.

**The cross-corpus comparison is confounded by subset composition and must not be read as a corpus
effect.** Who&When scores nearly three times TraceElephant's step accuracy *despite* its
observations being **reconstructed** rather than recorded — the opposite of the ordering that made
TraceElephant primary. Before that is interesting it has to survive the obvious alternative: the two
rows are computed on differently-selected subsets. `handoffs_only` keeps only traces whose annotated
step falls on an inter-agent boundary, and that is **118 of 220 (54%) on TraceElephant against 150 of
184 (82%) on Who&When**. TraceElephant's discarded 46% are not a random half — they are the traces
whose decisive step is an intra-agent tool turn, which is plausibly a different and harder
population. So the reversal may be real, or it may be that TraceElephant's surviving subset is the
harder one. Nothing here separates the two, and the entry claims no corpus effect.

This is the same defect as the open question about the `handoffs_only` default, promoted from a
tidiness concern to a live threat: the default does not merely trim the sample, it **selects** it,
differently per corpus, in the arm that carries the comparison. Whether inter-agent handoffs are the
right unit is a genuine design question — the conditional construct is defined at a handoff — but it
has to be answered on its merits and then the discarded population reported, not left as a silent
filter in front of a headline table.

`cpvi_transfer` and `cpvi_refit` are `unavailable` in both runs, each with its reason: no frozen
simulator statistic exists yet, and DSE-042's replay has not been run on these steps. That is the
honest state of the comparison and it is visible in the table rather than in a footnote. The MAST
trace-level arm reads **0.088 bits (0.070–0.107)** of category information, with CPVI reported
`not_applicable` because MAST publishes no observation/message split at all.

**So-what / takeaways.**

1. **The judge numbers are a replication and must be tabled as one.** They are not the published
   Who&When figures and are not comparable to them as if the same annotator had been used. DSE-047
   is the ticket that owes the baselines table its methods, substrates and dates.
2. **Abstention is a result, not a gap.** Because it is decoded rather than caught, the abstention
   rate per procedure is a property of the open-weight judge worth reporting — plausibly the most
   interesting thing the replication says about the cost axis the contribution now rests on.
3. **The only part of RQ3a that queues is the judge.** Everything else is embeddings and sklearn.
   That is worth knowing before the next allocation window is planned, and it is why the offline
   arms are frozen first rather than last.

---

## 2026-08-28 — The task was never the task: a masked orientation hold, a stale quantum, and a jitter the actuator could not correct

- **Area:** the action model (`sim/actions.py`), the difficulty ladder's semantics (§9.2), the
  feasibility certificate (`sim/feasibility.py`), the start-pose jitter (§9.3), what the agents are
  told about the actuator (`agents/prompts.py`), and the gate's statistic set (§11.4).
- **Status:** implemented (DSE-059–063). Creates a new task generation; run 227886 is unaffected and
  its D25 record stands.

**Trigger.** D25 attributed the RQ1 null to a rotation quantum that had been sized against a load
which no longer existed. That was true and insufficient. Asked to size the replacement impulse, two
independent attempts (mine, and a cross-model review) both produced a number that would have left
the hard rung unreachable — which prompted measuring the windows instead of deriving them, and the
measurement did not agree with either the code's account of itself or the register's.

**Findings.** Five, in increasing order of seriousness.

1. **The rotation quantum was stale** (D25's finding, confirmed): `angular_impulse = 0.5` was sized
   for the T's moment of inertia (0.2927) and inherited by the bar's (0.1708), so 34°/action became
   57.79°. Free rotation is *exactly* deterministic — standard deviation `0.000000` across 37 start
   angles at seven impulses — so the 49.1° ± 16.9° seen in the data is a **point mass at 57.79° with
   a 26% contact-truncated tail**, not noise. That distinction is what makes the rest a lattice
   problem with an exact answer.
2. **"Step < window" is the wrong sizing criterion.** The bar is symmetric under a half-turn and both
   rotate directions are available, so the set reachable in *k* actions is the lattice
   `θ₀ + m·step (mod 180°)`, and reachability is whether that **orbit enters** the window — which is
   not monotone in the step. A 9.25° step leaves hard unreachable where 11.67° reaches it. Both
   independent sizing attempts failed on exactly this.
3. **`_ANG_RES` was silently coupled to the quantum.** The planner's pose-dedup bucket was a bare
   18°, correct against a 57.8° step and wrong the moment the step fell below it: a bucket wider than
   the step collapses consecutive rotations into one search state, so the planner would prune the
   poses the threading manoeuvre needs. Fixing the impulse alone would have broken the certifier.
4. **The start-pose jitter posed a control problem the action set cannot express.** A flat bar clears
   the narrowest channel only if its centre sits within 0.12 units of the slit, but N/S moves it in a
   deterministic 1.034-unit quantum — so from a continuous `y_range` of (1.5, 4.5) the reachable set
   is a lattice that misses the target for most starts. **Geometry alone capped success at
   77/39/23%**, before either agent reasoned about anything. The certificate could not see it because
   A\* searches from the canonical pose, where `y` is exactly `slit_y`.
5. **`hold_orientation` masked contact rotation rather than preventing it — and this is the one that
   matters.** It restored the pre-action angle *after* the settle. From 30° on medium (geometric
   window ±17.2°) the load rotates itself to **0.48°** mid-action under contact torque, slips through
   the channel, and is written back as 30.00°. So the DSE-058 degeneracy the register records as
   closed was **hidden, not closed**; the realised apertures were far softer than certified (medium's
   true window was ±32.6°, not ±17.2°); and **the recorded angle is not the angle at which the load
   passed the gap**.

**Impact.** Finding 5 is the serious one, because it reaches the measurement and not only the task.
Every message that faithfully reported "angle 30°" was faithful to a state that was not the operative
one, so the G3 groundedness result — messages match the recorded state to 0.01 units — is now a
claim about *recorded* state only. It does not disturb the CPVI estimates, which condition on that
same recorded state throughout and are internally consistent, but "grounded" is entitled to mean less
than it appeared to. Findings 1–4 together mean the RQ1 grid could not have produced an information
gradient on any arm: hard was geometrically dead, and medium and hard alike were capped by a
positional lattice nobody had computed.

**Risk reduced.** The class of fault is one constant silently outliving the assumption that set it —
four instances of it here, three of them invisible to every test in the repo. The specific risk
closed is spending a second 8-hour GPU allocation on a grid that cannot express the effect it is
designed to measure.

**Correction paths considered and rejected.**

- *Keep hard at 0.50 and pick a step that fits.* Rejected. Of 25 round step angles from 8° to 20°,
  exactly three give full coverage at 0.50, and several leave the rung unreachable at any budget.
  That is survival by arithmetic coincidence, and a later 0.5° retune would kill the rung silently —
  the failure class being fixed, wearing a new hat.
- *Re-derive all three apertures from declared window/step ratios.* Rejected as elegance bought with
  the one thing the new generation preserves: easy and medium keep their apertures, so the ladder is
  comparable to 227886 on two of three rungs. The ratios the shipped values imply are documented
  instead.
- *Give N/S a finer impulse than E/W to fix finding 4.* Rejected: it adds a second actuator parameter
  and a second control problem, when the jitter's job is to make seeds genuine replications, not to
  test sub-quantum positioning. Scoping `y_range` to the alignment tolerance is the smaller change
  and leaves the manipulation the ladder actually grades untouched.
- *Re-run 227886 under the corrected physics and replace the result.* Rejected, as in D25. It is a
  different task; it gets a different `dataset_hash` and a separate entry.

**The fix.** DSE-059 makes the hold real (infinite moment for the duration of a non-rotate action —
reproduces the geometric window to 0.01° across seven apertures, against 20.03° and non-monotone for
restore-after); authors `ROTATION_STEP_DEG = 12.0` and *derives* the impulse from it; derives
`_ANG_RES` from the step and refuses to load if it is not below it; pins `collision_slop`; scopes
`y_range` to a derived alignment tolerance; and moves hard 0.50 → 0.64. DSE-060 refuses to certify a
path whose rotations are not free-space rotations. DSE-063 refuses to certify a task the scripted
rotate-then-push policy cannot solve from every jittered start.

**Result.** All three rungs certify with **plannable** paths — `ROT+ ×5, E ×7` / `ROT+ ×7, E ×8` /
`ROT+ ×7, E ×7` — every rotation free-space, expressible in one sentence of natural language. The
scripted policy solves **32/32** seeds at every rung (it solved 11/7/4 before). Budgets 30/35/35.

**Declared decisions, so they are on the page rather than in the diff.**

1. **Difficulty grades by rotation-count slack (±3 / ±1 / ±0), not by rotation count.** Medium and
   hard both need seven rotations; what separates them is that easy tolerates a miscount of three,
   medium one, and hard none. Separating them by *count* would require hard's window to exclude the
   lattice point medium's admits — which is precisely the knife-edge geometry this entry removes. The
   trade is deliberate. Note the previous ladder did not separate them by count either (10 and 10).
2. **Widening hard is not softening it.** Its slack stays ±0; only its dependence on an arithmetic
   coincidence between step and window goes away. Effective aperture 0.54 remains inside the
   rotation-required band [0.3, 1.4).
3. **The prompt now states the action quanta** ("about 1.03 units", "exactly 12 degrees").
   Deliberate: hard tolerates no miscount, so leaving the step implicit would make the rung a
   constant-*discovery* task and the capability arm would measure the wrong thing. It follows
   standard practice in embodied spatial-reasoning benchmarks — REM supplies "rotate right 15°"
   alongside the observation and still finds models collapse under full rotation, which is the
   capability this task is meant to stress. `PROMPT_VERSION` → v7.
4. **`InfoStatistic` is retired in favour of `FailStatistic`** (DSE-061). They rank handoffs
   identically — Spearman −1.000000, exactly — so the gate had two statistics, not three, and a
   result "holding for both" held once. `FailStatistic` survives because it is monotone in the
   quantity the gate is calibrated against. The key `"info"` still resolves, so 227886's manifests
   replay.
5. **The 32B arm stays deferred, and the reason is unchanged by it being cheap.** It turns out to
   need no development work — `configs/model/qwen32b.yaml` is already pinned, so it is a submit-line
   flag. That lowers activation energy, not evidential value: an arm whose function is testing
   capability on a task with headroom belongs on the generation that *has* headroom. It runs after
   the corrected ladder reports, as a 96-episode C0-only seed-paired arm.
6. **The Phase-1 escalation rule is deferred to the new generation, not ignored.** Its trigger is
   keyed to the outcome of a grid whose task has since been shown not to express the manipulation.
   Re-reading it against 227886 would be reading a rule against a run it was not written for.

**So-what.**

1. **A feasibility certificate that proves only *existence* is not enough, and this is the
   transferable lesson.** A\* returned sound physics for two years' worth of certificates; the paths
   it returned were contact-exploiting tricks no agent could plan, from a canonical pose no episode
   ever started at. Three limbs are needed and now all three are enforced: a path exists, an agent
   could *state* it, and the obvious policy *finds* it from the real start distribution.
2. **The register caught the wrong thing first, and the correction is the interesting part.** D25's
   account was right about the mechanism it named and wrong about its numbers, because it derived
   windows that it could have measured. Both this entry's findings 2 and 5 were reached only by
   measuring — and finding 5 contradicts a claim the register itself records as settled.
3. **Groundedness needs to be defined against the trajectory, not the record.** The strongest result
   in the 227886 diagnosis was G3 passing emphatically. It still passes as stated, but the state it
   was measured against was partly a fiction of the hold. Any future groundedness claim should say
   which state it means.

---

## 2026-08-28 — The information gradient could not have appeared: the rotation quantum was tuned for a load that no longer exists

- **Area:** the difficulty ladder's semantics (§9.2), what the RQ1 null means (§9.6), which CPVI
  target a runtime gate should threshold (§9.5), and the pre-run certificate a task generation must
  ship with (§9.2).
- **Status:** **post-run**, and the first entry in this register written after a result was read.
  Declared as **D25** in `docs/methodology.md` §10.5. Every part is confirmatory — computed on job
  227886's recorded output — and is reported as such throughout. **No threshold in §5, §6 or §9.10
  moves, and RQ1's frozen `Y` is not re-pointed.**

### Trigger

Job 227886 — the D23 characterisation run, `preceptx-rq1` at commit `9170a74`, dataset
`54ed65e6cc9e7d17`, 480 episodes over C0–C4 × easy/medium/hard × seeds 0–31. D23(b) entered `medium`
prospectively on the strength of an A\* feasibility certificate: solvable on 10/10 seeds, oracle 11
steps against a budget of 25. It returned **8/160 successes (5.0%)**, with C0-medium at **3/32**.
Overall the run scored **25/480 (5.2%)**. The full analysis is `docs/rq1_227886_diagnosis.md`; this
entry records only the design consequences.

### Finding

**1. The rotation quantum is larger than the tolerance window it has to land in, and the reason is a
parameter that outlived its load.** `StepConfig.angular_impulse = 0.5` is documented in
`sim/actions.py` as *"Sized for controllable rotation (~34 deg per action) so an agent can aim **the
T** for threading."* DSE-057 replaced the T with the convex bar. Moment of inertia: T **0.2927**,
bar **0.1708** — the bar is **1.71× lighter to spin**, so the same impulse sweeps 1.71× the angle.
Predicted 34° × 1.71 = 58°; **measured 49.1° ± 16.9°** over every rotate action in the dataset. The
angular window in which the bar is thin enough to thread is **90° / 44° / 17°** for easy / medium /
hard. **On medium one action is larger than the entire target; on hard it is three times the
target.** Fraction of C0 episodes that ever reached a threadable pose: **100% / 69% / 22%**.

**2. DSE-058 removed the only sub-quantum control in the same commit that enlarged the quantum.**
`hold_orientation = True` restores the pre-action angle after every non-rotate action — confirmed in
the data, where `E`/`N`/`S`/`W` produce Δangle = 0.0000 with sd 0.0000. It correctly closed a real
degeneracy (up to 114° of contact rotation threading the channel with no rotate command ever
issued). But contact rotation had also been the only source of *fine* angular adjustment. **Two
changes in one commit, both pushing the same way, neither checked against the threading tolerance.**

**3. The failure is not the model's.** 99.5% of messages quote an (x, y); of those **99.98% land
within 0.01 units of the true state** (median error 0.00004), and 97.9% quote the true angle within
0.01 rad. The messages state the correct plan in the correct order — *"rotate it slightly to align it
horizontally before pushing it east"*. **The sender asks for "slightly"; the interface has no
"slightly".** A larger model produces the same correct intent into the same actuator.

**4. C1 is not a degraded channel — it is a collapsed policy, and it inverts the mediator.** Across
all **2,240** C1 handoffs the emitted action is `E` **100%** of the time: zero rotations, zero N/S,
**21.7 collisions per episode** against C0's 3.2. C1 is simultaneously the **highest** leakage-
corrected CPVI (**+0.118** against C0's +0.083) and the **only** 0/96 condition. H2's mediator moves
opposite to the outcome.

**5. Every cross-condition CPVI contrast is null under D23's declared estimand.** C0−C1 −0.036
[−0.089, +0.018]; C0−C2 −0.008; C0−C3 −0.016; C0−C4 +0.027 [−0.005, +0.060]. Per seed the C0−C4 gap
is positive in **19/32** seeds, mean +0.027, sd 0.112 — seed noise. On the outcome side only C1's
contrast excludes zero (Cliff's δ −0.063 [−0.115, −0.021]), and C1 is item 4. **C2 — added by D23(b)
precisely to separate a channel effect from an identity component — separates nothing, because there
is no effect to separate.**

**6. The pooled measurement, unlike the contrast, is the strongest it has been.** Leakage-corrected
pooled mean CPVI **+0.0895 bits [+0.0753, +0.1036]** against a random-label control of −0.0021 and a
permutation null of +0.0034, with a **PVI − CPVI gap of +0.111** (54% of apparent message value is
state echo). That is ~2× the three prior datasets (+0.058 / +0.048 / +0.039) on a much tighter
interval, and it now replicates across **four runs and two task geometries**.

### Impact

Unaddressed, RQ1's null would have been written up as *"degrading the channel does not measurably
change coordination"* on a task where **31% of medium episodes and 78% of hard episodes never once
entered a state in which success was physically possible**, and where the one arm that did move the
outcome did so by collapsing the recipient's policy to a constant. That is a floor effect and a
broken arm reported as a finding about information. Every difficulty contrast in the run is
confounded with an actuator ceiling that no reader could have detected from the artefacts.

### Risk reduced

The RQ1 null becomes **diagnosed** rather than merely observed, which is the difference between a
reportable negative result and a failed experiment. Separately, the gate acquires a target it can
actually be built on: item 4 of *The fix* shows the state-target ranking would have selected the
collapsed arm.

### Correction paths considered and rejected

**(a) Re-point `Y` at the graded chamber outcome.** Rejected. §4 froze `Y` at `y_binary_progress`
and D24 already records that re-choosing it "would rescue the gate without touching the defect,
which is the forbidden move". The graded outcome enters as a **labelled exploratory** analysis
alongside the confirmatory null, never in place of it.

**(b) Escalate to Qwen3-32B on the full factorial.** Rejected on both power and mechanism — see
*The fix*, item 5.

**(c) Re-tune `angular_impulse` and re-run this grid.** Rejected as a *correction to this run*. The
quantum is genuinely wrong and should change, but changing it produces a different task, hence a
different `dataset_hash`; job 227886 stands as the diagnosed baseline rather than being silently
superseded. Result-freezing discipline: this is a re-freeze, not an overwrite.

**(d) Repair the H1 mixed model.** Rejected in favour of demotion. `_handoff_model`'s
`vc={"episode": "0 + C(episode)"}` expands 480 episode dummies and does not converge (`|grad|`
4.2–8.3, "MLE may be on the boundary"). The episode-cluster bootstrap already carries every interval
reported. **The H1 p-values in `rq1.json` are not quoted anywhere in the write-up.**

### The fix

1. **The feasibility certificate gains an actuator check.** `sim/feasibility.py` currently certifies
   that a solution *path exists* under A\*. It must also certify that **one action quantum fits
   inside the tolerance window the aperture implies** — the check that would have refused this grid
   before submission. A path can exist on a lattice an agent reasoning in continuous coordinates
   cannot navigate; oracle feasibility and agent feasibility are different properties and only the
   first was ever tested. **Every task generation ships with both from here.**
2. **`angular_impulse` is re-derived from the aperture, not inherited.** Targets from the measured
   windows: **≈0.20** for ~20°/action (inside medium's 44°), **≈0.08** for ~8° (inside hard's 17°).
   Not applied to job 227886.
3. **A degenerate-arm check runs in-flight, not post hoc.** Per-condition action entropy in the
   runner; an arm emitting a single action **halts the sweep** rather than being averaged into a
   gradient. C1's entropy is exactly **0.000** against 2.03–2.19 elsewhere — detectable on the first
   cell, and it ran for 96 episodes.
4. **The gate's CPVI target is specified, and the specification is load-bearing.** Scored within each
   condition so no probe can read the condition tag, with a within-condition message null:

   | | C0 | C1 | C2 | C3 | C4 |
   |---|---|---|---|---|---|
   | action entropy | 2.070 | **0.000** | 2.071 | 2.030 | 2.188 |
   | state-target CPVI | +0.083 | **+0.118** | +0.091 | +0.099 | +0.056 |
   | **listening-target CPVI** | **+0.435** | **0.000** | **+0.449** | **+0.403** | **+0.462** |
   | success | 6/96 | **0/96** | 5/96 | 8/96 | 6/96 |

   **A gate thresholding the state target ranks the collapsed arm first; a gate thresholding the
   recipient's next action ranks it last, at exactly zero.** This is the *positive signalling without
   positive listening* distinction (Lowe et al., AAMAS 2019; Jaques et al. on causal influence) in an
   LLM-handoff setting, and the novel part is not the phenomenon but that **a V-information gate
   would have selected the broken arm**. Nulls −0.007 to −0.016.
5. **The 32B arm is sized, and pre-logged as defensive.** Not 480 episodes (that re-measures a
   collapsed arm and a geometrically dead difficulty, and ~19h exceeds the 8h wallclock class) and
   not 32 (~36% power on the most optimistic contrast — rhetoric, not evidence). **C0 only, 3
   difficulties × the same 32 seeds, 96 episodes, ~3.8h**, primary readout the **graded chamber
   outcome, seed-paired**, frozen binary `Y` untouched. Enumerated in advance: 32B ≈ 14B kills
   *"your model was too weak"*; 32B > 14B refines the control-versus-recognition decomposition but
   **cannot overturn it**, because hard is geometrically dead and medium's window is narrower than
   the stride. **No dissertation claim moves either way**, which is what makes it defensive and what
   fixes its dose at the minimum sufficient one. Precision parity required: 14B ran bf16, so 32B must
   — AWQ-INT4 would confound capability with quantisation.

### Result of the fix

The RQ1 null is reportable with a mechanism attached, at the cost of one parameter and one
certificate. The measurement result is unaffected and stands at its strongest: **+0.0895 bits
[+0.0753, +0.1036]**, clean against both controls, replicated across four runs and two geometries,
with a working graded outcome (chamber-reached sd 0.71 easy, 0.62 medium) that is *not* at floor. The
target-free prospective twin separates successful from failed episodes at **≈0.89 AUROC within
cell** — the answer to the circularity objection — on 25 successes, 1–7 per cell, which is thin and
is reported as thin.

### So-what / takeaways

1. **Oracle feasibility is not agent feasibility, and only the first was ever certified.** A\* found
   an 11-step solution on medium because A\* searches the lattice exhaustively. An agent reasoning in
   continuous coordinates and asking for "slightly" cannot navigate a lattice whose stride exceeds
   the target. The certificate tested the wrong property, confidently, for two task generations.
2. **A parameter tuned against one object silently mis-specifies when the object changes.** The
   impulse comment still names the T. Nothing in the type system, the tests or the certificate
   related the impulse to the load's moment, so a 1.71× change passed through unremarked. Any
   constant sized against a physical property needs that property in its derivation, not in a comment.
3. **A collapsed arm and a degraded arm are different objects and must not share an axis.** C1 read
   as the far end of an information gradient for three runs. It is a policy collapse — 100% one
   action, 21.7 collisions — and it posts the *highest* message-information score precisely because
   its recipient stopped listening. **The statistic a gate thresholds decides whether that is
   detected or rewarded.**
4. **A null needs a mechanism to be a result.** The distance between "degrading the channel did not
   change coordination" and "the channel could not have changed coordination, because 31% of medium
   episodes never entered a solvable state, and here is the 49°-versus-44° reason" is the distance
   between a failed experiment and a contribution.
5. **This is the first entry written after reading a result, and the register's value now rests on
   the negative decisions.** Y not re-pointed, the factorial not escalated, this run not silently
   re-run under a corrected constant. Those three are logged here precisely because they are the ones
   nobody would have noticed being made quietly.

### Correction, same day — three of this entry's numbers were wrong, and the mechanism was incomplete

Written before the windows were **measured** rather than derived. Corrected here rather than edited
silently above, so the register shows the correction being made:

1. **The windows quoted (90°/44°/17°) came from the nominal aperture and omitted the wall radius.**
   Geometric fit is 76.2°/34.3°/8.3°. But the windows that actually governed run 227886 are wider
   still — 98.6°/65.2°/16.3° — because of point 3 below. All three sets differ, and only the measured
   one describes the run.
2. **"Medium was at floor because the step exceeds the window" is false.** Medium's operative window
   was 65.2° against a 57.8° step: the step fits. 100% of medium episodes reached a passable
   orientation (this entry said 69%). Only **hard** was geometrically dead (0.13% triple coincidence,
   22% ever passable). Medium's floor is control and recognition, not impossibility — which changes
   what the fix has to achieve, and means the RQ1 null cannot be attributed to geometry on medium.
3. **The mechanism was incomplete, and the missing half is more serious than the impulse.**
   `hold_orientation` restores the pre-action angle *after* the settle; it does not prevent the load
   rotating *during* it. Starting at 30° on medium, the load reaches **0.48°** mid-action, slips
   through, and is recorded at 30.00°. So the DSE-058 degeneracy this entry credits with being closed
   was **hidden, not closed**, and the recorded angle is not the angle at which the load passed the
   gap. See `docs/rq1_227886_diagnosis.md` §5a; the fix is D26.
4. **The proposed impulse targets (≈0.20, ≈0.08) were sized by "step < window", which is the wrong
   criterion** — with half-turn symmetry and bidirectional rotation, reachability is orbit coverage
   of `θ₀ + m·step (mod 180°)`, which is not monotone in the step. 0.08 would have left hard
   unreachable: the error this entry diagnoses, repeated in its own remedy.

**None of this disturbs the entry's decisions.** Y is still not re-pointed, the factorial is still
not escalated, this run is still not re-run under a corrected constant, and the RQ1 null still stands
as recorded for the dataset as run. What changes is the *account* of why, and one load-bearing claim
about groundedness (§2 of the diagnosis) is now qualified rather than unconditional.

---

## 2026-08-28 — H4 as written could be confirmed by a proxy that tracks nothing but the channel label

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
