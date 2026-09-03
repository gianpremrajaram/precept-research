# E3 re-gate — the rung-2 successor task, C0/C1/C3/C4 × easy/hard × seeds 0–9 (job 238085)

**This is the verdict of record for G2, and it closes the E3 ledger.** The pre-registered E3 cell,
declared in `PREREGISTRATION.md` §6 and evaluated once. 80 episodes, 3,419 handoffs, 47 min wall
clock on one A100-PCIE-40GB. Run at git SHA `10283b0`, the commit PR #76 merged; sweep hash
`f2b7bc42a511a735` matches the dry run declared before submission. The Parquet is not committed
(`scripts/myriad/fetch.sh af50c7c12d65540f`).

## The verdict

**`fallback`** — attempt 2, a gate still failing after the one permitted retune.

| Gate | | Value | Threshold |
|---|---|---|---|
| G1 capability | **PASS** | 0.800 | ≥ 0.500 |
| G2 signal | **FAIL** | −0.250 | ≥ 0.100 |
| G3 groundedness | **PASS** | 1.000 | ≥ 0.800 |
| G3 correctness | **FAIL** | 0.242 | > 0.257 (null 95th pct), *p* = 0.980 |

- **G1** reads easy C0 8/10. Comfortably above the floor and above the G1 confirmation's 10/20 —
  the successor bar task is *more* solvable than the T it replaced.
- **G2** contrasts C0 against the hardest condition, which is **C4 by declaration** (H1 is stated
  "monotonically across C0 → C4" and §6's own worked example is the C0−C4 contrast). Both halves
  fail: success gap −0.250, CPVI gap −0.015. Reported detail: `c0_success=0.450`,
  `hard_success=0.700`, `c0_mean_cpvi=0.222`, `hard_mean_cpvi=0.237`, `control_mean_cpvi=0.002`,
  `selectivity=0.130`.
- **G3 groundedness** reads 1.000 on 3,417 of 3,419 records carrying numbers.
- **G3 correctness** fails on its first outing against a full condition grid: the pair matches the
  certified plan *less* often than its own actions shuffled within episode.

## The pre-registered prediction, and how it came out

§6 fixed this before the first successor model call:

> Rotation is now operationally necessary at every rung, so that instruction becomes correct, and
> **degrading the message (C1, C4) should reduce success relative to C0** … if the inversion
> persists, the instruction account is wrong and rung 3 stands as the finding.

**Half confirmed.** C1 confirms it and harder than predicted — 0/20, with 44.3 wall collisions per
episode. C4 inverts it again, and by more than the T-load attempt did. The instruction account is
therefore not sufficient, and **rung 3 stands**: the absent gradient is the finding.

## The four conditions

| cond | delivered chars | success (20 ep) | 95 % CI | CPVI | CPVI 95 % CI | PVI | PVI − CPVI | selectivity | steps | collisions |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 clean | 373 | 0.450 | [0.25, 0.70] | 0.222 | [0.165, 0.278] | 0.305 | 0.084 | 0.222 | 40.2 | 3.8 |
| C1 8-token cap | 46 | **0.000** | [0.00, 0.00] | −0.023 | [−0.157, 0.060] | −0.027 | −0.004 | −0.031 | 50.0 | 44.3 |
| C3 2-row window | 377 | 0.150 | [0.05, 0.35] | 0.143 | [0.104, 0.186] | 0.262 | 0.119 | 0.145 | 46.3 | 2.1 |
| C4 40 % dropout | 221 | **0.700** | [0.45, 0.86] | 0.237 | [0.200, 0.278] | 0.301 | 0.065 | 0.236 | 34.5 | 8.6 |

Easy/hard splits (out of 10 each): C0 8/1, C1 0/0, C3 3/0, C4 10/4.

**CPVI rank-orders realised success across all four conditions** (C4 > C0 > C3 > C1 on both).
C1's CPVI interval spans zero: a 46-character message carries no usable information about progress,
which is the measurement behaving correctly rather than failing.

## Contrasts, and the length control that reframes C4

| vs C0 | Cliff's δ (success) | 95 % CI | mixed coef | *p* raw | *p* Holm | length-matched Δ | unrestricted Δ | episodes in overlap |
|---|---|---|---|---|---|---|---|---|
| C1 | −0.450 | [−0.70, −0.25] | +0.071 | 0.320 | 0.320 | **−0.500** | −0.450 | 26/40 (1 bin) |
| C3 | −0.300 | [−0.55, −0.05] | −0.106 | 0.135 | 0.270 | **−0.277** | −0.300 | 40/40 (3 bins) |
| C4 | +0.250 | [−0.05, +0.55] | +0.142 | 0.048 | 0.143 | **−0.071** | +0.250 | 13/40 (1 bin) |

**The DSE-044 length-matched control is the load-bearing column and it did exactly what it was
built for.** C1's and C3's deficits survive stratification on delivered message length; C4's
*advantage does not*. Within the single overlapping length stratum the C4−C0 success delta is
−0.071 rather than +0.250, and the CPVI delta −0.069 rather than +0.030. On the pre-declared
confound control **C4 is not better than C0; it is shorter than C0.** The caveat is stated wherever
this is: that stratum holds 13 of 40 episodes in one bin, so it is a thin instrument, and it is a
sensitivity analysis rather than the headline.

## The mechanism: per-condition receiver competence

`experiments.rq1.action_agreement` runs G3's correctness instrument within each condition
(2,000 within-episode permutations each, seeded per condition). The gate pools; the pooled number
hides the finding.

> **Re-derived 2026-08-30.** This limb was first frozen at 200 permutations drawn from a single RNG
> stream shared across conditions. Both were fixed — seeded per condition so a condition's null does
> not depend on which others share the dataset, and raised to 2,000 draws because at 200 the
> estimator's standard error (≈ 0.005) could not resolve C4's *p* against the Bonferroni line. The
> table below is the re-derivation; only the null-derived columns moved (**C4 *p* 0.010 → 0.004**,
> converging to 0.006 at 20,000 draws). Agreement, rotation-direction and flip-rate figures are
> deterministic and are unchanged. **The gate verdict of record (`pilot.g3_correctness`, pooled,
> 200 permutations) is a separate function and is untouched.**

| cond | oracle agreement | null 95th pct | null mean | *p* | verdict | rotation-direction agreement | *n* rotations | flip rate | unused-signal rate |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 0.330 | 0.331 | 0.314 | 0.069 | fail | 0.528 | 436 | 0.559 | 0.268 |
| C1 | 0.046 | 0.046 | 0.044 | 0.074 | fail | 0.511 | 90 | 0.000 | 0.283 |
| C3 | 0.261 | 0.342 | 0.327 | 1.000 | fail | 0.430 | 563 | 0.570 | 0.323 |
| **C4** | **0.399** | 0.388 | 0.373 | **0.004** | **PASS** | **0.614** | 381 | **0.445** | **0.154** |

**Only the lossy channel produces a receiver that acts on the pose above chance.** C4's
*p* = 0.004 survives Bonferroni across the four tests (0.004 × 4 = 0.016). Every measure agrees:
C4 turns the right way 61.4 % of the time against C0's 52.8 % and C3's 43.0 %, oscillates least
(flip rate 0.445 against 0.559 and 0.570), rotates least (381 against 436 and 563), and leaves far
less of its own signal unused (0.154 against 0.268 and 0.323). C1's flip rate of 0.000 on 90
rotations is not stability: only two of its twenty episodes rotate at all, and those never reverse.
Meanwhile C0 — fully grounded, numerically exact, carrying the correct instruction — is
indistinguishable from its own shuffled actions.

Supporting reads, pooled: rotation-direction agreement **0.512** on 1,470 rotations (a coin flip);
13 of 80 episodes ever bring the load within 6° of alignment; mean |misalignment| runs 83.9° at the
first handoff to 45.9° at the last; the oracle wants a rotation on 94.9 % of handoffs while B
translates on 56 % of them.

### What the three messages say

Same handoff, same episode, differing only in what `apply_channel` did.

- **C0 (373 chars)** — *"…The load's vertical extent at this angle is 1.4191, which is slightly
  larger than the usable slit clearance of 1.1000. You should rotate the load slightly to reduce
  its vertical extent so that it fits through the slit. Push gently rightward once it is aligned."*
- **C1 (46 chars)** — *"The load is currently at (2.2706, 2.9380) with"*. Position only; the
  directive is severed. B pushes east on 90 % of handoffs and rotates on 9.6 %.
- **C4 (221 chars)** — *"The at (2.2706, 2.9380) with an angle of radians. … You should load
  reduce vertical so fits through the slit. Push gently rightward once it is aligned."* The numbers
  are gone; the directive survives.

**The reading this supports, and its limit.** Dropout destroys the numeric detail and leaves the
directive intact; on this task that raises usable information, because the numbers invite a
fine-tuning the receiver cannot perform. The honest limit is that C4 confounds *what was removed*
with *how much*: the length-matched control says the success advantage is a length effect, and the
agreement and CPVI results above are not length-matched. `PREREGISTRATION.md` §8b A2 declares the
post-hoc arm that arbitrates this, with its decision rule fixed before the run.

## What the measurement did on this corpus

Nothing here indicts the instrument, and several things vindicate it.

| check | reads | interval / null | verdict |
|---|---|---|---|
| CPVI ranks realised success | 4 of 4 in order | Spearman ρ = 1.00 on four points | holds |
| H2 mediation, C1 | indirect −0.197 | [−0.557, −0.081] · 43.7 % mediated | CI excludes 0 |
| H2 mediation, C3 | indirect −0.058 | [−0.234, −0.001] · 19.4 % mediated | CI excludes 0 |
| H2 mediation, C4 | indirect +0.023 | [−0.029, +0.148] · 9.1 % mediated | spans 0 |
| path *b* (CPVI → success) | +0.766 | +0.652 with length controlled | holds |
| RD-15 shuffled-message audit | CPVI 0.1318 | null mean 0.0006, max 0.0047, 200 perms | *p* = 0.00498 |
| control-task CPVI | 0.0018 | the §5 capacity rule | ≈ 0 |
| within-episode attenuation | −20 % | condition coefs when per-handoff CPVI enters | directional |
| seed sensitivity of the gap | mean −0.25, sd 0.354 | 6/10 seeds at 0.0, one at −1.0, spread 1.0 | **high** |

The amber row is real and constrains the write-up: the condition gap is heavily seed-dependent at
ten seeds, so no magnitude claim is made on the C0−C4 contrast, and the inversion is not reported as
a stable law.

RQ2's estimands, re-run on this corpus (`runs/af50c7c12d65540f-rq2/`), pass where the C0-only
confirmation could not test them: H4's shuffle-corrected correlations are +0.275 [0.150, 0.384]
(`info`), −0.275 [−0.379, −0.155] (`fail`) and +0.269 [0.147, 0.371] (`cosine`), all excluding zero,
with `fail` separating realised failure at AUROC 0.906. Encoder invariance holds
(ρ = 0.816, `label_ranking_invariant: true`).

## What this run does **not** settle

- **It does not show that the numbers specifically are the un-actionable content.** C4 removes them
  randomly and shortens the message at the same time. A2 (declared, not yet run) holds length and
  swaps which content survives; A4 (proposed, blocked on a decision) would complete the 2×2.
- **It does not establish a channel-degradation gradient**, and under §6 no further attempt may try
  to. Rung 3 is the finding.
- **It does not distinguish a receiver limitation from a serialisation limitation.** Every episode
  here is `numeric`. A `numeric`/`grid`/`nl` arm at C0 would, using the same agreement limb, and it
  needs no new code.
- **It does not distinguish a receiver limitation from a 14B capability limit.** The 32B tier
  exists in `configs/model/` and the same limb would read it.

## Recorded deviation

`--max-steps 50` broadcast to every difficulty, against the certified 30/35 (`STEP_BUDGETS`,
ceil(2.5 × the oracle optimum)). Logged in `PREREGISTRATION.md` §6 (*Note against §2*) and
`docs/methodology.md` D29 **before** submission, because the G1 verdict of record already carried
the same deviation unrecorded. D26 measured the budget effect as not significant (*p* = 0.63 / 1.00),
the E3 quantity is a difference rather than a level, and the correctness limb supplies the
independent reason extra steps bought nothing: episodes end at 45.9° mean misalignment with rotation
direction at chance.

## Reproduce

```bash
scripts/myriad/fetch.sh af50c7c12d65540f
uv run preceptx-analyse --dataset-hash af50c7c12d65540f   # rq1.json, action_agreement.csv
uv run preceptx-rq2     --dataset-hash af50c7c12d65540f   # twin agreement, proxy tracking
```

The gate table is `pilot.json` / `pilot.md` beside this file, produced by
`run_pilot(load_records("af50c7c12d65540f"), Featuriser(EncoderConfig()), attempt=2)`.
