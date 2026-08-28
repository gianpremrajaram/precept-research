# RQ1 job 227886 — run evaluation and actuator diagnosis

> **Status:** post-run analysis of the D23 characterisation run. Not a gate attempt and carries no
> G1/G2/G3 verdict (D23(c)). The confirmatory RQ1 result against the frozen `y_binary_progress`
> stands as `runs/54ed65e6cc9e7d17-report/rq1.json` reports it; everything below either reproduces
> that or is labelled exploratory.
>
> **Run identity.** Git commit `9170a74` · job `227886` · sweep hash `afcd6a53ee11edd7` · dataset
> hash `54ed65e6cc9e7d17` · `Qwen/Qwen3-14B@40c069824f4251a91eefaf281ebe4c544efd3e18` ·
> C0–C4 × easy/medium/hard × seeds 0–31 · 480 episodes · 11,052 handoffs · driver `preceptx-rq1`.

The cluster report and an independent local reproduction from the parquet agree to three decimals
(C0 mean CPVI 0.08854 on the cluster, 0.08857 locally), so every figure below is confirmed twice.

---

## 0. Run status

Sweep finished 08:24:39; analysis finished **11:01:46** — **2h37m of A100 time spent on CPU-only
statistics**, roughly a third of the eight-hour job. Three figures were skipped:
`matplotlib absent (install the 'viz' extra)`. The container lacks the viz extra, so an eight-hour
job produced no figures.

One number from the cluster table that is not in the local reproduction and turns out to be the key
to the whole run: **C1 averages 21.7 collisions per episode** against C0's 3.2.

---

## 1. Why they cannot solve it

### The short version

They cannot solve it because **the steering wheel only turns in 49-degree jerks and the parking
space is 44 degrees wide.**

That is not a metaphor for something subtler. It is literally the finding.

### The task

The load is a bar: **1.4 long, 0.3 thick**. To pass a vertical gap it must be turned nearly flat, so
only its 0.3 thickness has to fit. It starts at 1.49 rad — about **85°, nearly straight up**.

The exact angular window in which the bar passes. These are **measured**, by pushing the bar at
each angle through the real channel at certification fidelity (`substeps=64`) and bisecting the
boundary — not derived from the aperture, for a reason §5a explains:

| Difficulty | Gap width | Bar passes within | (geometric fit alone) |
|---|---|---|---|
| easy | 1.2 | **±49.3° of flat** (a 98.6° window) | ±38.1° |
| medium | 0.8 | **±32.6° of flat** (a 65.2° window) | ±17.2° |
| hard | 0.5 | **±8.2° of flat** (a 16.3° window) | ±4.1° |

The measured windows are much *wider* than geometric fit allows, because the load is forced through
gaps it does not fit (§5a). The measured column is the one that governed run 227886.

And the actuator, measured over all 5,961 `ROT+`/`ROT-` actions in the dataset:

> **median 57.79°, mean 49.1°, sd 16.9° — and 74.2% land within 0.5° of 57.79°**

That distribution is not noise. It is a **deterministic 57.79° quantum with a 26% contact-truncated
tail**: in free space one rotate action turns the bar 57.79° with standard deviation *exactly zero*
(measured across 37 start angles at seven impulses), and the only thing that ever reduces it is
hitting a wall mid-action. The mean of 49.1° is the truncation dragging the point mass down.

- **easy**: step 57.8°, window 98.6° wide. Fits comfortably. Landable.
- **medium**: step 57.8°, window **65.2°** — the step fits, but only just: the lattice of reachable
  orientations offers about one landing point in the window. Marginal, not impossible.
- **hard**: step 57.8°, window **16.3°** — the step is **3.5× the entire window**.

You cannot reliably stop inside a window narrower than your stride.

### The data says exactly this

Fraction of C0 episodes that, at any point in ~23 steps, ever got the bar thin enough to pass:

| | ever thin enough |
|---|---|
| easy | **100%** |
| medium | **100%** |
| hard | **22%** |

**Only hard is geometrically dead.** Every medium episode reached a passable orientation at some
point; 78% of hard episodes never did. Medium's floor is therefore a *control and recognition*
failure, not a physical impossibility — a distinction that matters for what the fix has to be.

Thin enough is not sufficient — you need thin enough **and** vertically aligned with the gap **and**
within one push of the wall, all in the same step. Taking those in turn as: `|angle-from-flat| ≤ the
measured half-window`; `|com_y − 3.0| ≤ (aperture − y-extent)/2`; and `min distance to either channel
mouth ≤ 1.05` (one E-push) — that triple coincidence occurs on **18.9% / 6.4% / 0.13%** of easy /
medium / hard handoffs (C1 excluded). That is the ceiling before the model thinks at all.

### The model is not the problem

A verbatim raw message from the run:

> *"The load is currently at (2.8823, 3.1090) with an angle of 1.2315 radians... The next wall is at
> x = 4.0000, and the slit in that wall spans y = 2.4000 to 3.6000... However, the load is currently
> angled, so we need to **rotate it slightly** to align it horizontally before pushing it east.
> Pushing east now would not be effective without proper alignment. **Rotate the load slightly
> counterclockwise** to reduce its angle toward 0 radians."*

Correct geometry, correct constraint, correct ordering, correct plan. Checked at scale rather than
anecdotally:

- **99.5%** of messages quote an (x, y) position; of those **99.98% land within 0.01 units of the
  true state**, median error 0.00004
- **97.9%** quote the true angle within 0.01 rad

The messages are essentially perfectly grounded. **G3 does not merely pass, it passes emphatically.**

**The model asks for "slightly". The action interface has no "slightly".** Every "slightly" becomes
a 57.79° slam. The agents issue correct intent into an actuator that cannot express it.

### A second, smaller failure

When the golden state does occur — thin enough, aligned, next to the wall — what do they do?

| action chosen | share |
|---|---|
| **E (the only correct one)** | **36.1%** |
| ROT+ | 35.4% |
| ROT− | 19.7% |
| other (N/W/S) | 8.8% |

They take the winning move **36%** of the time (n = 675 golden handoffs, C1 excluded), and when they
do, it clears the wall **65%** of the time. A genuine recognition failure, but second-order: even a
perfect recogniser only converts the 6.4% of medium handoffs where the state exists at all, and the
0.13% on hard.

### C1 is a different animal

C1 truncates to 8 tokens, so agent B receives a fragment: `"The load is currently at (3.1013,
2.0092) with"`. No instruction, just a dangling clause. Across **all 2,240 C1 handoffs the action is
"E" 100% of the time.** Zero rotations. Zero N/S.

The policy has **collapsed to a constant**: it pushes east into the wall forever, which is why C1
racks up 21.7 collisions per episode and scores 0/96.

**C1 is not "a channel with less information". It is a broken arm.** It does not sit on the same
degradation axis as C2/C3/C4; it fell off the axis. This is why it posts the *highest* state-target
CPVI while achieving the *worst* outcome.

---

## 2. Is there still communication data?

**Yes — emphatically, and it is the good part of the run.** The communication layer and the actuation
layer are separate, and only one of them is broken.

| | |
|---|---|
| Handoffs | **11,052** |
| Message length | ~500 characters of natural language |
| Groundedness | **99.98%** within 0.01 units of true state |
| Genuine information per handoff | **+0.0895 bits**, 95% CI [+0.0753, +0.1036] |
| Random-label control | −0.0021 (clean) |
| Permutation null | +0.0034 (clean) |
| PVI − CPVI gap | **+0.111** — 54% of apparent value is state echo |
| No-outcome statistic → success | **AUROC ≈ 0.89** within cell |

That corrected 0.0895 bits is **roughly double** the three previous datasets (+0.058 / +0.048 /
+0.039), with a much tighter interval, and now replicates across **four runs and two task
geometries**.

A **working graded outcome** also exists. Binary success is 5%, but chamber-reached has real
variance:

| | chamber 1 | chamber 2 | chamber 3 |
|---|---|---|---|
| easy | 37% | 46% | 17% |
| medium | 73% | 19% | 8% |
| hard | 98% | 2% | 1% |

Easy sd 0.71, medium sd 0.62. Usable. Only hard is dead.

**Communication was measured successfully. Task success was not.** Those are different failures and
the data separates them.

---

## 3. What the numbers are

### Outcome (episode success, Cliff's δ vs C0, episode-cluster bootstrap)

| | successes | Cliff's δ vs C0 | 95% CI |
|---|---|---|---|
| C0 | 6/96 | — | — |
| C1 | **0/96** | **−0.063** | **[−0.115, −0.021]** |
| C2 | 5/96 | −0.010 | [−0.073, +0.052] |
| C3 | 8/96 | +0.021 | [−0.052, +0.094] |
| C4 | 6/96 | 0.000 | [−0.073, +0.063] |

C1 is the only interval excluding zero — and C1 is a collapsed policy, not a degraded channel. C2,
the surface-matched delay arm D23 added specifically to separate a channel effect from an identity
component, separates nothing, because there is no effect to separate.

Difficulty: easy 16/160, medium 8/160, hard 1/160.

### CPVI by condition, leakage-corrected (D23's declared estimand)

| | C0 | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|
| corrected CPVI | +0.083 | **+0.118** | +0.091 | +0.099 | +0.056 |

**Every C0−Ck corrected contrast straddles zero.** Best is C0−C4 = +0.027 [−0.005, +0.060]. Per
seed, the C0−C4 gap is positive in **19/32 seeds**, mean +0.027, sd 0.112 — that is seed noise.

**The mediator points the wrong way.** C1 is simultaneously highest-CPVI and the only zero-success
condition, so H2 is not merely unsupported; the sign is inverted.

---

## 4. What it means for the dissertation

### The thesis changes shape, and gets stronger

**What was intended:** *"Information at the agent boundary predicts coordination success, and a
runtime gate can exploit that."*

**What the data supports:** *"Measuring information at an agent boundary is much harder than the
literature admits. Here is an estimator with the artefacts removed, here is what it does and does
not tell you about outcomes, and here is a worked case where the information is real and measurable
while the outcome is choked by something else entirely — plus the method to tell those apart."*

A **measurement-and-methodology thesis** rather than an **effect thesis**. Less flashy, considerably
harder to attack, and the one the data supports. An effect thesis reporting a weak gradient on a
5%-success task dies to one viva question — *"how do you know that is not a floor effect?"* — with no
answer available. A methodology thesis that reports the floor effect, diagnoses it to the actuator,
and shows the channel intact anyway has answered that question on page one.

### Per research question

**RQ1 — returns a null with a diagnosed cause.** Report the confirmatory result against the frozen
`y_binary_progress` exactly as pre-registered, then add a clearly-labelled **exploratory** section
using the graded chamber outcome, where CPVI does correlate (ρ +0.31 to +0.48 on easy and medium).
Confirmatory null plus labelled exploratory follow-up is standard. What must **not** happen is
re-pointing RQ1's Y — D24 already forbids it, and that discipline is what makes the null credible.

**RQ2 — untouched, now well-powered, promote to headline.** Offline, runs on this dataset, does not
care whether the arena is solvable. 11,052 handoffs is a far better base than the earlier 40–80
episode datasets.

**RQ3a — untouched, and the fallback ladder says promote it.** CLAUDE.md states that failing the
Phase-1 gates elevates RQ3a to the headline and that RQ3a can carry the dissertation alone. Attempt 2
returned `fallback` on 27 August; this run confirms that diagnosis. Taking a pre-planned fallback is
evidence of good design, not of trouble.

**RQ3b — re-scope from causal to calibration.** "Blocking bad handoffs improves success" cannot be
demonstrated at 5% success — there is no headroom for an intervention to move. What *can* be
demonstrated is that the gate **fires on the right handoffs**, using the no-Y statistic at ≈0.89
AUROC within cell. That is a precision/calibration study. Any causal arm belongs on **easy only**.

### Code, in priority order

1. **The rotation quantum** (§5 below). The only change that would alter results. Treat as a new task
   generation with a new dataset hash, not a patch.
2. **Split `analyse_rq1` off the GPU job.** It held an A100 for 2h37m doing CPU statistics.
   `preceptx-rq2` already has the right shape.
3. **Add the `viz` extra to the container.** An eight-hour job produced no figures.
4. **Demote, do not repair, the mixed model.** `_handoff_model`'s `vc={"episode": "0 + C(episode)"}`
   expands 480 episode dummies and will not converge (`|grad|` 4.2–8.3, "MLE may be on the
   boundary"). **The H1 p-values in `rq1.json` are not trustworthy and should not be quoted.** The
   episode-cluster bootstrap already carries the inference.
5. **Detect degenerate arms in-flight.** C1 emitting one action 2,240 times should halt a sweep, not
   be averaged into a gradient. A per-condition action-entropy check in the runner catches it on the
   first cell.

### Myriad

Do not run another 480-cell factorial on this arena — it buys a tighter estimate of a quantity whose
ceiling is set by the actuator. See §6 for the sizing analysis of the 32B question.

---

## 5. Root cause: the impulse was tuned for a load that no longer exists

Verified against the repository, and it makes the fix a one-line parameter change with a principled
target rather than a vague "the actuator is coarse".

`StepConfig.angular_impulse = 0.5` carries this comment in `sim/actions.py`:

> *"Sized for controllable rotation (~34 deg per action) so an agent can aim **the T** for
> threading."*

It was calibrated against the **T-load**. DSE-057 then replaced the T with a **convex bar**:

| load | moment of inertia | ω = J/I at J = 0.5 |
|---|---|---|
| T (old) | 0.2927 | 1.708 rad/s |
| bar (successor) | **0.1708** | **2.927 rad/s** |

The bar's moment is **1.71× smaller**, so the same impulse produces **1.71× more rotation**.
Predicted 34° × 1.71 = 58°; measured **49.1°** (damping and settling absorb the rest).

**The rotation quantum was never re-derived when the load changed.** The comment still documents a
load that no longer exists.

Compounding it, DSE-058 set `hold_orientation = True`, which restores the pre-action angle after
every non-rotate action — confirmed in the data, where `E`/`N`/`S`/`W` produce Δangle = 0.0000 with
sd 0.0000. That correctly closed a real degeneracy (contact rotation of up to 114° threading the
channel with no rotate command ever issued). But it also removed the **only** source of sub-quantum
angular adjustment. **One commit simultaneously enlarged the quantum by 1.71× and removed the escape
hatch, and neither effect was flagged against the threading tolerance.**

**The fix is not simply "use a smaller impulse", and the obvious sizing is wrong.** The bar is
symmetric under a half-turn and both rotation directions are available, so the reachable set after
*k* rotate actions is the lattice `θ₀ + m·step (mod 180°)` for `|m| ≤ k`. What matters is therefore
not whether the step *fits inside* the window but whether that **orbit enters** it within budget —
and that is not monotone in the step. A step of 9.25° leaves hard unreachable where 11.7° reaches
it. Sizing the impulse by "step < window" reproduces the same class of error being fixed here.

**This belongs in `sim/feasibility.py` as a pre-run certificate, not in post-hoc analysis.** The
check is: *does the reachable orientation lattice enter the passing window from every jittered start,
within the step budget?* Every task generation should ship with that certificate. Had it existed,
this run would not have been submitted in this form. The corrected actuator, the re-derived ladder
and that certificate are DSE-059–063; the design rationale is D26.

## 5a. The deeper fault: `hold_orientation` masked contact rotation rather than preventing it

§5 explains why the quantum was too coarse. It does not explain why the *measured* windows in §1 are
15–20° wider than geometric fit permits. That has a separate and more serious cause.

`hold_orientation` restores the pre-action angle **after** the settle completes. It does not stop the
load rotating **during** it. Instrumenting the angle inside the settle loop, on medium, starting at
30° — well outside the ±17.2° geometric window:

| start angle | minimum angle *during* the action | angle *recorded* after | outcome |
|---|---|---|---|
| 20.0° | **0.26°** | 20.00° | passed |
| 25.0° | **0.39°** | 25.00° | passed |
| 30.0° | **0.48°** | 30.00° | passed |
| 32.0° | **0.49°** | 32.00° | passed |

The load rotates itself flat under contact torque, slips through the channel, and is then snapped
back to its original angle. **Two consequences, and the second is the one that matters:**

1. The DSE-058 degeneracy — *"a translation-only action sequence threaded the channel with no rotate
   command ever issued"* — was **never closed. It was hidden.** The state rotation is reverted, so it
   does not appear in the record; the threading still happens. The certified apertures were therefore
   softer in practice than on paper, which is why medium's real window was ±32.6° and not ±17.2°.
2. **The recorded angle is not the angle at which the load passed the gap.** The trajectory that
   produced the outcome ran through configurations the record does not contain. Every message that
   faithfully reported "angle 30°" was faithful to a state that was not the operative one.

Point 2 qualifies the G3 groundedness result in §2. Messages match the *recorded* state to 0.01
units, and that finding stands as stated — the agents describe what they are shown. But what they are
shown omits the mid-action configuration that decided the outcome, so grounding in the recorded state
is weaker evidence about grounding in the physical trajectory than it appears. It does **not** affect
the CPVI estimates, which condition on the same recorded state throughout and are internally
consistent; it affects what "grounded" is entitled to mean.

The fix is a true hold: give the body infinite moment for the duration of a non-rotate action, so no
contact torque can spin it, rather than reverting the angle afterwards. Measured against geometric
fit across seven apertures:

| hold | max error vs geometric window | monotone in aperture |
|---|---|---|
| restore-after (as run) | **20.03°** | no |
| zero angular velocity each substep | 1.12° | yes |
| **infinite moment during the action** | **0.01°** | yes |

Only the third makes the aperture mean what the ladder says it means. This is DSE-059.

### What the golden-state conversion actually shows

Recomputed under the definitions stated in §1: of the **244** handoffs in a golden state where `E`
was chosen, **65.2% cleared the wall on the next step**. Good but not free — the residual is states
that satisfy all three conditions at the coarse resolution of the criterion yet still catch an edge.

But episode success needs the coincidence **at each of two walls** (three chambers), which squares an
already small number:

| | episodes that ever saw a golden state | P(success \| saw one) | overall success |
|---|---|---|---|
| easy | 38.3% (49/128) | 18.4% | 12.5% |
| medium | 24.2% (31/128) | 22.6% | 6.2% |
| hard | 2.3% (3/128) | 33.3% | 0.8% |

*(C1 excluded. The golden-state definition is wall-1-specific, so it under-counts episodes that
thread wall 1 and reach a second golden state further east — which is why P(success | saw one) ×
P(saw one) sits below the measured rate on easy.)*

An earlier draft of this analysis quoted an "≈80/20 control/recognition" split. **That was too
glib and is withdrawn.** The accurate decomposition is three-layered: reaching a threadable state
(4.1% / 1.7% / 0.16% of handoffs), recognising it (38.3%), and needing both **twice per episode**.
The conversion step itself is lossless.

---

## 6. The 32B question

Neither extreme. See `docs/experiment_design_log.md` (D25) for the registered decision; the sizing is:

- The 480 episodes are 5 conditions × 96. A **matched comparative arm re-runs C0 only — 96 episodes**
  (3 difficulties × 32 seeds), not 480.
- A **32-episode probe is rhetoric, not evidence** (~36% power on the most optimistic binary
  contrast).
- **480 is the wrong target**: re-running C1 at 32B re-confirms a collapsed policy; hard is
  *model-free dead* because the 0.16% triple-coincidence rate is bounded by geometry, not by weights;
  and ~19h does not fit the 8h wallclock class.
- The **graded chamber outcome at n=96, seed-paired**, is the sensitive instrument — it detects a
  ~0.2-chamber lift, and it is the same outcome the exploratory RQ1 follow-up already uses.

Enumerate what the arm can return. If 32B ≈ 14B: capability is not the bottleneck and *"your model
was too weak"* dies in the viva. If 32B > 14B on easy/medium graded: it refines the control-versus-
recognition split; it cannot overturn it, because hard is geometrically dead and medium's window is
narrower than the actuator's stride. **Either way no dissertation claim moves.** That is a defensive
arm, and the correct dose for a defensive arm is the minimum sufficient one.

**The scientifically live version** of "does capability matter" is 14B-vs-32B on a task whose
actuator ceiling has been lifted — there the comparison has headroom and it directly serves the
measurement/actuation dissociation. Sequence: fix the quantum, run a C0-only ladder (~1h), *then*
decide whether 32B earns its hours.

---

## 7. Contributions, narrowed against prior work

1. **A permutation null for cross-arm V-information contrasts.** 78–96% of a reported cross-condition
   CPVI gap can be nothing but the condition label. *Conditioning* a V-information estimate on extra
   variables is not new — Hewitt et al., cited in the PVI paper, on Xu et al.'s predictive
   V-information. **What is unpublished is the within-condition permutation null for experimental
   contrasts and the quantification of the artefact.** Frame it as *a null model for experimental
   contrasts*, not as *conditioning out labels*, and cite Hewitt explicitly.
2. **The gate-inversion result.** "Informative but useless" is Lowe et al.'s *positive signalling
   without positive listening* (AAMAS 2019), with Jaques et al. operationalising the listening side
   as causal influence. **The phenomenon is not new.** The novel part is that **a V-information gate
   would have ranked the collapsed arm first** — highest CPVI (+0.118), 0/96 success — in an
   LLM-handoff setting with a truncation channel. See §8: the listening-side target inverts the
   ranking and fixes it.
3. **The measurement/actuation dissociation, with the instrument to detect it.** Communication
   measurably informative and grounded to four decimals; success at floor; the bottleneck provably
   downstream of the channel. Most multi-agent communication work cannot distinguish a bad channel
   from a bad actuator.
4. **A target-free statistic that predicts outcomes.** The prospective twin never sees the realised
   outcome and separates successful from failed episodes at ≈0.89 AUROC within cell — the answer to
   the circularity objection. *Caveat that must be stated: 25 successes total, 1–7 per cell.*
5. **A negative result with a quantified mechanism.** "The gradient did not appear, and here is the
   49°-versus-44° reason it could not have" is a contribution; an undiagnosed null is not.
6. **The pre-registration process itself** — a 25-entry register, a frozen Y not re-pointed after
   seeing results, a decision rule committed to code before the data existed, a gate not attempted a
   third time, and the negative decisions logged alongside the positive ones.

---

## 8. Which CPVI target a gate should use

Prompted by the Lowe/Jaques framing. Computed **within each condition separately**, so no probe can
read the condition tag, with a 5-permutation within-condition message null.

| condition | action entropy | state-target CPVI (corrected) | **listening-target CPVI (corrected)** | success |
|---|---|---|---|---|
| C0 | 2.070 | +0.083 | **+0.435** | 6/96 |
| C1 | **0.000** | **+0.118** ← highest | **0.000** ← lowest | **0/96** |
| C2 | 2.071 | +0.091 | **+0.449** | 5/96 |
| C3 | 2.030 | +0.099 | **+0.403** | 8/96 |
| C4 | 2.188 | +0.056 | **+0.462** | 6/96 |

*Listening target = the recipient's next macro-action. Nulls: −0.007 to −0.016, i.e. clean. C1's
recipient policy is a single constant action, so its entropy is exactly 0 and no message can carry
information about it — reported as 0 rather than estimated.*

**A gate thresholding state-target CPVI ranks the collapsed arm first. A gate thresholding
listening-target CPVI ranks it last, at exactly zero.** The choice of CPVI target determines whether
a runtime gate detects a collapsed channel or is fooled by it. This is a second, independent witness
of the collapse alongside action entropy and the 21.7 collisions per episode, and it wires the work
directly into the signalling/listening literature.

**An examiner will ask what C1's +0.118 is measured *against*.** The answer must be on the page: it
is a **state/outcome** target (`y_binary_progress`). Against the *listening* target it is zero by
construction. That is the point, not a defect.

---

## 8b. RQ3b gate calibration on this dataset

`gate.calibration.calibrate` run on all 11,052 handoffs, target **realised failure** (never CPVI —
the R5 circularity guard), 0.2 firing-rate budget, out-of-fold under a group split by episode.
Report: `runs/54ed65e6cc9e7d17-gate/calibration.json`.

| statistic | threshold | orientation | firing rate | AUROC | ECE |
|---|---|---|---|---|---|
| `info` | −0.0649 | −1 | 0.200 | **0.870** | 0.0035 |
| `fail` | +0.9923 | +1 | 0.200 | **0.870** | 0.0055 |
| `cosine` | −0.6535 | −1 | 0.200 | **0.619** | 0.000 |

`ece_reliable: true` at n = 11,052 — the first dataset large enough for the ECE to mean anything.

### The D23 identity check, applied to the gate

C1 is 100% failure, and its delivered messages are ~46 characters against ~500 elsewhere. A
statistic that merely recognises "this is a C1 message" therefore predicts failure perfectly on a
fifth of the data. Re-running the calibration sliced:

| slice | `info` / `fail` | `cosine` |
|---|---|---|
| all conditions | 0.870 | 0.619 |
| **C1 excluded** | **0.838** | **0.522** |
| within C0 | 0.662 | 0.637 |
| within C2 | 0.777 | 0.632 |
| within C3 | 0.870 | 0.544 |
| within C4 | 0.752 | 0.585 |

Two conclusions, pointing opposite ways:

- **The probe-backed statistics survive.** `info`/`fail` drop only 0.870 → 0.838 without C1 and hold
  0.66–0.87 within condition. They are reading message content, not the channel tag. **This is the
  RQ3b premise, and it holds.**
- **`CosineStatistic` does not.** It falls to **0.522 — chance** — once C1 is removed, so its
  headline 0.619 was almost entirely the collapsed arm. Within condition it recovers only to
  0.54–0.64. **The one statistic that exists specifically to answer the circularity objection is the
  one most dependent on the degenerate arm**, and it must be reported that way rather than as a
  clean probe-independent corroboration.

### `InfoStatistic` and `FailStatistic` are the same statistic

Spearman on their out-of-fold scores is **−1.000000** — exactly rank-identical, opposite sign. That
is why their AUROC agrees to three decimals in all six slices above. Both are probe-backed on the
same embeddings predicting the same binary label, so this is expected rather than a defect, but the
gate has **two** independent statistics, not three, and `GateConfig.statistic_key` offers a choice
between two of them that does not exist. Their correlation with `cosine` is ±0.474.

---

## 8c. RQ2 on this dataset — `runs/54ed65e6cc9e7d17-rq2/`

`preceptx-rq2 --dataset-hash 54ed65e6cc9e7d17`, offline, both encoders, completed 15:42:53.

### The label comparison re-selects the frozen label, blind

Under the D24 rule fixed in `experiments/rq2.py`'s constants before any of this run's output existed:

| label | minority share | corrected mean CPVI | 95% CI | encoder ρ | twin ρ | admissible |
|---|---|---|---|---|---|---|
| `y_binary_progress` *(frozen)* | 0.418 | +0.0892 | [+0.0763, +0.1028] | **0.689** | 0.489 | **yes** |
| `y_continuous_displacement` | — | +0.1147 | [+0.0895, +0.1408] | 0.585 | — | **yes** |
| `y_discrete_config` | **0.014** | **+0.1527** | [+0.1322, +0.1737] | 0.642 | 0.582 | no |
| `y_terminal_success` | **0.036** | +0.0315 | [+0.0201, +0.0446] | 0.839 | 0.303 | no |

**Recommended RQ3b gate target: `y_binary_progress`** — the same label §4 froze, selected on
encoder-invariance (0.689 vs 0.585) among the two admissible candidates. Encoder ρ is above the
0.50 re-freeze flag, so **no encoder re-freeze is triggered**.

Two things worth reading carefully:

- **The two labels that look most attractive are ruled out by the floor effect, automatically.**
  `y_discrete_config` has the **highest** corrected CPVI of all four (+0.153, interval well clear of
  zero) and `y_terminal_success` is the outcome anyone would want to gate on — and both fail
  admissibility on minority share (1.4% and 3.6% against a 0.10 floor). The rule caught the very
  effect this run diagnosed, without being told about it.
- **This does not contradict §4's exploratory recommendation.** `y_discrete_config` remains the right
  *exploratory outcome* precisely because it carries the most signal; it is a poor *gate target*
  because its classes are too imbalanced to threshold. Different jobs, different criteria.

### H4 as specified fails; the gate survives anyway

Three statistics, Spearman against CPVI, reported real / null / corrected under `DECLARED_ORIENTATION`:

| statistic | ρ real | ρ null | **ρ corrected** | 95% CI | AUROC failure |
|---|---|---|---|---|---|
| `info` | −0.106 | **−0.179** | **+0.073** | [+0.007, +0.133] | 0.131 |
| `fail` | +0.106 | **+0.179** | **−0.073** | [−0.133, −0.007] | **0.869** |
| `cosine` | −0.085 | −0.066 | **−0.019** | [−0.078, +0.035] | 0.381 |

- **No statistic tracks CPVI.** The best corrected ρ is **+0.073**, barely clearing zero;
  `cosine` is **−0.019**, indistinguishable from it. For `info` and `fail` the **null is larger than
  the real correlation** (0.179 vs 0.106) — exactly the artefact D24 built the null to catch, and it
  caught it. **H4 as written in DSE-022 is not supported.**
- **But `fail` predicts failure at AUROC 0.869.** So the proxy→CPVI link fails while the
  proxy→outcome link holds. **That is the link RQ3b actually needs**, and it is the honest framing:
  *the runtime statistic is a good failure predictor and a poor CPVI proxy, and those were always
  different claims.*
- **`info`'s 0.131 is the pre-declared sign being wrong, reported as-is.** `info` and `fail` are
  rank-identical (§8b), so under a shared declared orientation of −1.0 one must read 1 − the other.
  D24 fixed the sign in advance specifically so it could not be fitted to the data; the calibration
  module derives orientation instead (because a gate must act) and reads 0.870 for both. Both
  numbers are correct for their own question.

### H3 twin agreement

Pooled Pearson **0.259**, Spearman **0.489**, Bland-Altman bias +0.029 with limits of agreement
**[−0.710, +0.769]**, `n_kl_capped = 0`. Per condition Spearman runs 0.286 (C4) to 0.645 (C1).

The correlation is moderate but **the limits of agreement are ±0.74 bits on a quantity whose mean is
0.09 bits** — the prospective twin is *not* a per-handoff substitute for CPVI, and must not be
written up as one. It does work at the **episode** level (§2: AUROC 0.832 pooled, 0.887 mean across
cells), which is the level a gate aggregates to anyway.

---

## 9. Reproduction

Everything above is recomputed from `runs/54ed65e6cc9e7d17/*.parquet` (480 parts, 11,052 rows) with
the shipped estimators — `measure.pvi_cpvi.cpvi`, `.pvi`, `.control_task_cpvi`, `measure.twin`,
`experiments.rq2._null_cpvi`'s permutation — under `ProbeConfig(n_repeats=5)` for point estimates
and `n_repeats=1` inside permutation nulls, matching `RQ1Config` and `RQ2Config` defaults. Intervals
are episode-cluster bootstraps (2000 draws), paired where a corrected quantity is reported.

---

## 10. What still needs Myriad, and what does not

The single most useful fact from this run's operations: **job 227886 held an A100 for 2h37m running
statsmodels.** Everything in §8b and §8c above was produced on a laptop, offline, from artefacts the
GPU job had already written. Sorting the remaining work by whether it actually needs a GPU is worth
more than any scheduling trick.

### Needs no GPU at all — do these first, they are the core build

| work | driver | status |
|---|---|---|
| RQ2 (H3, H4, label + encoder comparison) | `preceptx-rq2` | **done**, §8c |
| RQ3b gate calibration | `gate.calibration.calibrate` | **done**, §8b |
| RQ1 re-analysis under corrected estimands | `analyse_rq1` | ready; runs on the login node |
| RQ3a corpus load and schema mapping | `experiments/rq3a_load` | offline by construction — the loaders take local paths and touch no network |
| Signal decomposition, localisation | `experiments/rq3a` | offline |

**RQ3a's loaders are offline by design.** Only its *replay labeller* needs a served model. That
means the promoted headline arm is mostly laptop work, and the GPU is needed for exactly one step.

### Needs a GPU, in dependency order

1. **RQ3a counterfactual replay (`rq3a_replay`) — the headline arm's only GPU dependency, so it goes
   first.** `Y1` is degenerate on both corpora (TraceElephant 220/220 failures, Who&When 184/184) and
   `Y2` is forbidden for circularity, so replay is the *only* route to a within-trace two-class
   target. Budget is `selected_steps × replays_per_step (default 5) × calls_per_replay`; `project()`
   takes no backend and refuses on the projected **minimum**, so run it as a dry run first and let it
   size the job rather than guessing. This is now on the critical path in a way it was not a week ago.
2. **The corrected-quantum C0 ladder.** After `angular_impulse` is re-derived (§5): C0 only, 3
   difficulties × 32 seeds, 96 episodes, ~1h. **This is the run that tells you whether the ceiling
   lifts**, and everything about a future factorial depends on its answer. Do not schedule a
   factorial before it returns.
3. **The 32B defensive arm (§6), only if the queue permits.** C0 only, 96 episodes, seed-paired,
   ~3.8h, bf16 for precision parity. Logged as declined at full scale in D25.

### Sequencing and parallelism

The honest read is that **there is very little to parallelise, because the dependencies are real**.
The corrected-quantum ladder must precede any factorial; the 32B arm is only interpretable on a task
whose ceiling has lifted, so it should follow the ladder rather than run beside it.

What genuinely can overlap:

- **RQ3a replay and the corrected-quantum ladder are independent** — different corpora, different
  code paths, no shared artefact. They are the one pair worth submitting together.
- **Every analysis step overlaps with every GPU step**, once the analysis stops living inside the
  serving job. That is the fix in §4 item 2, and it is worth more than queue tricks: it returns ~2.5h
  of A100 per run and lets a sweep's analysis be re-run, corrected and re-run again without touching
  the allocation.

Two operational fixes to land before the next submission, both cheap:

- **Split the analysis out of `scripts/myriad/pilot.sh`.** The driver should write the dataset and
  exit; analysis runs on the login node against the dataset hash, exactly as `preceptx-rq2` already
  does. This also removes the failure mode that nearly cost this run its report — an analysis still
  running when the pull happened.
- **Add the `viz` extra to the container.** Two eight-hour jobs have now produced zero figures.

### What not to schedule

A full C0–C4 × 3 × 32 factorial on the current arena, at any model size. Until the quantum is
re-derived and the ladder confirms the ceiling has moved, a factorial buys a tighter estimate of a
number whose value is set by `angular_impulse`, not by the channel.

