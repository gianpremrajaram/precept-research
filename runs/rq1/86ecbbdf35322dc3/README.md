# G1 confirmation — v9, C0, seeds 12–31 (job 236653)

**This is the verdict of record for G1.** Declared before the run, evaluated once, on seeds disjoint
from every run that informed the threshold. 60 episodes, 2,597 handoffs, 37 min wall clock on one
A100-40GB. Manifest, summary and calibration are frozen here; the Parquet is not committed
(`scripts/myriad/fetch.sh 86ecbbdf35322dc3`).

## The verdict

**G1 PASS — and by nothing.** The pre-registered threshold (PREREGISTRATION §6) is *C0 self-play
episode success ≥ 0.5 at easy difficulty*. The run returned **10/20 = 0.500**: a pass on `>=`, with
a Wilson 95% interval of **[0.299, 0.701]**. A design whose true rate is exactly 0.5 passes this
gate about half the time, so this is a pass on the letter of the declaration and not a demonstration
of comfortable capability. It is reported that way everywhere it is reported.

| difficulty | success | solved seeds |
|---|---|---|
| easy | **10/20** | 12, 16, 19, 20, 21, 23, 25, 28, 29, 31 |
| medium | 3/20 | 13, 14, 29 |
| hard | 1/20 | 30 |
| all | 14/60 (0.233 [0.133, 0.350]) | |

Seeds 12–31 were chosen because the threshold was set with 0–11 in view. Against A2's matched cells
the rate fell on unseen seeds — easy 7/12 (0.583) → 10/20 (0.500), medium 3/12 (0.250) → 3/20
(0.150) — which is the optimism a pilot-informed threshold is expected to carry, and the reason the
gate was declared on unseen seeds rather than re-read on the pilot's.

**Two declarations existed and only one is the register.** `docs/myriad.md` §9a stated the gate as
*easy ≥ 8/20 AND medium ≥ 3/20* (0.40 easy, plus a medium clause); PREREGISTRATION §6 states
*≥ 0.5 easy*, easy-only, and `PilotConfig.g1_success_floor` implements that. The run satisfies both
(easy 10 ≥ 8 and ≥ 0.5; medium 3 ≥ 3), so the verdict is not in dispute — but at easy 8/20 or 9/20
the two documents would have disagreed about a gate outcome. The pre-registration is the register of
record; the runbook line was a paraphrase and has been corrected to quote it.

## What this run does **not** settle

- **G2 signal is unassessed, not passed.** Its declared cell is C0/C1/C3/C4 × easy/hard × seeds 0–9.
  This run is **C0 only**, so there is no condition contrast: `contrasts` is empty and the mixed
  model correctly refuses to fit (`"only condition C0 is present, so C(condition) is rank-deficient"`).
  An unassessable gate is not a failed one and spends no retune. The E3 re-gate is the next run.
- **G3 splits, and the corpus fails the half that matters.** The grounding limb scores **0.9998**
  here (2,597/2,597 messages cite numbers; 2,593 are perfectly grounded; per-difficulty 1.0000 /
  0.9997 / 0.9997). On its own that number is worth almost nothing — attempt 2 scored 0.999 on a
  corpus whose modal *inference* was wrong. The second limb PREREGISTRATION declares at F0,
  *agreement with the oracle's next action*, was implemented on 2026-08-29, after this verdict and
  before the E3 cell existed, and it reads **FAIL** here:

  | read | value |
  |---|---|
  | oracle-action agreement | **0.285** |
  | within-episode permutation null, 95th pct (the gate) | 0.322 |
  | null mean (the state-blind rate) | 0.315 |
  | permutation *p* | **1.000** |

  The pair matches the certified rotate-then-push plan **less** often than its own actions do when
  shuffled inside the episode. Three supporting reads say the same thing and do not depend on how the
  limb breaks ties near 90°: only **4 of 60** episodes ever bring the load within 6° of alignment
  (0.65 % of handoffs); **rotation-direction agreement is 0.519** on 1,400 rotations, a coin flip at
  SE 0.013; and mean |misalignment| runs 84.5° at the first handoff to 36.3° at the last, which is
  what a bounded random walk does from a broadside start rather than what convergence looks like.
  ROT+ and ROT- are issued 723 and 678 times — near-perfect balance, i.e. oscillation.

  **So the 0.500 G1 pass was not obtained by executing the certified plan.** That is a prospective
  statement about E3, recorded before E3 ran: if the C0→C4 gradient comes back flat, the mechanism
  is already named — B is not acting on the pose, so degrading the message about the pose has little
  to change. Per-difficulty the limb fails everywhere (easy 0.282 vs 0.337, medium 0.246 vs 0.283,
  hard 0.323 vs 0.360).

## What it does settle: the measurement primitive replicates

Every CPVI quantity reproduces on disjoint seeds, while the outcome rate moves:

| quantity | A2 pilot (24 ep, seeds 0–11) | confirmation (60 ep, seeds 12–31) |
|---|---|---|
| mean CPVI | 0.1876 [0.115, 0.249] | **0.1853 [0.152, 0.217]** |
| mean PVI | 0.2728 | 0.2695 |
| **PVI − CPVI gap** | 0.0851 | **0.0842** |
| control CPVI | −0.0024 | −0.0016 |
| selectivity | 0.1900 | 0.1869 |
| per-handoff CPVI sd | 0.0282 | 0.0300 |

Agreement is within about 1% on every line, with the interval roughly halving in width, on seeds the
first estimate never saw. The outcome rate over the matched cells moved from 0.417 to 0.325 over the
same change. **The outcome is seed-sensitive; the measurement is not** — which is the property a
measurement primitive has to have, measured rather than assumed.

Supporting reads from the same report:

- **Shuffled-message audit:** observed CPVI 0.1853 against a null of −0.0010 (sd 0.0011, max 0.0020)
  over **200** permutations, ***p* = 0.00498** — the 1/201 floor, i.e. the observed value again
  exceeded every permutation, now at ten times the resolution. The observed CPVI sits ~173 null-SDs
  above the null mean. (The first freeze ran 20 permutations and could only report *p* = 0.0476, the
  1/21 floor, which reads in print as a marginal pass; `n_shuffle` is now 200 by default and this
  report is regenerated at that count.)
- **Selectivity 0.187** against a control task whose CPVI is −0.002: the Hewitt–Liang control
  separates "the message carries this" from "the probe learned the task".
- **Signal decomposition:** absent-signal rate 0.412 [0.357, 0.465], unused-signal rate 0.265
  [0.215, 0.321]. Progress accompanies high CPVI in 610/1,298 handoffs against 229/1,299 at low CPVI.
- **Confound to carry forward:** partial Spearman of CPVI against message length is **0.439**. The
  length-matched contrast is empty here because there is only one condition; it is not optional in
  the E3 re-gate.

## It re-keys the RQ3a transfer arm, as pre-declared

`calibration.json` here is this run's gate calibration (`preceptx-calibrate --dataset-hash
86ecbbdf35322dc3`), against realised episode failure and never CPVI: `fail` threshold 0.9336,
orientation +1, held-out AUROC **0.7538**, ECE 0.031 over n = 2,597, and `ece_reliable` is true for
the first time (n ≥ 200). The episode-cluster 95% CI is **[0.638, 0.870]** with no resample at or
below 0.5 — against A2's 0.593 [0.444, 0.737], which straddled chance. The statistic being
transferred is now demonstrably informative at home.

`cosine` reads AUROC 0.569 with **orientation −1** here against +1 on A2. A near-chance,
single-class statistic flipping sign between calibration sets is exactly why orientation is read
from the report and never typed as a flag, and why `cosine` is not the transferred arm.

The joblib is a trained probe and is gitignored, so it is not in this directory. Regenerate before
any run passing `--transfer`:

```bash
scripts/myriad/fetch.sh 86ecbbdf35322dc3                     # stages runs/86ecbbdf35322dc3/*.parquet
uv run preceptx-calibrate --dataset-hash 86ecbbdf35322dc3    # writes runs/86ecbbdf35322dc3-calibration/
uv run preceptx-rq3a ... --transfer runs/86ecbbdf35322dc3-calibration
```

Pointing `--transfer` at *this* directory raises: the report is here, the probe is not.
