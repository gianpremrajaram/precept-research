# A2 — prompt v9 with budget slack (job 233499)

**This is a pilot, not a gate.** It is one of two arms that established which configuration the G1
capability gate is declared over; the gate itself is evaluated on `86ecbbdf35322dc3` (seeds 12–31),
which is disjoint from this run's seeds by design. Nothing here is a G1 verdict.

C0 only, numeric only, easy+medium, seeds 0–11, `max_steps` 50. 24 episodes, 1013 handoffs,
**10/24 success** (easy 7/12, medium 3/12).

**What it isolates.** The step budget, against `9f46e0e34fab81cf`, from which it differs in
`max_steps` alone (the prompts are byte-identical). The budget effect is **not** significant —
easy gained 3 lost 1 (McNemar exact p=0.63), medium gained 3 lost 2 (p=1.00) — so cap 50 is kept
for headroom, never credited as the mechanism. Against the v7 baseline's matched cells it gains
6 easy seeds and loses 0 (p=0.031).

**This is the configuration G1 is declared over.** CPVI here is 0.188 [0.115, 0.249] against PVI
0.273 (gap 0.085, control CPVI −0.002) — the first demonstration of the measurement primitive on a
task with outcome variance.

**What it cannot support.** No condition gradient (one condition), so `mixed_model` is refused and
`seed_sensitivity` reports the per-seed success rate rather than a C0-minus-hardest gap — see
DSE-067 in `docs/experiment_design_log.md` for why the old code returned zeros instead.

**It also backs the RQ3a transfer arm.** `calibration.json` here is this run's gate calibration
(`preceptx-calibrate --dataset-hash 8902072e1f47b6de`), against realised episode failure and never
CPVI: `fail` threshold 0.743, orientation +1, held-out AUROC 0.593, ECE 0.037 over n=1013; `cosine`
AUROC 0.504, i.e. chance, and single-class by construction.

**The `fail` AUROC is not established above chance.** Its episode-cluster 95% CI is [0.444, 0.737]
(2,000 resamples of the 24 episodes, matching the clustering `calibrate` cross-fits on), and 10.8%
of resamples fall at or below 0.5. Report it as underpowered, not as weak-but-real; the orientation
`+1` that the RQ3a transfer arm multiplies by is the sign of that same interval.

**Superseded as the transfer source, 29 Aug.** The G1 confirmation (`86ecbbdf35322dc3`) read out and
the transfer arm was re-keyed to it, as declared below before that run existed. `runs/rq3a/*/` now
transfers the statistic fitted on the confirmation, not this one; this calibration is kept as the
**sensitivity arm**, and the pair is the evidence the RQ3a headline is not an artefact of one
calibration set. See `runs/rq1/86ecbbdf35322dc3/README.md` for both columns — briefly: at home the
confirmation is much better (AUROC 0.754 [0.638, 0.870] against 0.593 [0.444, 0.737]), on
TraceElephant it is slightly worse (agent 0.525 against 0.576), and `cpvi_transfer` beats both
surface baselines with non-overlapping intervals under **either**.

The `fail` statistic fitted on this dataset still reproduces the sensitivity arm.
**The joblib is a trained probe and is gitignored, so it is not in this directory** — regenerate it
before any run that passes `--transfer`:

```bash
scripts/myriad/fetch.sh 8902072e1f47b6de        # stages runs/8902072e1f47b6de/*.parquet
uv run preceptx-calibrate --dataset-hash 8902072e1f47b6de   # writes runs/8902072e1f47b6de-calibration/
uv run preceptx-rq3a ... --transfer runs/8902072e1f47b6de-calibration
```

Pointing `--transfer` at *this* directory raises: the report is here, the probe is not.

*The declaration this run made, now discharged:* re-key the transfer arm to the G1 confirmation when
that reads out — 60 episodes against this run's 24, on unseen seeds; 40 of the 60 are in the two
gated difficulties and the other 20 add `hard`, which this run does not have. It read out on 29 Aug
and the re-key was taken **unconditionally**, including where it lowered the headline number.

Raw Parquet is not committed. Refetch with `scripts/myriad/fetch.sh 8902072e1f47b6de`.
