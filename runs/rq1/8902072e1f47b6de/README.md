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
AUROC 0.504, i.e. chance, and single-class by construction. The `fail` statistic fitted on this
dataset is what `runs/rq3a/*/` transfers to the log corpora — the joblib itself is a trained probe
and is gitignored, so it is refit from this dataset rather than committed. Re-key the transfer arm
to the G1 confirmation when that reads out: same configuration, 3x the episodes, unseen seeds.

Raw Parquet is not committed. Refetch with `scripts/myriad/fetch.sh 8902072e1f47b6de`.
