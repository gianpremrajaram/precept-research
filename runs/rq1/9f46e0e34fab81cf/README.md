# A1 — prompt v9 at the certified step budgets (job 233498)

**This is a pilot, not a gate.** It is one of two arms that established which configuration the G1
capability gate is declared over; the gate itself is evaluated on `86ecbbdf35322dc3` (seeds 12–31),
which is disjoint from this run's seeds by design. Nothing here is a G1 verdict.

C0 only, numeric only, easy+medium, seeds 0–11, `max_steps` 30/35. 24 episodes, 724 handoffs,
**7/24 success** (easy 5/12, medium 2/12).

**What it isolates.** The v9 clearance line, against the matched cells of the frozen v7 baseline
`188a3d556b824e3e`. Seed-matched, it gains 4 easy seeds and loses 0. Its sibling `8902072e1f47b6de`
differs from it in `max_steps` alone and isolates the step budget, which turned out not to matter.

**What it cannot support.** No condition gradient (one condition), so `mixed_model` is refused and
`seed_sensitivity` reports the per-seed success rate rather than a C0-minus-hardest gap — see
DSE-067 in `docs/experiment_design_log.md` for why the old code returned zeros instead.

Raw Parquet is not committed. Refetch with `scripts/myriad/fetch.sh 9f46e0e34fab81cf`.
