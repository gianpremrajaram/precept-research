# RQ1 corrected-generation C0 capability smoke — `188a3d556b824e3e`

| | |
|---|---|
| Dataset hash | `188a3d556b824e3e` |
| Sweep hash | `2c9d3a487785141b` |
| Simulation digest | `92b0c63b141ab074` |
| Git commit | `3824de603647e23f70f0c7f7f9c2521ee19d5153` |
| Scheduler job | `232980` (Myriad, NVIDIA A100-PCIE-40GB) |
| Model | `Qwen/Qwen3-14B` @ `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| Prompt version | `v7` |
| Conditions · serialisations · difficulties | `C0` · `numeric` · `easy, medium, hard` |
| Seeds | 0–31 |
| Episodes · handoffs | 96 · 3,198 |
| Wall time | 3,730 s |

## What this run is, and is not

It is a **C0-only capability/characterisation grid** on the DSE-059–063 corrected task. It is
**not** an RQ1 C0–C4 information-gradient result, not a causal or gate verdict, and not evidence
for or against any cross-condition claim: it has one condition, so there is no contrast in it.

## Headline

**1 of 96 episodes reached the goal** (easy 1, medium 0, hard 0). Every episode saturated its step
budget. 3,148 of 3,198 handoffs never left chamber 1.

The actuator correction is *not* what failed — `ROTATION_STEP_DEG = 12.0` is realised exactly and
seed 7 reaches the goal at step 28 of 30. The binding constraint is the observation: 99.9% of
messages quote the load's angle, 6.6% attempt the projection that decides whether it fits, and
97.6% of eastward pushes were issued at poses that could not pass the slit. The full mechanism,
with the rejected correction paths, is `docs/experiment_design_log.md` (2026-08-29) and
`docs/myriad.md` section 9.

## Files

- `manifest.json` — the run manifest as written by the job (git SHA, resolved sweep, seeds, model
  and encoder revisions, serving substrate, dependency versions, the exact command).
- `summary.json` — episode/handoff counts, success rate, wall time.
- `lineage.csv` — one row per dataset generation, so a reader can see which runs are comparable.
- `checksums.sha256` — SHA-256 of the raw-data release asset below.

## Raw data

The 96 Parquet parts are **not** in Git (`runs/**/*.parquet` is ignored, and Git LFS is restricted
to final figures and the demo trace). They are published as a GitHub Release asset:

- tag `rq1-c0-smoke-188a3d556b824e3e`, asset `rq1-c0-smoke-188a3d556b824e3e-parquet.tar.zst`
- verify with `shasum -a 256 -c checksums.sha256`

## Reproducing the analysis

```bash
preceptx-analyse --dataset-hash 188a3d556b824e3e
```
