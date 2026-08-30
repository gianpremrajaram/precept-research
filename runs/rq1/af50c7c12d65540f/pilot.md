# Pilot gate report (G1/G2/G3)

- dataset: `af50c7c12d65540f`
- episodes: 80 | seeds: 10 | attempt: 2
- encoder: `BAAI/bge-base-en-v1.5@a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | probe: logistic (C=1.0, R=5)
- **recommendation: INVOKE FALLBACK - a gate still failed after the retune; take the ladder below.**

| Gate | Pass | Value | Threshold |
| --- | --- | --- | --- |
| G1 capability | PASS | 0.800 | 0.500 |
| G2 signal | FAIL | -0.250 | 0.100 |
| G3 groundedness | PASS | 1.000 | 0.800 |
| G3 correctness | FAIL | 0.242 | 0.257 |

- **G1 capability**: n_easy_c0_episodes=10.000, n_easy_c0_success=8.000 — easy C0 only; the hard cell is G2's business, not G1's
- **G2 signal**: c0_success=0.450, hard_success=0.700, success_gap=-0.250, cpvi_gap=-0.015, c0_mean_cpvi=0.222, hard_mean_cpvi=0.237, min_cpvi_gap=0.000, control_mean_cpvi=0.002, selectivity=0.130 — hard=C4; gate requires both the success gap and the CPVI gap to clear
- **G3 groundedness**: n_records=3419.000, n_with_numbers=3417.000
- **G3 correctness**: n_records=3419.000, n_perm=200.000, state_blind_mean=0.251, excess_over_null_mean=-0.008, null_max=0.261, p_value=0.980, oracle_frac_push=0.051, rotation_direction_agreement=0.512 — agreement with the certified scripted policy, against a within-episode permutation null on B's own actions; the gate is real > every permutation

## Fallback ladder (if a gate fails after the one retune)

1. Elevate RQ3a (TraceElephant external validity) to the headline - it is the pre-planned fallback and can carry the dissertation alone.
2. Simplify the task (wider slits, fewer chambers, or a shorter horizon) and re-pilot once.
3. Reframe RQ1 as a diagnostic negative: report the absence of an information gradient as the finding rather than chasing a positive.