# RQ2 recommendation

- **Frozen primary Y (RQ1):** `y_binary_progress`
- **Recommended RQ3b gate target:** `y_terminal_success`
- **Encoder:** `BAAI/bge-base-en-v1.5`

Recommended gate target for RQ3b: y_terminal_success. Chosen by the rule declared in experiments/rq2.py before any RQ1 outcome was read - admissibility first, then encoder-invariance, then the label's own twin agreement, and corrected effect size only as a tie-break inside 0.05. RQ1's frozen primary Y (y_binary_progress) is unchanged by this either way; the register forbids re-pointing it after results.
