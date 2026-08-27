"""Terminal limit-cycle diagnostic for a pilot dataset (E3 attempt 1 vs attempt 2).

Attempt 1 failed G1 with 53 % of its failed episodes ending in a period-1 or period-2 cycle;
prompt v5 was the retune aimed at exactly that. This re-derives the same statistic so the two
attempts are comparable on one table, and breaks it out by condition - which is what shows
whether channel degradation is acting as a cycle-breaking randomiser rather than as an
information loss (a negative G2 success gap has no other obvious mechanism).

    uv run python scripts/diagnose_cycles.py runs/1c994b87bbca8257 runs/eddd19c654515bb2
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

# Four repeats of the pattern: three would fire on ROT+,ROT-,ROT+ , which is a plausible two-step
# correction rather than a policy that has stopped moving.
MIN_CYCLE = 4


def terminal_cycle(actions: list[str]) -> int:
    """Length of the trailing period-1 or period-2 repeat, 0 if shorter than MIN_CYCLE."""
    best = 0
    for period in (1, 2):
        if len(actions) < period * 2:
            continue
        tail = actions[-period:]
        run = 0
        i = len(actions)
        while i >= period and actions[i - period : i] == tail:
            run += period
            i -= period
        # A period-2 "cycle" of one repeated action is really period-1; do not double-count it.
        if period == 2 and tail[0] == tail[1]:
            continue
        best = max(best, run)
    return best if best >= MIN_CYCLE else 0


def episodes(path: Path) -> pd.DataFrame:
    parts = sorted(glob.glob(str(path / "*.parquet")))
    if not parts:
        raise SystemExit(f"no parquet parts under {path}")
    df = pd.concat([pd.read_parquet(p) for p in parts]).sort_values(["episode_id", "step"])
    rows = []
    for _, g in df.groupby("episode_id", sort=True):
        acts = [json.loads(a)["action"] if isinstance(a, str) else a["action"] for a in g["action"]]
        cyc = terminal_cycle(acts)
        rows.append(
            {
                "condition": g["condition"].iloc[0],
                "difficulty": g["difficulty"].iloc[0],
                "seed": int(g["seed"].iloc[0]),
                "success": bool(g["success"].any()),
                "steps": len(acts),
                "cycle": cyc,
                "cycle_frac": cyc / len(acts),
                "last": ",".join(acts[-6:]),
            }
        )
    return pd.DataFrame(rows)


def confound_table(path: Path) -> pd.DataFrame:
    """Per-condition step-level measures. Three independent ways of seeing the same thing:
    if degradation is acting as a randomiser rather than as information loss, the mangled
    conditions get *less* stuck and make *more* progress than the clean one."""
    df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(path / "*.parquet")))])
    g = df.groupby("condition")
    return pd.DataFrame(
        {
            "handoffs": g.size(),
            "stuck_rate": g["stuck"].mean().round(3),
            "mean_progress": g["progress"].mean().round(3),
            "y_binary_rate": g["y_binary_progress"].mean().round(3),
        }
    )


def report(path: Path) -> None:
    ep = episodes(path)
    fail = ep[~ep["success"]]
    print(f"\n=== {path.name}  ({len(ep)} episodes, {len(fail)} failed) ===")
    print(
        f"failed episodes ending in a period-1/2 cycle: "
        f"{(fail['cycle'] > 0).sum()}/{len(fail)} = {(fail['cycle'] > 0).mean():.0%}"
        f" | mean share of steps spent in it: "
        f"{fail.loc[fail['cycle'] > 0, 'cycle_frac'].mean():.0%}"
    )
    by = ep.groupby(["condition", "difficulty"], sort=True)
    tbl = pd.DataFrame(
        {
            "n": by.size(),
            "success": by["success"].mean().round(3),
            "mean_steps": by["steps"].mean().round(1),
            "cycled": by.apply(
                lambda g: (g.loc[~g["success"], "cycle"] > 0).mean(), include_groups=False
            ).round(3),
        }
    )
    print(tbl.to_string())
    print("\nper-condition step-level measures (the confound, if there is one):")
    print(confound_table(path).to_string())
    print("\nfailed episodes, last 6 actions:")
    print(
        fail.sort_values(["condition", "difficulty", "seed"])[
            ["condition", "difficulty", "seed", "steps", "cycle", "last"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["runs"]:
        report(Path(arg))
