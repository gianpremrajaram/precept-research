"""Consistent figure styling for every RQ analysis (DSE-028).

One house style and one reusable interval plot so all RQ figures look the same and read the same
(means with uncertainty intervals, never bare points). Matplotlib is the optional ``viz`` extra, so
every entry point is guarded exactly like the calibration figure: absent the extra, the JSON/table
artefacts remain load-bearing and figure calls no-op with a log line. Nothing here knows about a
specific RQ - the drivers pass labels, means and intervals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RC = {  # the house style, applied per-figure so we never mutate global rcParams
    "figure.figsize": (6.0, 4.0),
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _pyplot() -> Any | None:
    """Return ``matplotlib.pyplot`` (Agg backend) or ``None`` when the viz extra is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib absent (install the 'viz' extra) - skipping figure")
        return None
    return plt


def ci_plot(
    labels: list[str],
    means: list[float],
    cis: list[tuple[float, float]],
    *,
    ylabel: str,
    title: str,
    path: Path,
) -> Path | None:
    """Point-with-interval plot over ordered ``labels`` (e.g. conditions). No-op without viz.

    ``cis`` are absolute (lo, hi) bounds row-aligned to ``means``; drawn as asymmetric error bars so
    bootstrap intervals are shown faithfully rather than as a symmetric +/- guess.
    """
    plt = _pyplot()
    if plt is None:
        return None
    x = list(range(len(labels)))
    lower = [m - lo for m, (lo, _hi) in zip(means, cis, strict=True)]
    upper = [hi - m for m, (_lo, hi) in zip(means, cis, strict=True)]
    with plt.rc_context(_RC):
        fig, ax = plt.subplots()
        ax.errorbar(x, means, yerr=[lower, upper], marker="o", capsize=4, linestyle="-")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path
