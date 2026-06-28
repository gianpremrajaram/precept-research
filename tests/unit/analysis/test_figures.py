"""DSE-028 figure styling: writes a PNG with the viz extra, no-ops cleanly without it."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from preceptx.analysis.figures import ci_plot

_HAS_MPL = importlib.util.find_spec("matplotlib") is not None


def test_ci_plot_noop_without_viz(tmp_path: Path) -> None:
    if _HAS_MPL:
        pytest.skip("matplotlib present; the no-op branch is unreachable")
    out = ci_plot(["C0"], [0.5], [(0.4, 0.6)], ylabel="y", title="t", path=tmp_path / "f.png")
    assert out is None and not (tmp_path / "f.png").exists()


def test_ci_plot_writes_png_with_viz(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = ci_plot(
        ["C0", "C4"],
        [0.8, 0.2],
        [(0.7, 0.9), (0.1, 0.3)],
        ylabel="success",
        title="outcome vs condition",
        path=tmp_path / "f.png",
    )
    assert out is not None and out.exists()
