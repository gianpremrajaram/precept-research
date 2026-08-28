from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from preceptx.gate.calibration import StatisticCalibration
from preceptx.gate.integration import GateConfig, RuntimeGate
from preceptx.gate.statistics import CosineStatistic, GateError, InfoStatistic
from preceptx.measure.featuriser import EncoderConfig, Featuriser


class _StubEncoder:
    """Deterministic 2-d embeddings: a text starting with 'a' is [1,0], anything else [0,1].

    Cosine is then exactly 1.0 for a message that echoes its observation's bucket and 0.0 for one
    that does not - so a threshold on the raw cosine is fully controllable from the test's strings.
    """

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float64]:
        return np.array(
            [[1.0, 0.0] if s.startswith("a") else [0.0, 1.0] for s in sentences], dtype=np.float64
        )


def _featuriser(tmp_path: Path) -> Featuriser:
    return Featuriser(EncoderConfig(cache_dir=tmp_path / "embed"), encoder=_StubEncoder())


def _cal(
    key: str = "cosine",
    *,
    threshold: float = 0.5,
    orientation: float = 1.0,
    firing_rate: float = 0.3,
) -> StatisticCalibration:
    return StatisticCalibration(
        key=key,
        threshold=threshold,
        orientation=orientation,
        firing_rate=firing_rate,
        auroc=0.7,
        ece=0.1,
        n_classes=2,
        reliability=[],
    )


def _gate(tmp_path: Path, **cfg: object) -> RuntimeGate:
    return RuntimeGate(
        GateConfig(mode="active", **cfg),  # type: ignore[arg-type]
        statistic=CosineStatistic(),
        calibration=_cal(),
        featuriser=_featuriser(tmp_path),
    )


# --------------------------------------------------------------------------------------------
# Active gate
# --------------------------------------------------------------------------------------------


def test_active_gate_blocks_above_the_threshold_and_passes_below(tmp_path: Path) -> None:
    gate = _gate(tmp_path)
    echo = gate.decide("a-state", "a-message", seed=0, step=0)  # cosine 1.0 >= 0.5
    informative = gate.decide("a-state", "b-message", seed=0, step=0)  # cosine 0.0 < 0.5
    assert echo.blocked and echo.score == pytest.approx(1.0)
    assert not informative.blocked and informative.score == pytest.approx(0.0)
    assert echo.threshold == 0.5


def test_active_gate_applies_the_calibrated_orientation(tmp_path: Path) -> None:
    # Guessing the sign silently inverts every block. A -1 orientation must flip which handoff
    # fires, not merely negate a number nobody reads.
    flipped = RuntimeGate(
        GateConfig(mode="active"),
        statistic=CosineStatistic(),
        calibration=_cal(threshold=-0.5, orientation=-1.0),
        featuriser=_featuriser(tmp_path),
    )
    assert not flipped.decide("a-state", "a-message", seed=0, step=0).blocked  # -1.0 < -0.5
    assert flipped.decide("a-state", "b-message", seed=0, step=0).blocked  # -0.0 >= -0.5


def test_active_gate_is_a_pure_function_of_the_scored_pair(tmp_path: Path) -> None:
    gate = _gate(tmp_path)
    a = gate.decide("a-state", "a-message", seed=0, step=0)
    b = gate.decide("a-state", "a-message", seed=99, step=41)
    assert (a.blocked, a.score) == (b.blocked, b.score)  # seed/step key controls only


# --------------------------------------------------------------------------------------------
# Fail-open (the deliberate inversion of the repo's fail-loud rule)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing", ["statistic", "calibration", "featuriser", "statistic,calibration,featuriser"]
)
def test_active_gate_fails_open_when_the_stack_is_incomplete(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, missing: str
) -> None:
    absent = set(missing.split(","))
    gate = RuntimeGate(
        GateConfig(mode="active"),
        statistic=None if "statistic" in absent else CosineStatistic(),
        calibration=None if "calibration" in absent else _cal(),
        featuriser=None if "featuriser" in absent else _featuriser(tmp_path),
    )
    with caplog.at_level(logging.WARNING):
        decision = gate.decide("a-state", "a-message", seed=0, step=0)  # would otherwise block
    assert not decision.blocked  # open, not closed, and not an exception
    assert decision.score is None
    assert "failing open" in caplog.text


def test_fail_open_warns_once_per_gate_not_once_per_handoff(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gate = RuntimeGate(GateConfig(mode="active"))
    with caplog.at_level(logging.WARNING):
        for step in range(40):
            gate.decide("a-state", "a-message", seed=0, step=step)
    assert len([r for r in caplog.records if "failing open" in r.message]) == 1


def test_matched_random_fails_open_without_a_calibrated_firing_rate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gate = RuntimeGate(GateConfig(mode="matched_random"))
    with caplog.at_level(logging.WARNING):
        assert not gate.decide("a", "a", seed=0, step=0).blocked
    assert "firing rate" in caplog.text


# --------------------------------------------------------------------------------------------
# Mode selection
# --------------------------------------------------------------------------------------------


def test_off_never_blocks_and_never_scores(tmp_path: Path) -> None:
    gate = RuntimeGate(
        GateConfig(mode="off"),
        statistic=CosineStatistic(),
        calibration=_cal(),
        featuriser=_featuriser(tmp_path),
    )
    decision = gate.decide("a-state", "a-message", seed=0, step=0)  # the active gate would block
    assert not decision.blocked and decision.score is None


def test_controls_ignore_the_handoff_and_fire_at_their_own_rate(tmp_path: Path) -> None:
    steps = range(2000)
    matched = RuntimeGate(GateConfig(mode="matched_random"), calibration=_cal(firing_rate=0.3))
    trigger = RuntimeGate(GateConfig(mode="random_trigger", random_rate=0.1))
    # Same echoing pair every step: the active gate would block all of them, the controls do not.
    m = [matched.decide("a-state", "a-message", seed=0, step=s).blocked for s in steps]
    t = [trigger.decide("a-state", "a-message", seed=0, step=s).blocked for s in steps]
    assert abs(np.mean(m) - 0.3) < 0.03  # the gate's calibrated firing rate
    assert abs(np.mean(t) - 0.1) < 0.03  # the configured fixed rate
    assert all(d.score is None for d in [matched.decide("a", "a", seed=0, step=0)])


def test_random_trigger_needs_no_calibration_or_encoder() -> None:
    gate = RuntimeGate(GateConfig(mode="random_trigger", random_rate=1.0))
    assert gate.decide("a", "a", seed=0, step=0).blocked  # no fail-open path taken


# --------------------------------------------------------------------------------------------
# Wiring mistakes fail LOUD, at construction, outside the episode loop
# --------------------------------------------------------------------------------------------


def test_mismatched_statistic_key_raises() -> None:
    with pytest.raises(GateError, match="statistic_key"):
        RuntimeGate(GateConfig(mode="active", statistic_key="cosine"), statistic=InfoStatistic())


def test_mismatched_calibration_key_raises() -> None:
    with pytest.raises(GateError, match="statistic_key"):
        RuntimeGate(GateConfig(mode="active", statistic_key="cosine"), calibration=_cal("info"))


def test_max_retries_is_exposed_to_the_runner() -> None:
    assert RuntimeGate(GateConfig(max_retries=3)).max_retries == 3
    assert GateConfig().max_retries == 1  # one re-prompt by default
