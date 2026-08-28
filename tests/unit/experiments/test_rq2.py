"""DSE-022 RQ2 analysis: the twin agrees, the proxies track, and the leakage null does its job.

Two fixtures carry the suite and they are deliberately opposites.

``_content_records`` puts real per-handoff content in the message ("clear progress"/"we are stuck"),
so a correct analysis must find twin agreement, message content that survives the within-condition
permutation, and an admissible label.

``_tag_records`` puts only the *condition name* in the message. Permuting messages within a
condition is then a literal no-op, so the leakage null equals the real score exactly and every
corrected quantity must come out at 0.0 - the known-answer case for the correction this module
exists to apply, and the case where the honest recommendation is "no label is admissible".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from preceptx.config import ConfigError
from preceptx.data.schema import Condition, HandoffRecord
from preceptx.experiments.rq2 import (
    DECLARED_ORIENTATION,
    PRIMARY_Y,
    RQ2Config,
    _auroc,
    _null_cpvi,
    analyse_rq2,
    write_rq2,
)
from preceptx.gate.statistics import episode_groups
from preceptx.measure.featuriser import EncoderConfig, Featuriser
from preceptx.measure.pvi_cpvi import ProbeConfig, shuffled_message_cpvi

# Small everywhere: the unit tier has a 30s budget and every knob here drives a probe refit.
_FAST = RQ2Config(
    probe=ProbeConfig(n_splits=2, n_repeats=1), n_boot=200, n_boot_track=60, n_shuffle=2
)


class _Encoder:
    """dim0 recovers the progress token; the rest is content-hash noise. ``blind`` sees nothing."""

    def __init__(self, blind: bool = False) -> None:
        self.blind = blind

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float64]:
        rows = []
        for s in sentences:
            if self.blind:  # every text collapses to one vector: no information at all
                rows.append([0.0, 0.0, 0.0, 0.0])
                continue
            flag = 1.0 if "progress" in s else (-1.0 if "stuck" in s else 0.0)
            seed = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
            rows.append([flag, *np.random.default_rng(seed).standard_normal(3).tolist()])
        return np.array(rows, dtype=np.float64)


def _record(cond: str, seed: int, step: int, y: int, msg: str, **over: object) -> HandoffRecord:
    c: Condition = cond  # type: ignore[assignment]
    fields: dict[str, object] = {
        "y_binary_progress": bool(y),
        "y_continuous_displacement": float(y) + 0.1 * step,
        "y_discrete_config": int(y) + (1 if step == 2 else 0),
        "y_terminal_success": bool(y),
    }
    fields.update(over)
    return HandoffRecord(
        episode_id=f"{cond}-s{seed}",
        step=step,
        condition=c,
        serialisation="numeric",
        difficulty="hard",
        model="m",
        seed=seed,
        state={},
        state_str=f"state {cond} s{seed} {step}",
        observation=f"state {cond} s{seed} {step}",  # no outcome token in the state
        message_raw=msg,
        message_delivered=msg,
        action={},
        pre_state={},
        post_state={},
        progress=0.0,
        success=bool(y),
        collision=False,
        stuck=False,
        **fields,  # type: ignore[arg-type]
    )


_PROGRESS_RATE = {"C0": 0.9, "C1": 0.7, "C2": 0.5, "C3": 0.3, "C4": 0.1}


def _records(message: str, n_seeds: int = 6, **over: object) -> list[HandoffRecord]:
    """``message`` is 'content' (the message reveals the handoff) or 'tag' (only the condition)."""
    out: list[HandoffRecord] = []
    for cond, rate in _PROGRESS_RATE.items():
        n = n_seeds * 3
        n_prog = round(rate * n)
        flags = [1] * n_prog + [0] * (n - n_prog)
        h = 0
        for seed in range(n_seeds):
            for step in range(3):
                y = flags[h]
                h += 1
                msg = ("clear progress" if y else "we are stuck") if message == "content" else cond
                out.append(_record(cond, seed, step, y, msg, **over))
    return out


def _content_records(**over: object) -> list[HandoffRecord]:
    return _records("content", **over)


def _tag_records(**over: object) -> list[HandoffRecord]:
    return _records("tag", **over)


def _featuriser(tmp_path: Path, name: str = "e", blind: bool = False) -> Featuriser:
    return Featuriser(EncoderConfig(cache_dir=tmp_path / name), encoder=_Encoder(blind))


def test_analyse_rq2_recovers_agreement_and_message_content(tmp_path: Path) -> None:
    result, frame = analyse_rq2(
        _content_records(), _featuriser(tmp_path), dataset_hash="d0", cfg=_FAST
    )

    pooled = next(t for t in result.twin if t.scope == "pooled").agreement
    assert pooled.spearman_rho > 0.5  # H3: the no-Y twin orders handoffs the way CPVI does
    assert pooled.n == result.n_handoffs
    assert [t.scope for t in result.twin][1:] == ["C0", "C1", "C2", "C3", "C4"]

    primary = next(y for y in result.labels if y.label == PRIMARY_Y)
    assert primary.status == "ok" and primary.admissible
    assert primary.corrected_ci[0] > 0.0  # message content survives the within-condition null
    assert primary.corrected_mean_cpvi < primary.mean_cpvi  # the null is a floor, never zero

    # The score frame is the join key: row-aligned, and the two estimators kept in separate columns.
    assert len(frame) == result.n_handoffs
    assert {"cpvi", "cpvi_shuffled_null", "twin_retrospective", "twin_prospective"} <= set(
        frame.columns
    )


def test_the_leakage_null_matches_the_rq1_permutation_null(tmp_path: Path) -> None:
    """Our per-handoff null must be the same null RQ1 reports as a pooled mean (RD-15)."""
    records = _content_records()
    e_s, e_m = _featuriser(tmp_path).featurise(records)
    y = np.array([int(bool(r.y_binary_progress)) for r in records])
    groups = episode_groups(records)
    conditions = np.array([r.condition for r in records])

    ours = _null_cpvi(e_s, e_m, y, "binary", groups, conditions, _FAST)
    theirs = shuffled_message_cpvi(
        e_s,
        e_m,
        y,
        groups,
        conditions,
        _FAST.probe,
        rng=np.random.default_rng(_FAST.seed),
        n_perm=_FAST.n_shuffle,
    )
    assert np.isclose(float(np.mean(ours)), float(np.mean(theirs)))


def test_a_condition_tag_message_corrects_to_exactly_zero(tmp_path: Path) -> None:
    """The known-answer case: permuting identical-within-condition messages changes nothing."""
    result, _ = analyse_rq2(_tag_records(), _featuriser(tmp_path), dataset_hash="d0", cfg=_FAST)

    for label in result.labels:
        assert label.status == "ok"
        assert label.corrected_mean_cpvi == pytest.approx(0.0, abs=1e-12)
        assert label.corrected_ci == pytest.approx((0.0, 0.0), abs=1e-12)
        assert not label.admissible  # the interval does not exclude zero, so nothing is admissible
    for proxy in result.proxies:
        assert proxy.spearman_corrected == pytest.approx(0.0, abs=1e-12)
        assert proxy.spearman_corrected_ci == pytest.approx((0.0, 0.0), abs=1e-12)
        assert proxy.spearman_cpvi == pytest.approx(proxy.spearman_shuffled)


def test_no_admissible_label_recommends_none(tmp_path: Path) -> None:
    result, _ = analyse_rq2(_tag_records(), _featuriser(tmp_path), dataset_hash="d0", cfg=_FAST)
    assert result.recommended_y is None  # a reportable outcome, not a crash and not a fallback
    assert "No label is admissible" in result.recommendation_note
    assert PRIMARY_Y in result.recommendation_note  # RQ1's frozen label is still named


def test_unscoreable_labels_keep_their_row(tmp_path: Path) -> None:
    records = _content_records(y_discrete_config=None, y_terminal_success=True)
    result, _ = analyse_rq2(records, _featuriser(tmp_path), dataset_hash="d0", cfg=_FAST)

    by_label = {y.label: y for y in result.labels}
    assert len(by_label) == 4  # every candidate keeps a row; none is silently dropped
    assert by_label["y_discrete_config"].status == "unavailable"
    assert by_label["y_terminal_success"].status == "degenerate"
    assert all(by_label[k].reason for k in ("y_discrete_config", "y_terminal_success"))
    assert by_label["y_terminal_success"].mean_cpvi != by_label["y_terminal_success"].mean_cpvi
    assert by_label[PRIMARY_Y].status == "ok"  # the scoreable ones are unaffected


def test_the_orientation_is_declared_not_fitted(tmp_path: Path) -> None:
    """A statistic pointing the wrong way must report AUROC < 0.5, not be flipped to hide it."""
    assert DECLARED_ORIENTATION == {"info": -1.0, "fail": -1.0, "cosine": -1.0}
    inverted = np.array([3.0, 2.0, 1.0, 0.0])
    assert _auroc(np.array([0, 0, 1, 1]), inverted) == pytest.approx(0.0)  # never clamped to >= 0.5
    assert _auroc(np.array([1, 1, 1, 1]), inverted) is None  # single class: None, not 0.5

    result, _ = analyse_rq2(_content_records(), _featuriser(tmp_path), dataset_hash="d0", cfg=_FAST)
    assert {p.key for p in result.proxies} == {"info", "fail", "cosine"}
    assert all(p.declared_orientation == DECLARED_ORIENTATION[p.key] for p in result.proxies)


def test_the_encoder_arm_is_skipped_loudly(tmp_path: Path) -> None:
    skipped, _ = analyse_rq2(
        _content_records(), _featuriser(tmp_path), dataset_hash="d0", cfg=_FAST
    )
    assert skipped.encoders.ran is False
    assert skipped.encoders.rho_primary_second != skipped.encoders.rho_primary_second  # NaN
    assert all(y.encoder_rho is None for y in skipped.labels)
    # Both encoder identities are recorded, so a skipped arm is still told apart from a run one.
    assert skipped.encoders.second_name == EncoderConfig().second_encoder


def test_a_disagreeing_second_encoder_flags_a_re_freeze(tmp_path: Path) -> None:
    result, _ = analyse_rq2(
        _content_records(),
        _featuriser(tmp_path, "e1"),
        dataset_hash="d0",
        second_featuriser=_featuriser(tmp_path, "e2", blind=True),
        cfg=_FAST,
    )
    assert result.encoders.ran is True
    assert result.encoders.rho_primary_second < 0.5
    assert result.encoders.label_ranking_invariant is False
    assert "ENCODER FLAG" in result.recommendation_note
    # Flagged, never performed: the pinned primary is still what the analysis recommends.
    assert result.recommended_encoder == EncoderConfig().name


def test_an_unlabelled_dataset_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=PRIMARY_Y):
        analyse_rq2(
            _content_records(y_binary_progress=None),
            _featuriser(tmp_path),
            dataset_hash="d0",
            cfg=_FAST,
        )
    with pytest.raises(ConfigError, match="both classes"):  # a constant Y fits no probe
        analyse_rq2(
            _content_records(y_binary_progress=True),
            _featuriser(tmp_path),
            dataset_hash="d0",
            cfg=_FAST,
        )
    with pytest.raises(ConfigError, match="no records"):
        analyse_rq2([], _featuriser(tmp_path), dataset_hash="d0", cfg=_FAST)


def test_write_rq2_emits_the_tables_and_the_recommendation(tmp_path: Path) -> None:
    result, frame = analyse_rq2(
        _content_records(), _featuriser(tmp_path), dataset_hash="d0", cfg=_FAST
    )
    out = write_rq2(result, tmp_path / "rq2", scores=frame)

    for name in (
        "rq2.json",
        "rq2_scores.parquet",
        "twin_agreement.csv",
        "proxy_tracking.csv",
        "label_comparison.csv",
        "recommendation.md",
    ):
        assert (out / name).exists(), name
    note = (out / "recommendation.md").read_text()
    assert PRIMARY_Y in note and "Frozen primary Y" in note
