from __future__ import annotations

import inspect

import numpy as np
import pytest
from numpy.typing import NDArray

from preceptx.config import ConfigError
from preceptx.data.schema import HandoffRecord
from preceptx.gate.statistics import (
    CosineStatistic,
    FailStatistic,
    GateError,
    InfoStatistic,
    episode_groups,
    failure_label,
    load_statistic,
    save_statistic,
    score_records,
)
from preceptx.measure.featuriser import EncoderConfig, Featuriser

Tup = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]]


def _separable(n: int = 200, d: int = 8, seed: int = 0) -> Tup:
    """Embeddings where the label is linearly separable in [e_s;e_m]; returns (e_s, e_m, y)."""
    rng = np.random.default_rng(seed)
    e_s = rng.standard_normal((n, d))
    e_m = rng.standard_normal((n, d))
    logit = e_s @ rng.standard_normal(d) + e_m @ rng.standard_normal(d)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return e_s.astype(np.float64), e_m.astype(np.float64), y


def _rec(ep: str, success: bool | None) -> HandoffRecord:
    return HandoffRecord(
        episode_id=ep,
        step=0,
        condition="C0",
        serialisation="numeric",
        difficulty="easy",
        model="m",
        seed=0,
        state={},
        state_str=f"state-{ep}",
        message_raw="r",
        message_delivered=f"msg-{ep}",
        action={},
        pre_state={},
        post_state={},
        progress=0.0,
        success=bool(success),
        collision=False,
        stuck=False,
        y_terminal_success=success,
    )


class _StubEncoder:
    """Torch-free encoder: a 2-D vector that varies with text length and a stable char sum."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> NDArray[np.float64]:
        rows = [[float(len(s)), float(sum(ord(c) for c in s) % 11)] for s in sentences]
        return np.array(rows, dtype=np.float64)


def test_cosine_in_range_and_probe_independent() -> None:
    rng = np.random.default_rng(1)
    e_s = rng.standard_normal((50, 8))
    e_m = rng.standard_normal((50, 8))
    stat = CosineStatistic()
    before = stat.score(e_s, e_m)
    assert before.shape == (50,)
    assert np.all(before >= -1.0001) and np.all(before <= 1.0001)
    stat.fit(e_s, e_m, np.zeros(50, dtype=int))  # fitting with any label changes nothing
    assert np.array_equal(before, stat.score(e_s, e_m))


def test_cosine_zero_norm_is_defined_not_nan() -> None:
    sc = CosineStatistic().score(np.zeros((3, 4)), np.ones((3, 4)))
    assert np.all(np.isfinite(sc)) and np.all(sc == 0.0)


def test_info_entropy_bounded_and_lower_when_predictable() -> None:
    e_s, e_m, y = _separable(seed=2)
    info = InfoStatistic()
    info.fit(e_s, e_m, y)
    h = info.score(e_s, e_m)
    assert np.all(h >= -1e-9) and np.all(h <= 1.0 + 1e-9)  # binary outcome -> [0, 1] bits
    rng = np.random.default_rng(3)
    noise = rng.standard_normal((len(y), 8))
    noisy = InfoStatistic()
    noisy.fit(noise, noise, (rng.random(len(y)) < 0.5).astype(int))
    assert float(np.mean(h)) < float(np.mean(noisy.score(noise, noise)))  # confident -> low entropy


def test_fail_outputs_probability_and_learns() -> None:
    e_s, e_m, fail = _separable(seed=4)
    stat = FailStatistic()
    stat.fit(e_s, e_m, fail)
    p = stat.score(e_s, e_m)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    assert float(p[fail == 1].mean()) > float(p[fail == 0].mean())  # higher P(fail) on the failures


def test_no_outcome_access_at_inference_signature() -> None:
    for cls in (InfoStatistic, FailStatistic, CosineStatistic):
        params = list(inspect.signature(cls.score).parameters)
        assert params == ["self", "e_s", "e_m"]  # no outcome can enter at score time


def test_score_before_fit_raises() -> None:
    e = np.zeros((2, 4))
    with pytest.raises(GateError):
        InfoStatistic().score(e, e)
    with pytest.raises(GateError):
        FailStatistic().score(e, e)


def test_single_class_fit_is_degenerate_not_crash() -> None:
    e_s, e_m, _ = _separable(seed=6)
    zeros = np.zeros(len(e_s), dtype=int)
    info = InfoStatistic()
    info.fit(e_s, e_m, zeros)
    assert np.allclose(info.score(e_s, e_m), 0.0)  # one class -> entropy 0
    fail0 = FailStatistic()
    fail0.fit(e_s, e_m, zeros)
    assert np.allclose(fail0.score(e_s, e_m), 0.0)  # base rate 0
    fail1 = FailStatistic()
    fail1.fit(e_s, e_m, np.ones(len(e_s), dtype=int))
    assert np.allclose(fail1.score(e_s, e_m), 1.0)  # base rate 1


def test_failure_label_and_none_guard() -> None:
    assert list(failure_label([_rec("a", True), _rec("a", False)])) == [0, 1]
    with pytest.raises(ConfigError):
        failure_label([_rec("a", None)])


def test_episode_groups_factorise_ids() -> None:
    recs = [_rec("a", True), _rec("a", True), _rec("b", True)]
    assert list(episode_groups(recs)) == [0, 0, 1]


def test_score_records_returns_scores_and_groups(tmp_path) -> None:  # type: ignore[no-untyped-def]
    recs = [_rec("a", True), _rec("a", False), _rec("b", True)]
    feat = Featuriser(EncoderConfig(cache_dir=tmp_path / "c"), encoder=_StubEncoder())
    scores, groups = score_records(CosineStatistic(), recs, feat)
    assert scores.shape == (3,) and list(groups) == [0, 0, 1]


def test_save_load_round_trip_and_key_guard(tmp_path) -> None:  # type: ignore[no-untyped-def]
    e_s, e_m, fail = _separable(seed=7)
    stat = FailStatistic()
    stat.fit(e_s, e_m, fail)
    save_statistic(stat, encoder=EncoderConfig(), train_dataset_hash="h", dir=tmp_path / "s")
    loaded = load_statistic("fail", dir=tmp_path / "s")
    assert np.allclose(loaded.score(e_s, e_m), stat.score(e_s, e_m))
    with pytest.raises(GateError):
        load_statistic("info", dir=tmp_path / "s")  # never persisted
