"""Tests for the entity classifiers and logit-offset calibration."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.model.interface import EntityModel
from hcsim.model.ml.classifiers import (
    BoostedModel,
    FeatureMask,
    LevelOnlyModel,
    LogisticModel,
    ModelError,
    calibrated,
    logit_offset,
)
from hcsim.model.ml.entity import ModelNotFittedError

LEVEL, FAMILY, TENURE, NOISE = 0, 1, 2, 3


def world(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Departures that rise with level, differ by family, and peak at mid tenure."""
    rng = np.random.default_rng(seed)
    x = np.column_stack(
        [
            rng.integers(0, 5, n),
            rng.integers(0, 4, n),
            rng.uniform(0, 120, n),
            rng.normal(size=n),
        ]
    ).astype(np.float64)
    family = np.array([0.0, 0.6, -0.4, 0.2])[x[:, FAMILY].astype(int)]
    tenure = 1.2 * np.exp(-(((x[:, TENURE] - 24) / 12) ** 2))
    p = 1 / (1 + np.exp(-(-3.0 + 0.3 * x[:, LEVEL] + family + tenure)))
    return x, (rng.random(n) < p).astype(np.float64), p


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


MASK = FeatureMask((LEVEL, FAMILY, TENURE), categorical=(FAMILY,))


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, ...]:
    x, y, _ = world(20_000, seed=1)
    held_x, held_y, held_p = world(20_000, seed=2)
    return x, y, held_x, held_y, held_p


def test_richer_models_beat_the_level_baseline_on_held_out_rows(
    data: tuple[np.ndarray, ...],
) -> None:
    x, y, hx, hy, hp = data
    baseline = LevelOnlyModel(LEVEL).fit(x, y)
    logistic = LogisticModel(MASK).fit(x, y)
    boosted = BoostedModel(MASK).fit(x, y)
    for model in (baseline, logistic, boosted):
        assert isinstance(model, EntityModel)
    losses = {
        m.ALGORITHM_NAME: log_loss(m.compute_probability(hx), hy)
        for m in (baseline, logistic, boosted)
    }
    assert losses["boosted"] < losses["logistic"] < losses["level_only"]
    assert losses["boosted"] - log_loss(hp, hy) < 0.01  # close to the truth's own loss


def test_masked_out_columns_never_move_a_prediction(data: tuple[np.ndarray, ...]) -> None:
    x, y, hx, _, _ = data
    shuffled = hx.copy()
    shuffled[:, NOISE] = np.random.default_rng(3).permutation(shuffled[:, NOISE])
    for model in (LogisticModel(MASK).fit(x, y), BoostedModel(MASK, max_iter=20).fit(x, y)):
        np.testing.assert_array_equal(
            model.compute_probability(hx), model.compute_probability(shuffled)
        )


def test_an_unattributed_category_reads_as_missing(data: tuple[np.ndarray, ...]) -> None:
    x, y, hx, _, _ = data
    unknown = hx[:5].copy()
    unknown[:, FAMILY] = -1
    for model in (LogisticModel(MASK).fit(x, y), BoostedModel(MASK, max_iter=20).fit(x, y)):
        p = model.compute_probability(unknown)
        assert np.isfinite(p).all()
        assert ((p > 0) & (p < 1)).all()


def test_the_logit_offset_hits_its_rate_and_keeps_the_ranking(data: tuple[np.ndarray, ...]) -> None:
    x, y, hx, _, _ = data
    boosted = BoostedModel(MASK, max_iter=20).fit(x, y)
    raw = boosted.raw_probability(hx)
    boosted.calibrate(hx, 0.02)
    shifted = boosted.compute_probability(hx)
    assert shifted.mean() == pytest.approx(0.02, rel=1e-6)
    assert (np.argsort(raw) == np.argsort(shifted)).all()
    assert boosted.offset < 0
    assert calibrated([0.5], 0.0).tolist() == pytest.approx([0.5])
    assert logit_offset([0.2, 0.4], 0.3) == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: FeatureMask(()), ModelError, "at least one feature"),
        (lambda: FeatureMask((0, 0)), ModelError, "repeat"),
        (lambda: FeatureMask((0,), categorical=(1,)), ModelError, "aren't among"),
        (
            lambda: LogisticModel(MASK).fit(np.zeros((2, 4)), np.array([0.0, 2.0])),
            ModelError,
            "0 or 1",
        ),
        (
            lambda: LogisticModel(MASK).fit(np.zeros((2, 4)), np.array([1.0, 1.0])),
            ModelError,
            "both outcomes",
        ),
        (lambda: LogisticModel(MASK).fit(np.zeros((0, 4)), np.zeros(0)), ModelError, "zero rows"),
        (
            lambda: LogisticModel(MASK).fit(np.zeros((2, 4)), np.zeros(3)),
            ModelError,
            "one entry per row",
        ),
        (
            lambda: LogisticModel(MASK).fit(np.zeros((2, 2)), np.array([0.0, 1.0])),
            ModelError,
            "features have 2 columns",
        ),
        (
            lambda: LogisticModel(MASK).fit(np.zeros(4), np.array([0.0])),
            ModelError,
            "\\(rows, features\\)",
        ),
        (
            lambda: LogisticModel(MASK).compute_probability(np.zeros((1, 4))),
            ModelNotFittedError,
            "fit the model",
        ),
        (
            lambda: BoostedModel(MASK).raw_probability(np.zeros((1, 4))),
            ModelNotFittedError,
            "fit the model",
        ),
        (lambda: logit_offset([], 0.1), ModelError, "at least one row"),
        (lambda: logit_offset([0.5], 1.0), ModelError, "strictly between"),
    ],
)
def test_bad_inputs_are_refused(call, error: type[Exception], message: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(error, match=message):
        call()


def test_a_booster_refuses_too_many_categories() -> None:
    x = np.column_stack([np.arange(400) % 300, np.arange(400) % 2]).astype(np.float64)
    with pytest.raises(ModelError, match="codes past 253"):
        BoostedModel(FeatureMask((0,), categorical=(0,))).fit(x, x[:, 1])
