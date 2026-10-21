"""Tests for the entity models."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.model.interface import EntityModel
from hcsim.model.ml.entity import ConstantModel, ModelNotFittedError, NaiveRateModel

LEVEL, CC = 0, 1


def rows(*pairs: tuple[int, int]) -> np.ndarray:
    return np.array(pairs, dtype=np.float64)


def test_constant_model_gives_every_row_one_probability() -> None:
    model = ConstantModel(0.25)
    assert isinstance(model, EntityModel)
    assert model.compute_probability(np.zeros((3, 5))).tolist() == [0.25] * 3
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ConstantModel(1.5)


def training() -> tuple[np.ndarray, np.ndarray]:
    # Level 0 in cost center 0: 40 rows, 10 events. Level 0 in cost center 1: 5 rows, 5 events.
    # Level 1 in cost center 0: 40 rows, 0 events.
    features = rows(*([(0, 0)] * 40 + [(0, 1)] * 5 + [(1, 0)] * 40))
    labels = np.array([1] * 10 + [0] * 30 + [1] * 5 + [0] * 40, dtype=np.float64)
    return features, labels


def test_each_row_reads_the_longest_prefix_with_enough_rows() -> None:
    model = NaiveRateModel([LEVEL, CC], min_population=30).fit(*training())
    assert isinstance(model, EntityModel)
    p = model.compute_probability(rows((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)))
    assert p[0] == pytest.approx(10 / 40)  # its full cell has 40 rows
    assert p[1] == pytest.approx(15 / 45)  # its cell has 5 rows: falls back to level 0
    assert p[2] == pytest.approx(0.0)
    assert p[3] == pytest.approx(0.0)  # unseen cell: falls back to level 1
    assert p[4] == pytest.approx(15 / 85)  # unseen level: falls back to the global rate


def test_a_lower_minimum_lets_small_cells_speak() -> None:
    model = NaiveRateModel([LEVEL, CC], min_population=5).fit(*training())
    assert model.compute_probability(rows((0, 1)))[0] == pytest.approx(1.0)


def test_no_key_columns_means_the_global_rate() -> None:
    model = NaiveRateModel([]).fit(*training())
    assert model.compute_probability(rows((0, 0)))[0] == pytest.approx(15 / 85)


def test_scoring_before_fitting_is_refused() -> None:
    with pytest.raises(ModelNotFittedError):
        NaiveRateModel([LEVEL]).compute_probability(rows((0, 0)))
    with pytest.raises(ModelNotFittedError):
        _ = NaiveRateModel([LEVEL]).global_rate


@pytest.mark.parametrize(
    ("labels", "match"),
    [(np.array([0.0, 2.0]), "0 or 1"), (np.array([1.0]), "same number of rows")],
)
def test_bad_labels_are_refused(labels: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        NaiveRateModel([LEVEL]).fit(rows((0, 0), (0, 1)), labels)


def test_fitting_on_zero_rows_is_refused() -> None:
    with pytest.raises(ValueError, match="zero rows"):
        NaiveRateModel([LEVEL]).fit(np.zeros((0, 2)), np.zeros(0))
    with pytest.raises(ValueError, match="at least 1"):
        NaiveRateModel([LEVEL], min_population=0)
