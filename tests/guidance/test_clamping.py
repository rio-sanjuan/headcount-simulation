"""Tests for clamping negative cells when the artifact is written, and counting them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcsim.guidance.config import GuidanceConfig
from hcsim.guidance.framework import Context
from hcsim.guidance.ingest import Series
from hcsim.guidance.pipeline import Format, PipelineResult
from hcsim.guidance.report import markdown
from hcsim.model.core.categories import GuidanceCategory

C = GuidanceCategory


def formatted(forecasts: dict) -> Context:
    months = pd.period_range("2025-01", periods=3, freq="M")
    frame = pd.DataFrame(
        {"month": months, "group": "A", "category": "attrition", "value": [1.0, 2.0, 3.0]}
    )
    series = Series(frame, "cost_center", months[-1], tuple(months))
    ctx = Context(
        {
            "series": series,
            "forecasts": forecasts,
            "paths": {},
            "grain": "cost_center",
            "config": GuidanceConfig(horizon=3),
        }
    )
    ctx.slots |= Format().execute(ctx)
    return ctx


def test_negative_cells_are_written_as_zero_and_counted_by_category_and_group() -> None:
    ctx = formatted(
        {
            ("A", C.ATTRITION): np.array([-1.0, 2.0, -0.5]),
            ("B", C.ATTRITION): np.array([1.0, -3.0, 1.0]),
            ("A", C.EXTERNAL_HIRES): np.array([0.0, 1.0, 2.0]),
        }
    )
    frame = ctx["artifact"].frame
    assert (frame["value"] >= 0).all()
    a = frame[(frame["cost_center"] == "A") & (frame["category"] == "attrition")]
    assert a.sort_values("forecast_month")["value"].tolist() == [0.0, 2.0, 0.0]
    note = ctx.report["format"]
    assert note["clamped_negative_cells"] == 3
    assert note["clamped_cells"] == {"attrition": {"A": 2, "B": 1}}


def test_no_negative_cell_means_nothing_counted() -> None:
    ctx = formatted({("A", C.ATTRITION): np.array([1.0, 2.0, 3.0])})
    assert ctx.report["format"]["clamped_negative_cells"] == 0
    assert ctx.report["format"]["clamped_cells"] == {}


def test_the_run_report_tables_the_clamped_cells() -> None:
    ctx = formatted(
        {
            ("A", C.ATTRITION): np.array([-1.0, 2.0, -0.5]),
            ("B", C.ATTRITION): np.array([1.0, -3.0, 1.0]),
        }
    )
    note = ctx.report["format"]
    report = {
        "clamped_negative_cells": note["clamped_negative_cells"],
        "clamped_cells": note["clamped_cells"],
    }
    text, _ = markdown(PipelineResult(ctx["artifact"], report, ctx["series"]))
    assert "3 negative cells clamped to zero." in text
    rows = [line for line in text.splitlines() if line.startswith("| attrition |")]
    assert rows == ["| attrition | A | 2 |", "| attrition | B | 1 |"]
