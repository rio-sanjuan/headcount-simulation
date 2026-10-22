"""Tests for the pipeline framework: ordered steps over a shared context."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from hcsim.guidance.config import GuidanceConfig, Reconciliation
from hcsim.guidance.framework import Context, Pipeline, PipelineError
from hcsim.guidance.pipeline import pipeline


@dataclass
class Toy:
    name: str
    needs: tuple[str, ...] = ()
    gives: tuple[str, ...] = ()
    problems: list[str] = field(default_factory=list)
    work: Callable[[Context], Mapping[str, Any]] | None = None
    ran: int = 0

    def validate(self, ctx: Context) -> list[str]:
        return list(self.problems)

    def execute(self, ctx: Context) -> Mapping[str, Any]:
        self.ran += 1
        if self.work is not None:
            return self.work(ctx)
        return {slot: f"{self.name}:{slot}" for slot in self.gives}


def test_steps_run_in_order_over_a_shared_context() -> None:
    double = Toy("double", ("x",), ("y",), work=lambda ctx: {"y": ctx["x"] * 2})
    add = Toy("add", ("x", "y"), ("z",), work=lambda ctx: {"z": ctx["x"] + ctx["y"]})
    ctx = Pipeline((double, add)).run(Context({"x": 3}))
    assert (ctx["y"], ctx["z"]) == (6, 9)
    assert [r.step for r in ctx.records] == ["double", "add"]
    assert ctx.records[1].gave == ("z",)
    assert all(r.seconds >= 0 for r in ctx.records)


def test_a_step_that_needs_and_gives_a_slot_rewrites_it() -> None:
    first = Toy("first", (), ("v",), work=lambda ctx: {"v": 1})
    bump = Toy("bump", ("v",), ("v",), work=lambda ctx: {"v": ctx["v"] + 1})
    assert Pipeline((first, bump)).run(Context())["v"] == 2


def test_building_a_pipeline_refuses_repeated_names_and_writers() -> None:
    with pytest.raises(PipelineError) as error:
        Pipeline((Toy("a", gives=("s",)), Toy("a"), Toy("b", gives=("s",))))
    assert error.value.step is None
    assert error.value.problems == [
        "step names must be unique: ['a'] repeat",
        "'b' gives 's', which 'a' already gives",
    ]


def test_a_run_refuses_slots_nothing_gives_before_any_step_runs() -> None:
    late = Toy("late", (), ("b",))
    early = Toy("early", ("a", "b"), ())
    with pytest.raises(PipelineError, match="'early' needs \\['a', 'b'\\]") as error:
        Pipeline((early, late)).run(Context())
    assert error.value.problems == ["'early' needs ['a', 'b'], which nothing before it gives"]
    assert early.ran == late.ran == 0
    assert Pipeline((early, late)).check(["a", "b"]) == []


def test_a_step_validates_before_it_executes_and_reports_every_problem() -> None:
    first = Toy("first", gives=("a",))
    broken = Toy("broken", ("a",), ("b",), problems=["one thing", "another thing"])
    after = Toy("after", ("b",))
    with pytest.raises(PipelineError) as error:
        Pipeline((first, broken, after)).run(Context())
    assert error.value.step == "broken"
    assert error.value.problems == ["one thing", "another thing"]
    assert "step 'broken' can't run:\n  - one thing\n  - another thing" in str(error.value)
    assert (first.ran, broken.ran, after.ran) == (1, 0, 0)


@pytest.mark.parametrize(
    ("returned", "problem"),
    [
        ({"a": 1, "b": 2}, "returned undeclared slots ['b']"),
        ({}, "didn't return ['a']"),
    ],
)
def test_a_step_must_return_exactly_what_it_gives(returned: dict[str, int], problem: str) -> None:
    step = Toy("step", gives=("a",), work=lambda ctx: returned)
    with pytest.raises(PipelineError) as error:
        Pipeline((step,)).run(Context())
    assert error.value.problems == [problem]
    assert error.value.step == "step"


def test_steps_note_their_report_sections() -> None:
    def work(ctx: Context) -> Mapping[str, Any]:
        ctx.note("noter", seen=1)
        ctx.note("noter", also=2)
        return {}

    ctx = Pipeline((Toy("noter", work=work),)).run(Context())
    assert ctx.report == {"noter": {"seen": 1, "also": 2}}


def test_a_missing_slot_names_what_the_context_holds() -> None:
    with pytest.raises(KeyError, match="no slot 'x' in the context; it holds \\['a'\\]"):
        Context({"a": 1})["x"]


def test_the_lite_pipeline_s_steps() -> None:
    assert pipeline(GuidanceConfig()).names == (
        "ingest",
        "validate",
        "forecast",
        "reconcile",
        "format",
    )
    assert pipeline(GuidanceConfig(reconciliation=Reconciliation(method=None))).names == (
        "ingest",
        "validate",
        "forecast",
        "format",
    )
    assert pipeline(GuidanceConfig()).check(["source", "config"]) == []


def test_the_lite_pipeline_validates_its_source_before_reading_it() -> None:
    with pytest.raises(
        PipelineError, match="a source is a DuckDB connection, a path, or a Source, not str"
    ) as error:
        pipeline(GuidanceConfig()).run(Context({"source": "warehouse", "config": GuidanceConfig()}))
    assert error.value.step == "ingest"


def test_the_lite_pipeline_refuses_a_bad_horizon_before_forecasting() -> None:
    unchecked = GuidanceConfig.model_construct(horizon=0)  # past load-time validation
    ctx = Context({"source": None, "config": unchecked, "series": object()})
    step = pipeline(GuidanceConfig()).steps[2]
    assert step.validate(ctx) == ["the horizon must be at least 1 month, got 0"]
