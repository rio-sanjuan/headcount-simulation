"""Tests for alignment preflight: the methods a run can't honor, and what a run resolved."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from hcsim.engine.alignment import AlignmentMethod, Valve
from hcsim.engine.config import ConfigError, SimulationConfig
from hcsim.engine.settings import from_record, to_record
from hcsim.model.core.guidance import GuidanceForecast
from tests.engine.test_config_records import _saved
from tests.engine.test_config_state_seeds import config as small_config
from tests.engine.test_growth import plan
from tests.engine.test_loop import config, guidance

TARGET = plan({"CC-01": 40.0, "CC-02": 40.0})


def covering(aligned: bool = True, **cells: dict[str, float]) -> GuidanceForecast:
    """Guidance for CC-01 alone, aligned to the plan or not."""
    built = guidance(cells or {"requisitions_opened": {"CC-01": 2.0}})
    return dataclasses.replace(built, aligned_to=TARGET.digest()) if aligned else built


def refused(**overrides: Any) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        config(**overrides)
    return [p for p in caught.value.problems if p.startswith("alignment: ")]


def test_a_method_that_aligns_needs_a_plan_and_aligned_guidance() -> None:
    assert refused(alignment_method="hybrid") == [
        "alignment: alignment_method hybrid aligns to a plan, and the run has none",
        "alignment: alignment_method hybrid drives the plan's flow through guidance aligned to "
        "it, and the run has no guidance",
    ]
    assert refused(alignment_method="demand", plan=TARGET, guidance=covering(False)) == [
        "alignment: alignment_method demand needs guidance aligned to the plan, and the run's "
        "guidance isn't aligned to any"
    ]
    assert refused(alignment_method="rate_calibration") == [
        "alignment: alignment_method rate_calibration aligns to a plan, and the run has none"
    ]


@pytest.mark.parametrize(
    ("method", "drives"),
    [("none", "nothing"), ("supply", "supply"), ("rate_calibration", "the domain rates")],
)
def test_guidance_aligned_to_the_plan_drives_flow_so_another_method_is_refused(
    method: str, drives: str
) -> None:
    assert refused(alignment_method=method, plan=TARGET, guidance=covering()) == [
        "alignment: the guidance is aligned to the plan, so it drives the plan's flow, while "
        f"alignment_method {method} makes the plan drive {drives}: run demand or hybrid, or "
        "guidance that isn't aligned to it"
    ]


def test_rate_calibration_cant_move_counts_guidance_fixes() -> None:
    fixed = covering(False, attrition={"CC-01": 1.0}, external_hires={"CC-01": 1.0})
    assert refused(alignment_method="rate_calibration", plan=TARGET, guidance=fixed) == [
        "alignment: rate calibration shifts the attrition and fill hazards, but guidance fixes "
        "the counts of ['attrition', 'external_hires'], which no shift moves"
    ]
    opened = covering(False)  # requisitions opened isn't a rate the shift moves
    assert config(alignment_method="rate_calibration", plan=TARGET, guidance=opened)


def resolved(method: str, **overrides: Any) -> dict[str, Any]:
    run = config(alignment_method=method, plan=TARGET, **overrides)
    return run.alignment.to_dict()


def test_a_run_records_which_valve_drives_each_cost_centers_target() -> None:
    digest = TARGET.digest()
    supply = resolved("supply")
    assert supply == {
        "method": "supply",
        "plan": digest,
        "aligned_to": None,
        "valves": ["supply"],
        "supply": ["CC-01", "CC-02"],
        "flow": [],
        "rate": [],
        "undriven": [],
    }
    hybrid = resolved("hybrid", guidance=covering())
    assert (hybrid["aligned_to"], hybrid["valves"]) == (digest, ["supply", "flow"])
    assert (hybrid["supply"], hybrid["flow"]) == (["CC-02"], ["CC-01"])
    # Under demand, guidance covers only CC-01, so nothing drives CC-02's target.
    demand = resolved("demand", guidance=covering())
    assert (demand["flow"], demand["undriven"]) == (["CC-01"], ["CC-02"])
    rate = resolved("rate_calibration")
    assert (rate["valves"], rate["rate"]) == (["rate"], ["CC-01", "CC-02"])
    none = resolved("none")
    assert (none["valves"], none["undriven"]) == ([], [])
    assert config().alignment.method is AlignmentMethod.SUPPLY
    assert config().alignment.valves == ()  # no plan, so nothing to drive
    assert config(alignment_method="hybrid", plan=TARGET, guidance=covering()).alignment.valves == (
        Valve.SUPPLY,
        Valve.FLOW,
    )


def test_the_resolution_rides_in_the_configuration_record(tmp_path: Path) -> None:
    run: SimulationConfig = small_config(
        models=_saved(tmp_path), plan=TARGET, alignment_method="rate_calibration"
    )
    record = json.loads(json.dumps(to_record(run)))
    assert record["settings"]["alignment.method"] == "rate_calibration"
    assert record["alignment"]["rate"] == ["CC-01", "CC-02"]
    rebuilt = from_record(
        record, schema=run.schema, models_root=tmp_path, awards=run.awards, plan=TARGET
    )
    assert to_record(rebuilt) == record
