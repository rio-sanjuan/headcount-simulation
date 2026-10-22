"""Tests for eligibility rules, their resolution, and the production defaults."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hcsim.engine.config import ConfigError
from hcsim.engine.eligibility import (
    DOMAIN_LAYOUTS,
    EligibilityError,
    EligibilityRule,
    resolve,
)
from hcsim.engine.settings import (
    PRODUCTION,
    from_record,
    load_settings,
    production_config,
    to_record,
)
from hcsim.model.core.layouts import EMPLOYEE, OPENING, EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import (
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    LOCATION,
    SimulationSchema,
)
from tests.engine.test_config_records import _saved
from tests.engine.test_config_state_seeds import awards, config, models

SCHEMA = SimulationSchema.build(
    {
        COST_CENTER: ["CC-01", "CC-02", "CC-03"],
        JOB_FAMILY: ["Data", "Sales"],
        JOB_LEVEL: [1, 7, 8],
        LOCATION: ["L"],
    },
    org_levels={"org_level1": {"CC-01": "ORG-A", "CC-02": "ORG-A", "CC-03": "ORG-B"}},
)


def _employees(levels: list[int], cost_centers: list[int]) -> np.ndarray:
    rows = EMPLOYEE.empty(len(levels))
    rows[:, EmployeeColumn.LEVEL] = [SCHEMA.vocabulary(JOB_LEVEL).code(v) for v in levels]
    rows[:, EmployeeColumn.COST_CENTER] = cost_centers
    return rows


def test_a_rule_excludes_or_admits_only_the_rows_it_selects() -> None:
    eligibility = resolve(
        [
            EligibilityRule.of("attrition", job_level=[8], reason="executives leave by plan"),
            EligibilityRule.of("promotion", exclude=False, org_level1=["ORG-A"]),
        ],
        SCHEMA,
    )
    rows = _employees([1, 8, 7, 8], [0, 1, 2, 2])
    assert eligibility.eligible("attrition", rows).tolist() == [True, False, True, False]
    # Only the cost centers under ORG-A may promote.
    assert eligibility.eligible("promotion", rows).tolist() == [True, True, False, False]
    # A domain with no rules allows everything.
    assert eligibility.eligible("transfers", rows).all()
    p = np.full(4, 0.5)
    np.testing.assert_array_equal(eligibility.restrict("attrition", rows, p), [0.5, 0, 0.5, 0])
    assert eligibility.restrict("transfers", rows, p) is p
    assert [r.reason for r in eligibility.for_domain("attrition")] == ["executives leave by plan"]


def test_rules_on_one_domain_combine() -> None:
    eligibility = resolve(
        [
            EligibilityRule.of("attrition", job_level=[8]),
            EligibilityRule.of("attrition", exclude=False, cost_center=["CC-01", "CC-03"]),
        ],
        SCHEMA,
    )
    rows = _employees([1, 1, 8, 7], [0, 1, 2, 2])
    assert eligibility.eligible("attrition", rows).tolist() == [True, False, False, True]


def test_requisition_domains_select_openings() -> None:
    eligibility = resolve([EligibilityRule.of("cancellation", job_family=["Sales"])], SCHEMA)
    reqs = OPENING.empty(2)
    reqs[:, OpeningColumn.JOB_FAMILY] = [0, 1]
    assert eligibility.eligible("cancellation", reqs).tolist() == [True, False]
    assert DOMAIN_LAYOUTS["external_fill"] is OPENING


def test_one_declaration_means_the_same_under_any_vocabulary() -> None:
    rule = EligibilityRule.of("attrition", job_level=[8])
    wider = SimulationSchema.build({COST_CENTER: ["CC-01"], JOB_LEVEL: [1, 2, 3, 8, 9]})
    for schema in (SCHEMA, wider):
        rows = EMPLOYEE.empty(1)
        rows[:, EmployeeColumn.LEVEL] = schema.vocabulary(JOB_LEVEL).code(8)
        assert not resolve([rule], schema).eligible("attrition", rows)[0]


def test_every_problem_in_every_rule_is_reported() -> None:
    rules = [
        EligibilityRule.of("growth", job_level=[1]),
        EligibilityRule.of("cancellation", perf_rating=[1]),
        EligibilityRule.of("attrition", job_level=[99], org_level9=["X"]),
    ]
    with pytest.raises(EligibilityError) as caught:
        resolve(rules, SCHEMA)
    problems = caught.value.problems
    assert problems[0].startswith("rule 1 (growth): no domain 'growth'")
    assert problems[1] == "rule 2 (cancellation): requisitions have no rating to select by"
    assert "no rollup 'org_level9'" in problems[2]
    assert "job_level has no labels ['99']" in problems[2]
    # The configuration reports them among its own.
    with pytest.raises(ConfigError, match="eligibility: rule 1"):
        config(eligibility=rules[:1])


def test_a_rule_round_trips_through_plain_data(tmp_path: Path) -> None:
    rule = EligibilityRule.of("attrition", job_level=[1], reason="by plan")
    assert EligibilityRule.from_dict(json.loads(json.dumps(rule.to_dict()))) == rule
    with pytest.raises(EligibilityError, match="can't read rule"):
        EligibilityRule.from_dict({"domain": "attrition"})
    with pytest.raises(EligibilityError, match="can't read rule"):
        EligibilityRule.from_dict(
            {"domain": "x", "where": {"weather": ["rain"]}, "exclude": True, "reason": ""}
        )
    # A configuration's record carries its rules in raw labels.
    built = config(models=_saved(tmp_path), eligibility=(rule,))
    record = json.loads(json.dumps(to_record(built)))
    assert record["eligibility"] == [rule.to_dict()]
    rebuilt = from_record(record, schema=built.schema, models_root=tmp_path, awards=awards())
    assert rebuilt.eligibility == (rule,)
    record["eligibility"] = [{"domain": "attrition"}]
    with pytest.raises(ConfigError, match="can't read rule"):
        from_record(record, schema=built.schema, models_root=tmp_path, awards=awards())


def test_production_defaults_sit_between_the_defaults_and_yaml(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("run:\n  num_trajectories: 50\n")
    layered = load_settings(
        yaml_path=path, env={}, overrides={"run.run_date": "2026-01-01"}, production=True
    )
    assert layered.values["simulation_months"] == PRODUCTION["run.simulation_months"]
    assert layered.sources["simulation_months"] == "production"
    assert (layered.values["num_trajectories"], layered.sources["num_trajectories"]) == (50, "yaml")
    built = production_config(
        schema=SCHEMA,
        models=models(),
        awards=None,
        env={},
        overrides={"run.run_date": "2026-01-01", "run.forecast_scope": "hcf"},
        eligibility=[EligibilityRule.of("attrition", job_level=[8])],
    )
    assert (built.simulation_months, built.num_trajectories) == (24, 200)
    assert built.results_streaming
    assert built.resolved_eligibility.for_domain("attrition")
