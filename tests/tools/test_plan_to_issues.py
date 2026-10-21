"""Tests for turning the plan into issues. Nothing here calls GitHub."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.plan_to_issues import (
    DEFAULT_PLAN,
    TITLES,
    Issue,
    MilestonePlan,
    build,
    main,
    parse_plan,
    preview,
    short_title,
    slug,
)
from tools.sanitize_scan import parse_patterns, scan_text

Plan = tuple[list[MilestonePlan], list[tuple[str, str]], list[str], list[dict[str, str]]]


@pytest.fixture(scope="module")
def plan() -> Plan:
    return parse_plan(DEFAULT_PLAN.read_text())


def built(plan: Plan) -> list[Issue]:
    milestones, edges, future, questions = plan
    return build(milestones, edges, future, questions, detail={"M0", "M1"})


def test_the_plan_parses_into_ten_milestones_and_102_epics(plan: Plan) -> None:
    milestones, edges, future, questions = plan
    assert [m.key for m in milestones] == [f"M{i}" for i in range(10)]
    assert sum(len(m.epics) for m in milestones) == 102
    assert ("M0", "M1") in edges
    assert len(future) == 4
    assert [q["epic"] for q in questions] == ["M3.11"]


def test_build_matches_what_was_created_on_github(plan: Plan) -> None:
    issues = built(plan)
    kinds = [i.labels[0] for i in issues]
    assert len(issues) == 174
    assert kinds.count("task") == 57
    assert len({i.key for i in issues}) == len(issues)  # plan ids are unique
    assert all(f"<!-- plan-id: {i.key} -->" in i.body for i in issues)


def test_every_title_override_names_a_real_task(plan: Plan) -> None:
    keys = {i.key for i in built(plan)}
    assert set(TITLES) <= keys


def test_a_later_milestone_keeps_its_tasks_as_a_checklist(plan: Plan) -> None:
    epic = next(i for i in built(plan) if i.key == "M3.11")
    assert "<!-- tasks: checklist -->" in epic.body
    assert "- [ ] " in epic.body


@pytest.mark.parametrize(
    ("text", "title"),
    [
        ("Generate a tiny world: about 200 employees.", "Generate a tiny world"),
        ("Add quality gates:", "Add quality gates"),
        (
            "Write `x` (stocks, rates, flows, quotas, the two identities, and metrics, " + "y" * 60,
            "Write `x`",
        ),
    ],
)
def test_short_titles_are_trimmed_cleanly(text: str, title: str) -> None:
    assert short_title(text) == title


def test_slug_matches_github_heading_anchors() -> None:
    assert slug("M1 — Walking skeleton (thin e2e path)") == "m1--walking-skeleton-thin-e2e-path"


def test_preview_lists_every_issue(plan: Plan) -> None:
    issues = built(plan)
    rendered = preview(plan[0], issues)
    assert rendered.count("\n### ") >= len(issues)


def test_the_dry_run_writes_a_preview_and_scans_it(tmp_path: Path) -> None:
    patterns = tmp_path / "patterns.txt"
    patterns.write_text("zebra-canary-\\d+\n")
    out = tmp_path / "preview.md"
    main(["--denylist", str(patterns), "--preview", str(out)])
    text = out.read_text()
    assert text.startswith("# Issue preview")
    assert scan_text(text, "preview", parse_patterns("zebra-canary-\\d+")) == []


def test_a_denylist_match_stops_the_run(tmp_path: Path) -> None:
    patterns = tmp_path / "patterns.txt"
    patterns.write_text("walking skeleton\n")  # a phrase the real plan contains
    with pytest.raises(SystemExit) as stop:
        main(["--denylist", str(patterns)])
    assert stop.value.code == 1
