"""Tests for the SQL template renderer."""

from __future__ import annotations

from pathlib import Path

import pytest

from hcsim.model.sql import render
from hcsim.model.sql.render import Template, TemplateError, identifier, literal


def test_named_and_optional_slots_fill_in_one_pass() -> None:
    t = Template("SELECT {{ columns }} FROM {{table}} {{where?}}")
    assert t.required == {"columns", "table"}
    assert t.optional == {"where"}
    assert t.slots == {"columns", "table", "where"}
    assert t.render(columns="a", table="t") == "SELECT a FROM t "
    assert t.render({"columns": "a", "table": "t"}, where="WHERE a > 1") == (
        "SELECT a FROM t WHERE a > 1"
    )


def test_a_value_is_never_scanned_again() -> None:
    t = Template("{{a}} and {{b}}")
    assert t.render(a="{{b}}", b="x") == "{{b}} and x"


def test_a_slot_used_twice_takes_one_value() -> None:
    assert Template("{{x}} + {{x}}").render(x="1") == "1 + 1"


def test_rendering_reports_every_problem_at_once() -> None:
    t = Template("{{a}} {{b}} {{c?}}", name="q")
    with pytest.raises(TemplateError, match=r"q: unfilled slots \['a', 'b'\]") as error:
        t.render(d="x", e="y")
    assert "no slots for ['d', 'e']" in str(error.value)
    with pytest.raises(TemplateError, match=r"values must be SQL text: \['a'\]"):
        t.render(a=1, b="x")  # type: ignore[arg-type]


def test_malformed_placeholders_are_refused_by_line() -> None:
    with pytest.raises(TemplateError, match=r"malformed placeholders on lines \[2, 3\]"):
        Template("SELECT 1\nFROM {{ table name }}\nWHERE {{x")
    with pytest.raises(TemplateError, match=r"both required and optional: \['x'\]"):
        Template("{{x}} {{x?}}")


def test_literals_and_identifiers_are_quoted() -> None:
    assert literal("O'Hara") == "'O''Hara'"
    assert literal(3) == "3"
    assert literal(2.5) == "2.5"
    assert literal(True) == "TRUE"
    assert literal(False) == "FALSE"
    assert identifier('a"b') == '"a""b"'


def test_packaged_templates_load_by_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(TemplateError, match="no template named 'nope'"):
        render.load("nope")
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "probe.sql").write_text("SELECT {{x}}")
    monkeypatch.setattr("hcsim.model.sql.render.resources.files", lambda _: tmp_path)
    template = render.load("probe")
    assert (template.name, template.render(x="1")) == ("probe", "SELECT 1")
