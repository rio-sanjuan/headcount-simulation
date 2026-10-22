"""Tests for the data profile report."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from hcsim.synth.cli import main
from hcsim.synth.codes import CATALOG
from hcsim.synth.config import QuirkTruth, preset
from hcsim.synth.profile import Profile, _cell, markdown_table, profile, render
from hcsim.synth.world import World, generate_world, write_world


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> tuple[World, Path, Profile]:
    quirks = QuirkTruth(cost_center_aliases=True, aggregate_only_share=0.2, correction_share=0.1)
    world = generate_world(dataclasses.replace(preset("tiny", seed=6), quirks=quirks))
    path = tmp_path_factory.mktemp("world") / "w.duckdb"
    write_world(world, path)
    with duckdb.connect(str(path), read_only=True) as con:
        return world, path, profile(con)


def test_the_profile_lists_every_table_and_its_size(written: tuple[World, Path, Profile]) -> None:
    world, _, result = written
    tables = result.sections["Tables"].set_index("table")
    for name in world.table_names:
        frame = world.table(name)
        assert tables.loc[name, "rows"] == len(frame)
        assert tables.loc[name, "columns"] == len(frame.columns)
    assert tables.loc["req_snapshot", "kind"] == "view"
    truth = set(result.sections["Truth"]["table"])
    assert truth == {f"truth.{name}" for name in ["manifest", *world.truth_tables]}
    assert set(result.columns) == set(world.table_names)  # views have no column profile


def test_the_workforce_section_counts_regular_flows(written: tuple[World, Path, Profile]) -> None:
    world, _, result = written
    txn = world.tables["workforce_txn"].astype({"trans_type": str, "account": str})
    regular = txn[txn["account"] == "ACCT-100"]
    year = pd.to_datetime(regular["event_date"]).dt.year
    by_year = result.sections["Regular workforce by year"].set_index("year")
    assert list(by_year.index) == list(range(2021, 2026))
    code = regular["trans_type"]

    def per_year(codes: list[str]) -> pd.Series:
        return code.isin(codes).astype(int).groupby(year).sum()

    hires = per_year(["HIRE"]) - per_year(["HIRE_REV"])
    departures = per_year(["TERM", "RIF"]) - per_year(["TERM_REV"])
    for y in by_year.index:
        assert by_year.loc[y, "hires"] == hires[y]
        assert by_year.loc[y, "departures"] == departures[y]
    last = pd.to_datetime(regular["event_date"]) == pd.Timestamp("2025-12-31")
    assert by_year.loc[2025, "year_end_headcount"] == int((last & (code == "ACTIVE")).sum())
    assert by_year["attrition_rate"].between(0.02, 0.4).all()


def test_codes_closures_and_checks(written: tuple[World, Path, Profile]) -> None:
    world, _, result = written
    codes = result.sections["Transaction codes"].set_index("trans_type")
    assert set(codes.index) == set(CATALOG)
    assert codes["transaction_rows"].to_dict()["SYS_ADJ"] == 0
    assert codes["aggregate_rows"].to_dict()["SYS_ADJ"] > 0
    assert codes["transaction_rows"].sum() == len(world.tables["workforce_txn"])
    closures = result.sections["Requisition intervals by how they closed"]
    assert closures["intervals"].sum() == len(world.tables["req_interval"])
    checks = result.sections["Coherence checks"]
    assert (checks["status"] == "ok").all()


def test_the_column_profile_follows_the_contract(written: tuple[World, Path, Profile]) -> None:
    _, _, result = written
    reqs = result.columns["req_interval"].set_index("column")
    assert "department" in reqs.index  # the aliased cost center
    assert reqs.loc["department", "missing"] == 0
    assert 0 < reqs["missing"].to_dict()["closed_date"] < 1
    assert reqs.loc["interval_number", "min"] == "1"


def test_the_rendered_tables_are_well_formed(written: tuple[World, Path, Profile]) -> None:
    _, _, result = written
    text = render(result, title="Profile")
    assert text.startswith("# Profile\n")
    for block in re.split(r"\n(?!\|)[^\n]*\n", text):
        rows = [line for line in block.splitlines() if line.startswith("|")]
        widths = {len(re.findall(r"(?<!\\)\|", line)) for line in rows}
        assert len(widths) <= 1, block[:200]
    assert "| 2025 |" in text
    assert "2,025" not in text
    assert "00:00:00" not in text


@pytest.mark.parametrize(
    ("value", "label", "expected"),
    [
        (None, False, ""),
        (float("nan"), False, ""),
        (pd.NaT, False, ""),
        (True, False, "True"),
        (1234, False, "1,234"),
        (2021, True, "2021"),
        (12.0, False, "12"),
        (0.12345, False, "0.1235"),
        (2.5, False, "2.5"),
        (pd.Timestamp("2021-01-31"), False, "2021-01-31"),
        (pd.Timestamp("2021-01-31 12:30"), False, "2021-01-31 12:30:00"),
        ("a|b", False, "a\\|b"),
    ],
)
def test_cells_print_plainly(value: object, label: bool, expected: str) -> None:
    assert _cell(value, label) == expected


def test_an_empty_table_says_so() -> None:
    assert markdown_table(pd.DataFrame({"a": []})) == "_none_"


def test_the_cli_writes_a_profile(
    written: tuple[World, Path, Profile], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, path, _ = written
    out = tmp_path / "reports" / "profile.md"
    assert main(["profile", "--db", str(path), "--out", str(out)]) == 0
    assert out.read_text().startswith("# World profile: w.duckdb\n")
    assert main(["profile", "--db", str(path)]) == 0
    assert "## Coherence checks" in capsys.readouterr().out
    assert main(["profile", "--db", str(tmp_path / "missing.duckdb")]) == 2
