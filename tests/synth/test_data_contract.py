"""The data contract and the world agree: every table, column, type, code, and quirk the world
writes is documented in `docs/data-contract.md`, and nothing documented is missing."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import duckdb
import pytest

from hcsim.synth.codes import CATALOG
from hcsim.synth.config import QuirkTruth, preset
from hcsim.synth.world import CONTRACT_VERSION, World, generate_world, write_world

DOC = Path(__file__).resolve().parents[2] / "docs" / "data-contract.md"
#: The contract writes the org hierarchy's levels as one row, since their number varies.
ORG_LEVEL = re.compile(r"org_level\d+")


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text()


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> tuple[World, Path]:
    world = generate_world(preset("small", seed=1))
    path = tmp_path_factory.mktemp("world") / "w.duckdb"
    write_world(world, path)
    return world, path


def _columns(path: Path, schema: str) -> dict[str, list[tuple[str, str]]]:
    with duckdb.connect(str(path), read_only=True) as con:
        rows = con.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = ? ORDER BY table_name, ordinal_position",
            [schema],
        ).fetchall()
    found: dict[str, list[tuple[str, str]]] = {}
    for table, column, kind in rows:
        name = "org_level<k>" if ORG_LEVEL.fullmatch(column) else column
        listed = found.setdefault(table, [])
        if (name, kind) not in listed:
            listed.append((name, kind))
    return found


def _documented_tables(text: str) -> dict[str, list[tuple[str, str]]]:
    """Each `### `table`` section's column rows, as (column, type)."""
    sections = {}
    source = text.split("## Transaction codes")[0]
    for block in re.split(r"^### ", source, flags=re.MULTILINE)[1:]:
        heading, _, body = block.partition("\n")
        match = re.match(r"`(\w+)`", heading)
        assert match, heading
        rows = re.findall(r"^\| `([\w<>]+)` \| `([A-Z]+)` \|", body, flags=re.MULTILINE)
        sections[match.group(1)] = rows
    return sections


def test_the_contract_states_the_version_the_world_records(
    text: str, written: tuple[World, Path]
) -> None:
    assert f"**Contract version: {CONTRACT_VERSION}.**" in text
    assert written[0].truth["contract_version"] == CONTRACT_VERSION


def test_every_source_table_and_column_is_documented(
    text: str, written: tuple[World, Path]
) -> None:
    documented = _documented_tables(text)
    actual = _columns(written[1], "main")
    assert set(documented) == set(actual)
    for table, columns in actual.items():
        assert documented[table] == columns, table


def test_every_truth_table_and_manifest_key_is_documented(
    text: str, written: tuple[World, Path]
) -> None:
    section = text.split("## The truth schema")[1].split("## Quirks")[0]
    rows = re.findall(r"^\| `(\w+)` \| ((?:`\w+`(?:, )?)+) \|", section, flags=re.MULTILINE)
    documented = {table: re.findall(r"`(\w+)`", columns) for table, columns in rows}
    actual = {
        table: [name for name, _ in columns]
        for table, columns in _columns(written[1], "truth").items()
        if table != "manifest"
    }
    assert documented == actual
    for key in written[0].truth:
        assert f"- `{key}`" in section, key


def test_every_transaction_code_is_documented_with_its_disposition(text: str) -> None:
    section = text.split("## Transaction codes")[1].split("## The truth schema")[0]
    rows = re.findall(r"^\| `([A-Z_]+)` \| (\w+) \| (-?\d) \|", section, flags=re.MULTILINE)
    documented = {code: (disposition, int(amount)) for code, disposition, amount in rows}
    assert documented == {code: (d, amount) for code, (d, amount, _) in CATALOG.items()}


def test_every_quirk_toggle_is_documented(text: str) -> None:
    section = text.split("## Quirks")[1].split("## Exports")[0]
    for field in dataclasses.fields(QuirkTruth):
        assert f"`quirks.{field.name}`" in section, field.name
    for toggle in ("openings.bulk_share", "compensation.missing_range_share"):
        assert f"`{toggle}`" in section
