"""Tests for the Parquet export and import."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from hcsim.synth.cli import main
from hcsim.synth.config import QuirkTruth, preset
from hcsim.synth.export import ExportError, export_parquet, import_parquet
from hcsim.synth.validate import validate
from hcsim.synth.world import World, generate_world, write_world


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory: pytest.TempPathFactory) -> tuple[World, Path]:
    quirks = QuirkTruth(snapshot_gap_share=0.2, cost_center_aliases=True)
    world = generate_world(dataclasses.replace(preset("tiny", seed=4), quirks=quirks))
    path = tmp_path_factory.mktemp("world") / "w.duckdb"
    write_world(world, path)
    return world, path


def _contents(path: Path) -> dict[str, pd.DataFrame]:
    """Every table and view, tables in stored order and views sorted, which have none."""
    with duckdb.connect(str(path), read_only=True) as con:
        listed = con.execute(
            "SELECT table_schema, table_name, table_type FROM information_schema.tables "
            "WHERE NOT (table_schema = 'truth' AND table_name = 'manifest')"
        ).fetchall()
        return {
            f"{schema}.{name}": con.execute(
                f'SELECT * FROM {schema}."{name}"' + (" ORDER BY ALL" if kind == "VIEW" else "")
            ).df()
            for schema, name, kind in listed
        }


def test_an_export_round_trips(warehouse: tuple[World, Path], tmp_path: Path) -> None:
    world, path = warehouse
    out = tmp_path / "export"
    with duckdb.connect(str(path), read_only=True) as con:
        files = export_parquet(con, out)
    names = {f.relative_to(out).as_posix() for f in files}
    assert {"workforce_txn.parquet", "equity_expense.parquet", "req_snapshot.parquet"} <= names
    assert {"truth/manifest.json", "truth/latent_plan.parquet"} <= names
    assert len(names) == len(world.table_names) + 1 + len(world.truth_tables) + 1
    manifest = json.loads((out / "truth" / "manifest.json").read_text())
    assert manifest == json.loads(json.dumps(world.truth))

    back = tmp_path / "back.duckdb"
    import_parquet(out, back)
    original, imported = _contents(path), _contents(back)
    assert original.keys() == imported.keys()
    for name, expected in original.items():
        got = imported[name]
        if name == "main.req_snapshot":  # a view comes back as a table, in its own order
            got = got.sort_values(list(got.columns), ignore_index=True)
        pd.testing.assert_frame_equal(got, expected, obj=name)
    with duckdb.connect(str(back), read_only=True) as con:
        assert all(r.passed for r in validate(con))
        (text,) = con.execute("SELECT manifest FROM truth.manifest").fetchone() or ("",)
    assert json.loads(text) == manifest


@pytest.mark.parametrize(
    ("compression", "codec"), [("zstd", "ZSTD"), ("uncompressed", "UNCOMPRESSED")]
)
def test_the_compression_is_chosen(
    warehouse: tuple[World, Path], tmp_path: Path, compression: str, codec: str
) -> None:
    world, path = warehouse
    out = tmp_path / compression
    with duckdb.connect(str(path), read_only=True) as con:
        export_parquet(con, out, compression)
        target = str(out / "workforce_txn.parquet")
        codecs = {
            r[0]
            for r in con.execute(
                "SELECT DISTINCT compression FROM parquet_metadata(?)", [target]
            ).fetchall()
        }
    assert codecs == {codec}
    if compression == "uncompressed":
        # Every label sits in the file as text, where a scan before upload finds it.
        data = (out / "workforce_txn.parquet").read_bytes()
        labels = set(world.tables["workforce_txn"]["cost_center"].astype(str))
        assert all(label.encode() in data for label in labels)


def test_an_export_needs_a_fresh_directory_and_a_known_compression(
    warehouse: tuple[World, Path], tmp_path: Path
) -> None:
    _, path = warehouse
    used = tmp_path / "used"
    used.mkdir()
    (used / "notes.txt").write_text("keep me\n")
    with duckdb.connect(str(path), read_only=True) as con:
        with pytest.raises(ExportError, match="isn't empty"):
            export_parquet(con, used)
        with pytest.raises(ExportError, match="unknown compression 'lz9'"):
            export_parquet(con, tmp_path / "new", "lz9")
    assert (used / "notes.txt").read_text() == "keep me\n"
    with pytest.raises(ExportError, match="no Parquet tables"):
        import_parquet(used, tmp_path / "x.duckdb")


def test_the_cli_exports_and_imports(
    warehouse: tuple[World, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, path = warehouse
    out = tmp_path / "files"
    assert main(["export", "--db", str(path), "--out", str(out), "--compression", "snappy"]) == 0
    assert "files to" in capsys.readouterr().out
    back = tmp_path / "back.duckdb"
    assert main(["import", "--parquet", str(out), "--out", str(back)]) == 0
    assert back.is_file()
    assert main(["export", "--db", str(path), "--out", str(out)]) == 2  # not empty now
    assert main(["export", "--db", str(tmp_path / "missing.duckdb"), "--out", str(out)]) == 2
    assert main(["import", "--parquet", str(tmp_path / "nothing"), "--out", str(back)]) == 2
    assert "isn't empty" in capsys.readouterr().out


def test_an_export_without_a_manifest_imports_without_one(
    warehouse: tuple[World, Path], tmp_path: Path
) -> None:
    _, path = warehouse
    out = tmp_path / "export"
    with duckdb.connect(str(path), read_only=True) as con:
        export_parquet(con, out)
    (out / "truth" / "manifest.json").rename(tmp_path / "set-aside.json")
    back = tmp_path / "back.duckdb"
    import_parquet(out, back)
    with duckdb.connect(str(back), read_only=True) as con:
        truth = {
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'truth'"
            ).fetchall()
        }
    assert "manifest" not in truth
    assert "latent_plan" in truth
