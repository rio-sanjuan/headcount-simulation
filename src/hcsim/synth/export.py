"""Parquet export and import: a written world as plain files, and back again.

An export holds one Parquet file per source table and view, the truth schema's tables under
`truth/`, and the manifest as `truth/manifest.json`. Parquet has no views, so a view is
written out as the rows it shows. Rows keep the order the warehouse holds them in.

Importing an export builds a DuckDB warehouse the rest of hcsim reads as it reads a generated
one, with each exported view back as a table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import duckdb

COMPRESSIONS: Final = ("zstd", "snappy", "uncompressed")
MANIFEST: Final = "manifest.json"
TRUTH: Final = "truth"


class ExportError(ValueError):
    """An export or import that can't proceed."""


def _literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def export_parquet(
    con: duckdb.DuckDBPyConnection, directory: Path, compression: str = "zstd"
) -> list[Path]:
    """Write every table, view, and truth table to `directory`, which must be new or empty.
    Returns the files written."""
    if compression not in COMPRESSIONS:
        raise ExportError(f"unknown compression {compression!r}; known: {', '.join(COMPRESSIONS)}")
    if directory.exists() and any(directory.iterdir()):
        raise ExportError(f"{directory} isn't empty: export to a new directory")
    (directory / TRUTH).mkdir(parents=True, exist_ok=True)
    listed = con.execute(
        "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_schema IN ('main', ?) ORDER BY table_schema, table_name",
        [TRUTH],
    ).fetchall()
    written = []
    for schema, name in listed:
        if (schema, name) == (TRUTH, "manifest"):
            row = con.execute(f"SELECT manifest FROM {TRUTH}.manifest").fetchone()
            manifest = row[0] if row else "{}"
            target = directory / TRUTH / MANIFEST
            target.write_text(json.dumps(json.loads(manifest), indent=2, sort_keys=True) + "\n")
        else:
            folder = directory if schema == "main" else directory / TRUTH
            target = folder / f"{name}.parquet"
            con.execute(
                f'COPY (SELECT * FROM {schema}."{name}") TO {_literal(target)} '
                f"(FORMAT parquet, COMPRESSION {compression})"
            )
        written.append(target)
    return written


def import_parquet(directory: Path, path: Path) -> None:
    """Build a fresh DuckDB warehouse at `path` from an export in `directory`."""
    tables = sorted(directory.glob("*.parquet"))
    if not tables:
        raise ExportError(f"no Parquet tables in {directory}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with duckdb.connect(str(path)) as con:
        con.execute(f"CREATE SCHEMA {TRUTH}")
        truth = sorted((directory / TRUTH).glob("*.parquet"))
        for schema, file in [*(("main", f) for f in tables), *((TRUTH, f) for f in truth)]:
            source = f"read_parquet({_literal(file)})"
            con.execute(f'CREATE TABLE {schema}."{file.stem}" AS SELECT * FROM {source}')
        manifest = directory / TRUTH / MANIFEST
        if manifest.is_file():
            con.execute(f"CREATE TABLE {TRUTH}.manifest (manifest JSON)")
            text = json.dumps(json.loads(manifest.read_text()))
            con.execute(f"INSERT INTO {TRUTH}.manifest VALUES (?)", [text])
