"""Sources: where the guidance pipeline reads its warehouse tables from.

Ingestion reads the same tables whatever holds them, through one DuckDB connection:

- `DuckDBSource`: a DuckDB file, opened read-only, or a connection already open;
- `FileSource`: a directory of table files, one per table, named `<table>.parquet` or
  `<table>.csv`, as a world's Parquet export writes them. Each file becomes a view;
- `FrameSource`: tables held in memory as pandas frames, each registered as a view.

Every source must hold the tables ingestion reads (`TABLES`), and the data contract's column
table when it has one. A source missing any of them is refused before anything is counted,
with every missing table named.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol

import duckdb
import pandas as pd

#: The tables ingestion reads.
TABLES: Final = ("workforce_agg", "dim_trans_type", "dim_account", "dim_org", "req_interval")
#: The file formats a `FileSource` reads, by suffix.
FORMATS: Final = {".parquet": "read_parquet", ".csv": "read_csv_auto"}


class SourceError(ValueError):
    """A source that can't supply the tables ingestion reads."""


class Source(Protocol):
    @property
    def description(self) -> str:
        """What the source is, for the run report."""
        ...

    def connect(self) -> AbstractContextManager[duckdb.DuckDBPyConnection]: ...


def missing_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    """The tables ingestion reads that `con` doesn't hold."""
    held = {
        row[0] for row in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    }
    return [t for t in TABLES if t not in held]


def _literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


@dataclass(frozen=True)
class DuckDBSource:
    path: Path | None = None
    connection: duckdb.DuckDBPyConnection | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if (self.path is None) == (self.connection is None):
            raise SourceError("a DuckDB source takes a path or a connection, not both or neither")

    @property
    def description(self) -> str:
        return f"duckdb:{self.path}" if self.path is not None else "duckdb:connection"

    @contextmanager
    def connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        if self.connection is not None:
            yield self.connection  # the caller's to close
            return
        assert self.path is not None
        if not self.path.is_file():
            raise SourceError(f"no DuckDB file at {self.path}")
        with duckdb.connect(str(self.path), read_only=True) as con:
            yield con


@dataclass(frozen=True)
class FileSource:
    directory: Path

    @property
    def description(self) -> str:
        return f"files:{self.directory}"

    def files(self) -> dict[str, Path]:
        """Each table's file, by table name."""
        if not self.directory.is_dir():
            raise SourceError(f"no directory at {self.directory}")
        found: dict[str, Path] = {}
        for path in sorted(self.directory.iterdir()):
            if path.suffix in FORMATS:
                if path.stem in found:
                    raise SourceError(f"two files for the table {path.stem!r} in {self.directory}")
                found[path.stem] = path
        return found

    @contextmanager
    def connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        files = self.files()
        with duckdb.connect() as con:
            for table, path in files.items():
                reader = FORMATS[path.suffix]
                con.execute(
                    f'CREATE VIEW "{table}" AS SELECT * FROM {reader}({_literal(str(path))})'
                )
            yield con


@dataclass(frozen=True)
class FrameSource:
    tables: Mapping[str, pd.DataFrame] = field(compare=False)

    @property
    def description(self) -> str:
        return "frames:" + ",".join(sorted(self.tables))

    @contextmanager
    def connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        with duckdb.connect() as con:
            for name, frame in self.tables.items():
                con.register(name, frame)
            yield con


def as_source(source: object) -> Source:
    """A source, a connection, or a path as a source: a path names a DuckDB file, or a
    directory of table files."""
    if isinstance(source, DuckDBSource | FileSource | FrameSource):
        return source
    if isinstance(source, duckdb.DuckDBPyConnection):
        return DuckDBSource(connection=source)
    if isinstance(source, Path):
        return FileSource(source) if source.is_dir() else DuckDBSource(source)
    raise SourceError(
        f"a source is a DuckDB connection, a path, or a Source, not {type(source).__name__}"
    )
