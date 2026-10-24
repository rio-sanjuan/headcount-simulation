"""Resolving a study's inputs: their provenance, the kinds it needs, a cache, and a probe.

**Provenance** says where an input or a result comes from: the synthetic world's actuals, a
fixture, or a mix. An input derived from others takes theirs, combined: one kind of source
throughout keeps it, and any two make a mix. A study's results take the combination of all its
inputs', and a result built on any fixture is **provisional**.

**The required kinds** (`required_kinds`) are every input kind a study needs, from its rung of
the run-mode ladder, its forecast scope, its plan, and its evaluators, closed over their
dependencies (see `kinds`):

- every study needs the seed and the fitted models;
- a guided forecast needs a guidance forecast, guided actuals the window's realized totals, and
  replay the realized history;
- a run that computes SBC needs the seed's equity, and one that can score it the actual SBC;
- a plan by path needs it as an artifact; a synthetic plan is built from the seed.

**The extract cache** (`ExtractCache`) keeps an extract's rows on disk, keyed by a SQL digest: of
the source, the SQL rendered for the study's population and contract, and the window. An
extract chunked along time is cached a calendar year at a time, the window's cutoffs splitting
it into pieces that share no outcome, so two studies whose windows overlap read the overlap
once. Every piece is checked against its kind's column contract, read or written.

**The source probe** (`probe`) checks, before anything runs, that a source can serve every kind
a study needs: an extract's SQL is planned against it, and every table another kind reads must
be there. It names every kind the source can't serve, and why.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import duckdb
import pandas as pd

from hcsim.engine.modes import ForecastScope, RunMode, cell
from hcsim.model.data.population import PopulationSpec
from hcsim.studies.kinds import KINDS, Family, InputKind, closure, contract_problems


class Provenance(StrEnum):
    SYNTHETIC_WORLD = "synthetic_world"
    FIXTURE = "fixture"
    MIXED = "mixed"


def combined(provenances: Iterable[Provenance | str]) -> Provenance:
    """One provenance from several: the one they share, or a mix."""
    found = {Provenance(p) for p in provenances}
    if not found:
        raise ValueError("there's no provenance to combine")
    if len(found) == 1:
        (only,) = found
        return only
    return Provenance.MIXED


def provisional(provenance: Provenance | str) -> bool:
    """Whether a result built on inputs of this provenance is provisional: it uses a fixture."""
    return Provenance(provenance) is not Provenance.SYNTHETIC_WORLD


@dataclass(frozen=True)
class Resolved:
    """One resolved input: its kind, its value, and where it comes from."""

    kind: str
    value: object
    provenance: Provenance


def derive(kind: str, value: object, inputs: Sequence[Resolved], own: Provenance) -> Resolved:
    """A derived input, its provenance its own source's combined with its inputs'."""
    return Resolved(kind, value, combined([own, *(r.provenance for r in inputs)]))


class ResolveError(ValueError):
    """Inputs a study can't have, every problem listed."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("can't resolve the study's inputs: " + "; ".join(problems))


def required_kinds(
    mode: RunMode | str,
    scope: ForecastScope | str,
    *,
    plan_path: bool = False,
    evaluator_kinds: Iterable[str] = (),
) -> tuple[str, ...]:
    """Every kind a study needs, dependencies first."""
    place = cell(mode, scope)
    needed = ["seed_state", "fitted_models"]
    if place.rung.reads_guidance:
        needed.append("realized_totals" if place.rung.realized_guidance else "guidance_forecast")
    if place.rung.replays_history:
        needed.append("replay_history")
    if place.computes_sbc:
        needed.append("seed_equity")
    if place.scores_sbc:
        needed.append("actual_sbc")
    if plan_path:
        needed.append("plan")
    needed += list(evaluator_kinds)
    unknown = sorted(set(needed) - set(KINDS))
    if unknown:
        raise ResolveError([f"no input kinds {unknown}"])
    return closure(needed)


# --- the extract cache ------------------------------------------------------------------------


def sql_digest(source: str, sql: str, start: dt.date, end: dt.date) -> str:
    """`x1-` and 16 hex characters of a SHA-256 over the source, the rendered SQL, and the
    window."""
    text = "\n".join((source, sql, start.isoformat(), end.isoformat()))
    return "x1-" + hashlib.sha256(text.encode()).hexdigest()[:16]


def year_pieces(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """(start, end] cut at every year-end inside it."""
    cuts = [dt.date(y, 12, 31) for y in range(start.year, end.year + 1)]
    edges = [start, *(c for c in cuts if start < c < end), end]
    return list(itertools.pairwise(edges))


@dataclass(frozen=True)
class CacheRead:
    """An extract's rows, and how many of its pieces the cache already held."""

    frame: pd.DataFrame
    pieces: int
    hits: int


@dataclass(frozen=True)
class ExtractCache:
    """Extracts kept on disk under `directory`, one parquet file per piece, keyed by SQL
    digest. `source` names the warehouse the rows come from, as a world's identity."""

    directory: Path
    source: str

    def read(
        self,
        con: duckdb.DuckDBPyConnection,
        kind: InputKind,
        population: PopulationSpec,
        cost_center: str,
        start: dt.date,
        end: dt.date,
    ) -> CacheRead:
        """`kind`'s rows over (start, end], each year's piece from the cache where it's held
        and from `con` where it isn't, checked against the kind's contract either way."""
        if kind.family is not Family.EXTRACT or kind.sql is None:
            raise ResolveError([f"{kind.name} isn't an extract, so has no rows to cache"])
        if not start < end:
            raise ResolveError([f"a window needs its start before its end, not {start} to {end}"])
        rendered = kind.sql(population, cost_center)
        self.directory.mkdir(parents=True, exist_ok=True)
        frames, hits = [], 0
        for lo, hi in year_pieces(start, end):
            path = self.directory / f"{kind.name}-{sql_digest(self.source, rendered, lo, hi)}"
            path = path.with_suffix(".parquet")
            if path.is_file():
                frame = pd.read_parquet(path)
                hits += 1
            else:
                frame = con.execute(rendered, {"start": lo, "end": hi}).df()
                frame.to_parquet(path, index=False)
            problems = contract_problems(kind, frame) if len(frame) else []
            if problems:
                raise ResolveError(problems)
            frames.append(frame)
        whole = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        return CacheRead(whole, len(frames), hits)


# --- the source probe -------------------------------------------------------------------------


def probe(
    con: duckdb.DuckDBPyConnection,
    kinds: Iterable[str],
    population: PopulationSpec,
    cost_center: str,
) -> list[str]:
    """Every kind of `kinds` the source can't serve, and why: an extract whose SQL won't plan
    against it, or another kind's tables it lacks."""
    present = {
        str(name).lower()
        for (name,) in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    }
    window = {"start": dt.date(2000, 1, 31), "end": dt.date(2000, 2, 29)}
    problems = []
    for name in kinds:
        kind = KINDS[name]
        if kind.family is Family.EXTRACT and kind.sql is not None:
            try:
                con.execute("EXPLAIN " + kind.sql(population, cost_center), window)
            except duckdb.Error as error:
                first = str(error).splitlines()[0]
                problems.append(f"the source can't serve {name}: {first}")
        missing = sorted(t for t in kind.tables if t not in present)
        if missing:
            problems.append(f"the source can't serve {name}: it has no tables {missing}")
    return problems
