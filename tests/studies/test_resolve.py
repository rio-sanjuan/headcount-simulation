"""Tests for resolving a study's inputs: provenance, the required kinds, the extract cache, and
the source probe."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from hcsim.model.core.contract import role_column
from hcsim.model.data.population import REGULAR
from hcsim.studies.kinds import KINDS
from hcsim.studies.resolve import (
    ExtractCache,
    Provenance,
    Resolved,
    ResolveError,
    combined,
    derive,
    probe,
    provisional,
    required_kinds,
    year_pieces,
)

W, F, X = Provenance.SYNTHETIC_WORLD, Provenance.FIXTURE, Provenance.MIXED


def test_provenance_combines_into_one_source_or_a_mix() -> None:
    assert combined([W, W]) is W
    assert combined([F]) is F
    assert combined([W, F]) is X
    assert combined(["synthetic_world", "mixed"]) is X
    with pytest.raises(ValueError, match="no provenance"):
        combined([])
    seed = Resolved("seed_state", object(), W)
    models = derive("fitted_models", object(), [seed], F)
    assert models.provenance is X
    assert (provisional(W), provisional(F), provisional(X)) == (False, True, True)


def test_a_study_needs_its_rungs_and_scopes_kinds() -> None:
    assert required_kinds("unguided", "hcf") == ("seed_state", "fitted_models")
    assert required_kinds("guided_forecast", "hcf+sbc") == (
        "seed_state",
        "fitted_models",
        "guidance_forecast",
        "seed_equity",
    )
    assert required_kinds("guided_actuals", "hcf+sbc") == (
        "seed_state",
        "fitted_models",
        "realized_totals",
        "seed_equity",
        "actual_sbc",
    )
    assert required_kinds("replay", "hcf", plan_path=True) == (
        "seed_state",
        "fitted_models",
        "replay_history",
        "plan",
    )
    assert required_kinds("unguided", "hcf", evaluator_kinds=["ratings"])[-1] == "ratings"
    with pytest.raises(ResolveError, match=r"no input kinds \['weather'\]"):
        required_kinds("unguided", "hcf", evaluator_kinds=["weather"])


def test_a_window_is_cut_at_every_year_end_inside_it() -> None:
    d = dt.date
    assert year_pieces(d(2023, 12, 31), d(2025, 6, 30)) == [
        (d(2023, 12, 31), d(2024, 12, 31)),
        (d(2024, 12, 31), d(2025, 6, 30)),
    ]
    assert year_pieces(d(2025, 1, 31), d(2025, 6, 30)) == [(d(2025, 1, 31), d(2025, 6, 30))]


def test_the_cache_reads_each_years_piece_once(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    world, _ = inputs
    kind = KINDS["requisition_months"]
    d = dt.date
    with duckdb.connect(str(world), read_only=True) as con:
        cost_center = role_column(con, "req_interval")
        cache = ExtractCache(tmp_path / "cache", source="tiny-5")
        first = cache.read(con, kind, REGULAR, cost_center, d(2023, 12, 31), d(2025, 12, 31))
        assert (first.pieces, first.hits) == (2, 0)
        again = cache.read(con, kind, REGULAR, cost_center, d(2023, 12, 31), d(2025, 12, 31))
        assert (again.pieces, again.hits) == (2, 2)
        pd.testing.assert_frame_equal(again.frame, first.frame)
        # A window over one of those years reuses its piece; the rows are what the warehouse
        # gives for the whole window.
        year = cache.read(con, kind, REGULAR, cost_center, d(2024, 12, 31), d(2025, 12, 31))
        assert (year.pieces, year.hits) == (1, 1)
        assert kind.sql is not None
        direct = con.execute(
            kind.sql(REGULAR, cost_center), {"start": d(2023, 12, 31), "end": d(2025, 12, 31)}
        ).df()
        key = list(direct.columns)
        pd.testing.assert_frame_equal(
            first.frame.sort_values(key).reset_index(drop=True),
            direct.sort_values(key).reset_index(drop=True),
        )
        # Another source keys its own pieces.
        other = ExtractCache(tmp_path / "cache", source="tiny-6")
        assert (
            other.read(con, kind, REGULAR, cost_center, d(2024, 12, 31), d(2025, 12, 31)).hits == 0
        )
        # A piece that no longer honors its contract is refused.
        for piece in (tmp_path / "cache").glob("requisition_months-*"):
            pd.DataFrame({"req_id": ["x"]}).to_parquet(piece, index=False)
        with pytest.raises(ResolveError, match="requisition_months needs the columns"):
            cache.read(con, kind, REGULAR, cost_center, d(2024, 12, 31), d(2025, 12, 31))
        with pytest.raises(ResolveError, match="isn't an extract"):
            cache.read(con, KINDS["plan"], REGULAR, cost_center, d(2024, 12, 31), d(2025, 12, 31))
        with pytest.raises(ResolveError, match="start before its end"):
            cache.read(con, kind, REGULAR, cost_center, d(2025, 12, 31), d(2025, 12, 31))


def test_the_probe_names_every_kind_a_source_cant_serve(inputs: tuple[Path, Path]) -> None:
    world, _ = inputs
    with duckdb.connect(str(world), read_only=True) as con:
        cost_center = role_column(con, "req_interval")
        assert probe(con, KINDS, REGULAR, cost_center) == []
    empty = duckdb.connect()
    empty.execute("CREATE TABLE equity_award (award_id BIGINT)")
    problems = probe(empty, ["growth_openings", "seed_equity"], REGULAR, '"cost_center"')
    assert problems[0].startswith("the source can't serve growth_openings: ")
    assert problems[1] == (
        "the source can't serve seed_equity: it has no tables "
        "['equity_holder', 'equity_valuation', 'equity_vest']"
    )
