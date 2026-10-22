"""Tests for transaction and template measures."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from hcsim.model.core.categories import Grain
from hcsim.model.core.contract import role_column
from hcsim.model.core.measures import (
    HEADCOUNT_STOCK,
    NET_DEPARTURES,
    NET_EXTERNAL_HIRES,
    PROMOTIONS_IN_PLACE,
    REQUISITIONS_CLOSED_UNFILLED,
    REQUISITIONS_OPENED,
    Measure,
    MeasureError,
    TransTypeMeasure,
)
from hcsim.model.core.transactions import (
    Disposition,
    TransType,
    TransTypeCatalog,
    TransTypeError,
    TransTypeInfo,
)
from hcsim.synth.config import QuirkTruth, preset
from hcsim.synth.world import generate_world, write_world

MEASURES: tuple[Measure, ...] = (
    HEADCOUNT_STOCK,
    NET_EXTERNAL_HIRES,
    NET_DEPARTURES,
    PROMOTIONS_IN_PLACE,
    REQUISITIONS_OPENED,
    REQUISITIONS_CLOSED_UNFILLED,
)


def _warehouse(path: Path, **quirks: object) -> duckdb.DuckDBPyConnection:
    config = dataclasses.replace(preset("tiny", seed=7), quirks=QuirkTruth(**quirks))  # type: ignore[arg-type]
    write_world(generate_world(config), path)
    return duckdb.connect(str(path), read_only=True)


@pytest.fixture(scope="module")
def warehouses(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[duckdb.DuckDBPyConnection, duckdb.DuckDBPyConnection]]:
    root = tmp_path_factory.mktemp("worlds")
    clean = _warehouse(root / "clean.duckdb")
    quirky = _warehouse(
        root / "quirky.duckdb",
        correction_share=0.3,
        no_show_pair_share=1.0,
        pending_hire_rows=True,
        cost_center_aliases=True,
    )
    yield clean, quirky
    clean.close()
    quirky.close()


def counted(con: duckdb.DuckDBPyConnection, measure: Measure) -> pd.Series:
    sql = measure.sql(role_column(con, measure.table))
    frame = con.execute(
        f"SELECT month, cost_center, sum(value) AS v FROM ({sql}) GROUP BY ALL"
    ).df()
    return frame.set_index(["month", "cost_center"])["v"].sort_index().astype(int)


def raw(con: duckdb.DuckDBPyConnection, *codes: str) -> pd.Series:
    frame = con.execute(
        "SELECT date_trunc('month', event_date) AS month, cost_center, trans_type, count(*) AS n "
        "FROM workforce_txn WHERE account = 'ACCT-100' GROUP BY ALL"
    ).df()
    frame = frame[frame["trans_type"].isin(codes)]
    return frame.groupby(["month", "cost_center"])["n"].sum().sort_index().astype(int)


def test_every_measure_passes_the_world_s_catalog(
    warehouses: tuple[duckdb.DuckDBPyConnection, ...],
) -> None:
    catalog = TransTypeCatalog.from_warehouse(warehouses[0])
    for measure in MEASURES:
        measure.check(catalog)
    assert {m.grain for m in MEASURES} == {Grain.STOCK, Grain.HEADCOUNT, Grain.REQUISITION}


def test_stocks_and_promotions_count_their_rows(
    warehouses: tuple[duckdb.DuckDBPyConnection, ...],
) -> None:
    clean = warehouses[0]
    pd.testing.assert_series_equal(
        counted(clean, HEADCOUNT_STOCK), raw(clean, "ACTIVE"), check_names=False
    )
    pd.testing.assert_series_equal(
        counted(clean, PROMOTIONS_IN_PLACE), raw(clean, "PROMO"), check_names=False
    )


def test_reversals_net_out_of_hires_and_departures(
    warehouses: tuple[duckdb.DuckDBPyConnection, ...],
) -> None:
    """Corrections and no-show pairs reverse within their month, so the net counts on the
    quirky world are the clean world's counts."""
    clean, quirky = warehouses
    assert int(raw(quirky, "HIRE_REV").sum()) > 0
    for measure, code in ((NET_EXTERNAL_HIRES, "HIRE"), (NET_DEPARTURES, "TERM")):
        got = counted(quirky, measure)
        expected = raw(clean, code)
        pd.testing.assert_series_equal(got[got != 0], expected, check_names=False)


def test_requisitions_count_at_first_and_last_sighting(
    warehouses: tuple[duckdb.DuckDBPyConnection, ...],
) -> None:
    clean, aliased = warehouses  # the second world names its cost centers differently
    for con in (clean, aliased):
        reqs = con.execute(
            f"SELECT *, {role_column(con, 'req_interval')} AS cc FROM req_interval"
        ).df()
        per = reqs.sort_values(["req_id", "interval_number"]).groupby("req_id")
        first = per.first()
        opened = first.assign(month=pd.to_datetime(first["opened_date"]).dt.to_period("M"))
        expected = (
            per["openings"].max().clip(upper=10).groupby([opened["month"], opened["cc"]]).sum()
        )
        got = counted(con, REQUISITIONS_OPENED)
        assert got.sum() == expected.sum()
        last = per.last()
        unfilled = last[last["close_reason"] == "unfilled"]
        assert counted(con, REQUISITIONS_CLOSED_UNFILLED).sum() == len(unfilled) > 0


def test_a_measure_refuses_codes_it_can_t_count() -> None:
    catalog = TransTypeCatalog(
        [
            TransTypeInfo("HIRE", Disposition.MEASURED, 1),
            TransTypeInfo("CTG_START", Disposition.EXCLUDED, 1),
        ]
    )
    pool = TransTypeMeasure("pool", {TransType.CTG_START: 1})
    with pytest.raises(MeasureError, match=r"pool: codes that aren't measured: \['CTG_START'\]"):
        pool.check(catalog)
    with pytest.raises(TransTypeError, match="HIRE_REV"):
        NET_EXTERNAL_HIRES.check(catalog)


def test_a_bad_declaration_reports_every_problem() -> None:
    with pytest.raises(MeasureError, match=r"signs must be \+1 or -1: \['HIRE'\]") as error:
        TransTypeMeasure("odd", {TransType.HIRE: 2}, grain=Grain.REQUISITION)
    assert "count heads, not requisitions" in str(error.value)
    with pytest.raises(MeasureError, match="no codes"):
        TransTypeMeasure("empty", {})


def test_the_expression_is_a_signed_case() -> None:
    assert NET_EXTERNAL_HIRES.expression() == (
        "(CASE trans_type WHEN 'HIRE' THEN 1 WHEN 'HIRE_REV' THEN -1 ELSE 0 END) * headcount"
    )
    assert NET_EXTERNAL_HIRES.codes == {TransType.HIRE, TransType.HIRE_REV}
    REQUISITIONS_OPENED.check(TransTypeCatalog([]))  # reads no codes
