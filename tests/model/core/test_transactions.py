"""Tests for the transaction vocabulary and its catalog."""

from __future__ import annotations

import duckdb
import pytest

from hcsim.model.core.transactions import (
    Disposition,
    TransType,
    TransTypeCatalog,
    TransTypeError,
    TransTypeInfo,
)
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> TransTypeCatalog:
    path = tmp_path_factory.mktemp("world") / "w.duckdb"
    write_world(generate_world(preset("tiny", seed=1)), path)
    with duckdb.connect(str(path), read_only=True) as con:
        return TransTypeCatalog.from_warehouse(con)


def test_the_world_defines_every_contract_code(catalog: TransTypeCatalog) -> None:
    assert catalog.known_types() == frozenset(TransType)
    assert len(catalog) == len(TransType)
    assert {e.code for e in catalog} == {t.value for t in TransType}


def test_dispositions_and_amounts_come_from_the_data(catalog: TransTypeCatalog) -> None:
    assert catalog.disposition(TransType.HIRE) is Disposition.MEASURED
    assert catalog.amount(TransType.TERM) == -1
    assert catalog.codes(Disposition.EXCLUDED) == {"CTG_START", "CTG_END"}
    assert catalog.codes(Disposition.IGNORED) == {"ADMIN", "SYS_ADJ"}
    assert catalog.info("PROMO").description
    assert "HIRE" in catalog
    assert "BOGUS" not in catalog


def test_a_changed_disposition_needs_no_code_change() -> None:
    catalog = TransTypeCatalog([TransTypeInfo("ADMIN", Disposition.MEASURED, 0)])
    assert catalog.disposition(TransType.ADMIN) is Disposition.MEASURED


def test_unknown_codes_are_refused_by_name(catalog: TransTypeCatalog) -> None:
    catalog.check(["HIRE", TransType.TERM])
    with pytest.raises(TransTypeError, match=r"missing from the catalog: \['NEW_A', 'NEW_B'\]"):
        catalog.check(["HIRE", "NEW_B", "NEW_A", "NEW_B"])
    with pytest.raises(TransTypeError, match="'NEW_A' isn't in the catalog"):
        catalog.info("NEW_A")


def test_a_bad_catalog_reports_every_problem() -> None:
    with pytest.raises(TransTypeError, match="defined twice") as error:
        TransTypeCatalog(
            [
                TransTypeInfo("HIRE", Disposition.MEASURED, 1),
                TransTypeInfo("HIRE", Disposition.MEASURED, 1),
                TransTypeInfo("ODD", Disposition.MEASURED, 2),
            ]
        )
    assert "'ODD' has amount 2" in str(error.value)
    rows = [
        {"trans_type": "A", "disposition": "counted", "amount": 1},
        {"trans_type": "B", "disposition": "maybe", "amount": 0},
    ]
    with pytest.raises(TransTypeError, match="'A' has disposition 'counted'") as error:
        TransTypeCatalog.from_rows(rows)
    assert "'B' has disposition 'maybe'" in str(error.value)
    assert "known: measured, excluded, ignored" in str(error.value)


def test_rows_without_a_description_read_as_empty() -> None:
    catalog = TransTypeCatalog.from_rows(
        [{"trans_type": "X", "disposition": "ignored", "amount": 0}]
    )
    assert catalog.info("X").description == ""
