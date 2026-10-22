"""Tests for the column catalog."""

from __future__ import annotations

import pytest

from hcsim.model.sql.catalog import COLUMNS, CatalogError, Kind, Use, check, entry, require


def test_every_org_level_shares_one_entry() -> None:
    for level in ("org_level1", "org_level7", "org_level12"):
        column = entry(level)
        assert (column.expression, column.kind) == (f"o.{level}", Kind.LABEL)
        assert column.uses == frozenset(Use)
    for bad in ("org_level0", "org_level", "org_levelx"):
        with pytest.raises(CatalogError, match="isn't in the column catalog"):
            entry(bad)


def test_uses_are_permitted_per_column() -> None:
    assert require("job_family", Use.GROUP).expression == "j.job_family"
    assert require("account_group", Use.POPULATION).expression == "a.account_group"
    refused = "'account_group' can't group by; it permits filter, population"
    with pytest.raises(CatalogError, match=refused):
        require("account_group", Use.GROUP)
    with pytest.raises(CatalogError, match="'trans_type' can't define a population"):
        require("trans_type", Use.POPULATION)
    with pytest.raises(CatalogError, match="'event_date' can't group by"):
        require("event_date", Use.GROUP)


def test_unknown_columns_list_the_known_ones() -> None:
    with pytest.raises(CatalogError, match=r"known: account, account_group, .*org_level<k>"):
        entry("salary")


def test_check_reports_every_refusal_at_once() -> None:
    check([("cost_center", Use.GROUP), ("event_date", Use.FILTER)])
    with pytest.raises(CatalogError, match="'salary' isn't") as error:
        check([("salary", Use.GROUP), ("is_manager", Use.GROUP)])
    assert "'is_manager' can't group by" in str(error.value)


def test_every_entry_lives_in_the_standard_join() -> None:
    aliases = {column.expression.split(".")[0] for column in COLUMNS.values()}
    assert aliases <= {"t", "o", "j", "l", "c", "a"}
    assert all(column.uses and column.description for column in COLUMNS.values())
