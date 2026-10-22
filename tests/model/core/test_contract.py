"""Tests for resolving a role's column through the warehouse's contract."""

from __future__ import annotations

import duckdb

from hcsim.model.core.contract import role_column


def test_a_warehouse_without_a_contract_uses_the_role_s_name() -> None:
    with duckdb.connect() as con:
        assert role_column(con, "req_interval") == '"cost_center"'


def test_the_contract_names_the_column_and_quotes_it() -> None:
    with duckdb.connect() as con:
        con.execute(
            "CREATE TABLE contract_columns AS SELECT * FROM (VALUES "
            "('req_interval', 'cost_center', 'dept\"id')) t(table_name, role, column_name)"
        )
        assert role_column(con, "req_interval") == '"dept""id"'
        assert role_column(con, "pending_start") == '"cost_center"'  # no row: the default
