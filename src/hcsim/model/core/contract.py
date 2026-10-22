"""The warehouse's column contract: which column plays a role in each table.

Systems built apart name the same thing differently. A warehouse may carry a
`contract_columns` table saying, for each table, which column holds a role such as the leaf
cost center. Readers resolve names through it rather than assuming one spelling.
"""

from __future__ import annotations

from typing import Final

import duckdb

CONTRACT_TABLE: Final = "contract_columns"
COST_CENTER: Final = "cost_center"


def role_column(con: duckdb.DuckDBPyConnection, table: str, role: str = COST_CENTER) -> str:
    """The column playing `role` in `table`, quoted for SQL. A warehouse with no contract
    table, or no row for the table, uses the role's own name."""
    (present,) = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [CONTRACT_TABLE]
    ).fetchone() or (0,)
    row = (
        con.execute(
            f"SELECT column_name FROM {CONTRACT_TABLE} WHERE table_name = ? AND role = ?",
            [table, role],
        ).fetchone()
        if present
        else None
    )
    name = str(row[0]) if row else role
    return '"' + name.replace('"', '""') + '"'
