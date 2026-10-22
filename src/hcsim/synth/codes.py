"""The transaction code catalog, and each code's disposition.

Every code in `workforce_txn` has one disposition:

- `measured`: it counts in regular headcount, as a stock row or a flow.
- `excluded`: it belongs to a population outside regular headcount, the contingent pool.
- `ignored`: it records something with no headcount meaning, such as an administrative change.

A code missing from the catalog is unknown, and readers refuse it rather than guess. The
catalog is written as the `dim_trans_type` table.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

#: Each code's disposition, its amount, and what it records.
CATALOG: Final[dict[str, tuple[str, int, str]]] = {
    "ACTIVE": ("measured", 1, "One row per person in post at each month-end"),
    "HIRE": ("measured", 1, "A regular employee's first day"),
    "TERM": ("measured", -1, "A regular employee's departure"),
    "RIF": ("measured", -1, "A departure in a termination program"),
    "TERM_PENDING": ("measured", 0, "Notice of a departure, dated on the day of leaving"),
    "XFER_OUT": ("measured", -1, "Leaving a role for a transfer"),
    "XFER_IN": ("measured", 1, "Arriving in a role by transfer"),
    "REORG_OUT": ("measured", -1, "Leaving a cost center in a reorganization"),
    "REORG_IN": ("measured", 1, "Arriving in a cost center in a reorganization"),
    "PROMO": ("measured", 0, "A promotion, in place"),
    "RECLASS": ("measured", 0, "A change of job family, in place"),
    "CONV_OUT": ("measured", -1, "Leaving regular headcount for the contingent pool"),
    "CONV_IN": ("measured", 1, "Joining regular headcount from the contingent pool"),
    "CTG_START": ("excluded", 1, "A contingent worker's start"),
    "CTG_END": ("excluded", -1, "A contingent worker's end"),
    "ADMIN": ("ignored", 0, "An administrative change with no effect on headcount"),
    "HIRE_REV": ("measured", -1, "Reverses a hire recorded in error"),
    "TERM_REV": ("measured", 1, "Reverses a departure recorded in error"),
    "HIRE_PENDING": ("measured", 0, "An accepted offer, dated on the future start"),
}
DISPOSITIONS: Final = ("measured", "excluded", "ignored")


def catalog_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trans_type": code, "disposition": d, "amount": a, "description": text}
            for code, (d, a, text) in CATALOG.items()
        ]
    )
