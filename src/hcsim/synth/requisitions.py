"""The world's requisition process: openings, fills, and unfilled exits.

Every requisition carries one opening in the lite world. Backfill opens on a departure with the
true backfill probability, in the vacated role. Growth opens per cost center at a Poisson rate
drawn per cost center. Each month, an open requisition fills externally (a hazard by age), fills
by transfer when an internal candidate exists, exits unfilled, or stays open. A transfer fill
vacates the mover's old role, which can open a backfill of its own, available from the next
month. Every value is invented.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import pandas as pd

REQ_COLUMNS = [
    "req_id",
    "req_type",
    "cost_center",
    "job_code",
    "job_family",
    "job_level",
    "location",
    "company",
    "is_manager",
    "openings",
    "opened_date",
    "closed_date",
    "close_reason",
    "filled_by",
]


@dataclass
class Requisition:
    req_id: int
    req_type: str
    role: dict[str, Any]
    opened_date: dt.date
    opened_month: int
    closed_date: dt.date | None = None
    close_reason: str | None = None
    filled_by: int | None = None


class RequisitionBook:
    """Every requisition the world opens, open or closed."""

    def __init__(self, first_id: int = 500_001) -> None:
        self.next_id = first_id
        self.open: dict[int, Requisition] = {}
        self.closed: list[Requisition] = []

    def open_requisition(
        self, req_type: str, role: dict[str, Any], date: dt.date, month: int
    ) -> Requisition:
        req = Requisition(self.next_id, req_type, dict(role), date, month)
        self.next_id += 1
        self.open[req.req_id] = req
        return req

    def close(
        self, req: Requisition, date: dt.date, reason: str, filled_by: int | None = None
    ) -> None:
        req.closed_date, req.close_reason, req.filled_by = date, reason, filled_by
        self.closed.append(self.open.pop(req.req_id))

    def frame(self) -> pd.DataFrame:
        reqs = sorted([*self.closed, *self.open.values()], key=lambda r: r.req_id)
        rows = [
            {
                "req_id": r.req_id,
                "req_type": r.req_type,
                **{
                    k: r.role[k]
                    for k in (
                        "cost_center",
                        "job_code",
                        "job_family",
                        "job_level",
                        "location",
                        "company",
                        "is_manager",
                    )
                },
                "openings": 1,
                "opened_date": r.opened_date,
                "closed_date": r.closed_date,
                "close_reason": r.close_reason,
                "filled_by": r.filled_by,
            }
            for r in reqs
        ]
        frame = pd.DataFrame(rows, columns=REQ_COLUMNS)
        return frame.astype({"filled_by": "Int64"})
