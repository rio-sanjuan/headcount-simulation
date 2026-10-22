"""Quota derivations: how a subset run reads quotas from an artifact built for more.

| derivation | quotas the subset gets |
|---|---|
| `UNCHANGED` | the artifact's, as they are: a whole-population run |
| `RESTRICT` | the quotas of the groups wholly inside the subset |
| `RESHARE` | inside groups in full, and each crossing group's quota times its share inside |
| `DISAGGREGATE` | each coarse group's quota split over its finer groups by a share basis |
| `REFIT` | none from the artifact: the forecast is refitted on the subset's own history |
| `UNGUIDED` | none: the domains run unguided |

A share comes from a basis, counted per group: the headcount stock, the open openings, a recent
flow of the category, or equal shares. A group with nothing of its basis splits equally.

Where quotas are known on two margins, such as org and family, raking (iterative proportional
fitting) adjusts a seed table until it matches both. The margins must have one total.

All functions work on plain arrays. A quota array is (groups,) or (steps, groups).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hcsim.model.data.grain import Placement

Floats = NDArray[np.float64]


class Derivation(StrEnum):
    UNCHANGED = "unchanged"
    RESTRICT = "restrict"
    RESHARE = "reshare"
    DISAGGREGATE = "disaggregate"
    REFIT = "refit"
    UNGUIDED = "unguided"


class ShareBasis(StrEnum):
    HEADCOUNT = "headcount"
    OPENINGS = "openings"
    FLOW = "flow"
    EQUAL = "equal"


class QuotaError(ValueError):
    """Quotas, shares, or margins that can't be combined."""


def _counts(values: ArrayLike, what: str) -> Floats:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or (array < 0).any():
        raise QuotaError(f"{what} must be finite and non-negative")
    return array


def inside_shares(inside: ArrayLike, total: ArrayLike) -> Floats:
    """Each group's share inside the subset: its basis inside over its basis in all. A group
    with none of the basis has share 0: by that basis, none of it is inside. Choose another
    basis if that drops a quota the subset should keep."""
    i, t = _counts(inside, "inside counts"), _counts(total, "group counts")
    if i.shape != t.shape:
        raise QuotaError("inside and total counts need one entry per group")
    if (i > t).any():
        raise QuotaError("a group can't have more inside the subset than in all")
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(t > 0, i / t, 0.0)
    return share


def derive(
    quota: ArrayLike,
    groups: tuple[str, ...],
    placement: Placement,
    derivation: Derivation,
    shares: ArrayLike | None = None,
) -> Floats | None:
    """The subset's quota per group, in `groups` order, or None when it runs unguided or
    refits. `shares` gives each crossing group's share inside, in `groups` order."""
    q = _counts(quota, "quotas")
    if q.shape[-1] != len(groups):
        raise QuotaError(f"quotas have {q.shape[-1]} groups, the artifact {len(groups)}")
    if derivation in (Derivation.UNGUIDED, Derivation.REFIT):
        return None
    if derivation is Derivation.UNCHANGED:
        return q.copy()
    keep = np.array([g in placement.inside for g in groups], dtype=np.float64)
    if derivation is Derivation.RESTRICT:
        return q * keep
    if derivation is Derivation.RESHARE:
        if shares is None:
            raise QuotaError("resharing needs each group's share inside the subset")
        s = _counts(shares, "shares")
        if s.shape != (len(groups),) or (s > 1).any():
            raise QuotaError("shares need one value in [0, 1] per group")
        crossing = np.array([g in placement.crossing for g in groups], dtype=np.float64)
        return q * (keep + crossing * s)
    raise QuotaError(f"{derivation.value} isn't a per-group derivation; use disaggregate()")


def disaggregate(quota: ArrayLike, parent: ArrayLike, basis: ArrayLike) -> Floats:
    """Split each coarse group's quota over its finer groups. `parent` gives each fine group's
    coarse group, and `basis` each fine group's share basis. A coarse group with no basis
    splits equally among its fine groups; one with no fine groups is refused, since its quota
    would vanish."""
    q = _counts(quota, "quotas")
    owner = np.asarray(parent, dtype=np.int64)
    weight = _counts(basis, "share basis")
    n = q.shape[-1]
    if owner.shape != weight.shape or owner.ndim != 1:
        raise QuotaError("parent and basis need one entry per fine group")
    if owner.size and (owner.min() < 0 or owner.max() >= n):
        raise QuotaError(f"parents must lie in 0..{n - 1}")
    orphaned = sorted(set(range(n)) - set(owner.tolist()))
    if orphaned:
        raise QuotaError(f"coarse groups {orphaned} have no finer groups to take their quota")
    totals = np.bincount(owner, weights=weight, minlength=n)
    sizes = np.bincount(owner, minlength=n).astype(np.float64)
    share = np.where(
        totals[owner] > 0,
        weight / np.where(totals[owner] > 0, totals[owner], 1),
        1.0 / sizes[owner],
    )
    result: Floats = q[..., owner] * share
    return result


@dataclass(frozen=True)
class RakingMarginals:
    rows: Floats
    columns: Floats
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        rows = _counts(self.rows, "row margins")
        columns = _counts(self.columns, "column margins")
        if rows.ndim != 1 or columns.ndim != 1:
            raise QuotaError("margins must be one-dimensional")
        if abs(rows.sum() - columns.sum()) > self.tolerance * max(1.0, rows.sum()):
            raise QuotaError(
                f"the margins disagree on the total: {rows.sum():g} by row, "
                f"{columns.sum():g} by column"
            )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)


def rake(seed: ArrayLike, margins: RakingMarginals, *, max_iterations: int = 1000) -> Floats:
    """Iterative proportional fitting: scale `seed` (rows by columns) until its row and column
    sums match the margins. A margin that needs mass where the seed has none can't be met."""
    table = _counts(seed, "the seed").copy()
    rows, columns = margins.rows, margins.columns
    if table.shape != (len(rows), len(columns)):
        raise QuotaError(f"the seed is {table.shape}, the margins {(len(rows), len(columns))}")
    empty_rows = np.flatnonzero((table.sum(axis=1) == 0) & (rows > 0))
    empty_columns = np.flatnonzero((table.sum(axis=0) == 0) & (columns > 0))
    if empty_rows.size or empty_columns.size:
        raise QuotaError(
            f"the seed is empty where the margins need mass: rows {empty_rows.tolist()}, "
            f"columns {empty_columns.tolist()}"
        )
    for _ in range(max_iterations):
        with np.errstate(invalid="ignore", divide="ignore"):
            table *= np.where(table.sum(axis=1) > 0, rows / table.sum(axis=1), 0.0)[:, None]
            table *= np.where(table.sum(axis=0) > 0, columns / table.sum(axis=0), 0.0)[None, :]
        gap = np.abs(table.sum(axis=1) - rows).max(initial=0.0)
        if gap <= margins.tolerance * max(1.0, rows.sum()):
            return table
    raise QuotaError(f"raking didn't converge in {max_iterations} iterations")
