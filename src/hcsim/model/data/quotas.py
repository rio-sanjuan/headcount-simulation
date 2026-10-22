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
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hcsim.model.data.grain import GrainRelation, Placement

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


class Refusal(StrEnum):
    """Combinations of a derivation and a run that are refused by name."""

    UNCHANGED_FOR_SUBSET = "unchanged_for_subset"
    SUBSET_DERIVATION_WITHOUT_SUBSET = "subset_derivation_without_subset"
    RESTRICT_CROSSING = "restrict_crossing"
    NO_GROUP_REACHES = "no_group_reaches"
    DISAGGREGATE_NOT_COARSER = "disaggregate_not_coarser"


class QuotaRefused(QuotaError):
    """A derivation refused for this run, with the refusal's name."""

    def __init__(self, refusal: Refusal, message: str) -> None:
        super().__init__(f"{refusal.value}: {message}")
        self.refusal = refusal


_SUBSET_DERIVATIONS: Final = frozenset({Derivation.RESTRICT, Derivation.RESHARE})


def check_derivation(
    derivation: Derivation, placement: Placement | None, *, coarser: bool = False
) -> None:
    """Refuse a derivation that can't serve this run. `placement` is None for a run of the
    whole population. `coarser` says the artifact groups more coarsely than the run needs."""
    if placement is None:
        if derivation in _SUBSET_DERIVATIONS:
            raise QuotaRefused(
                Refusal.SUBSET_DERIVATION_WITHOUT_SUBSET,
                f"{derivation.value} reads a subset's part of the quotas, but this run has no "
                "subset; use unchanged",
            )
    elif derivation is Derivation.UNCHANGED:
        raise QuotaRefused(
            Refusal.UNCHANGED_FOR_SUBSET,
            "unchanged quotas are the whole population's, so a subset would realize events "
            "meant for people outside it; use restrict, reshare, disaggregate, refit, or "
            "unguided",
        )
    elif placement.relation is GrainRelation.UNREACHABLE and derivation not in (
        Derivation.REFIT,
        Derivation.UNGUIDED,
    ):
        raise QuotaRefused(
            Refusal.NO_GROUP_REACHES,
            f"no group of the artifact reaches the subset, so {derivation.value} has nothing "
            "to read; use refit or unguided",
        )
    elif derivation is Derivation.RESTRICT and placement.relation is GrainRelation.CROSSING:
        raise QuotaRefused(
            Refusal.RESTRICT_CROSSING,
            f"groups {list(placement.crossing)} cross the subset's boundary, so restricting "
            "would keep their whole quotas or drop them; use reshare",
        )
    if derivation is Derivation.DISAGGREGATE and not coarser:
        raise QuotaRefused(
            Refusal.DISAGGREGATE_NOT_COARSER,
            "the artifact already groups at the run's grain, so there's nothing to split; "
            "use unchanged, restrict, or reshare",
        )


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
    placement: Placement | None,
    derivation: Derivation,
    shares: ArrayLike | None = None,
) -> Floats | None:
    """The run's quota per group, in `groups` order, or None when it runs unguided or refits.
    `placement` is None for a whole-population run. `shares` gives each crossing group's
    share inside, in `groups` order. A refused combination raises `QuotaRefused`."""
    if derivation is Derivation.DISAGGREGATE:
        raise QuotaError("disaggregate isn't a per-group derivation; use disaggregate()")
    check_derivation(derivation, placement)
    q = _counts(quota, "quotas")
    if q.shape[-1] != len(groups):
        raise QuotaError(f"quotas have {q.shape[-1]} groups, the artifact {len(groups)}")
    if derivation in (Derivation.UNGUIDED, Derivation.REFIT):
        return None
    if derivation is Derivation.UNCHANGED:
        return q.copy()
    if placement is None:  # pragma: no cover - check_derivation refuses these first
        raise QuotaError(f"{derivation.value} needs a subset's placement")
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
    raise QuotaError(f"no derivation {derivation!r}")  # pragma: no cover - the enum is closed


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
