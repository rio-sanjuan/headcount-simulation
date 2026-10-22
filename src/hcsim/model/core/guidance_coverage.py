"""Guidance coverage: what a forecast's categories mean for a run.

Given the categories an artifact carries, the domains a run has, and the domains it declares
unguided, every category gets one state:

- `composed`: its quota reaches a guided domain. An inner node of the fill tree that isn't
  carried is composed when its parts are, since it can be summed from them.
- `uncomposed`: only a fill-tree ancestor is carried, so the domains below share that
  ancestor's quota. A carried ancestor whose parts are missing is uncomposed too.
- `unmapped`: it's carried, but none of its consumers is a domain in this run.
- `unfitted`: neither it nor anything above it in the fill tree is carried, so its domains run
  unguided.
- `forecast_internal`: it's carried for the forecast's own use: the headcount anchor, or an
  inner node whose parts are carried too.
- `declared_unguided`: it's carried, but the run declares every one of its consumers
  unguided.

A domain's mapping to its category has a fidelity. It's `exact` when the domain realizes the
whole category, and `narrower` when other channels count toward the same series. External
fills, for instance, are only part of the external hires the transactions record, because
campus starts and pending starts are recorded as hires too.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from hcsim.model.core.categories import ANCHOR, CATEGORIES, GuidanceCategory
from hcsim.model.core.fill_tree import FILL_TREE, ancestors, leaves


class CoverageState(StrEnum):
    COMPOSED = "composed"
    UNCOMPOSED = "uncomposed"
    UNMAPPED = "unmapped"
    UNFITTED = "unfitted"
    FORECAST_INTERNAL = "forecast_internal"
    DECLARED_UNGUIDED = "declared_unguided"


class Fidelity(StrEnum):
    EXACT = "exact"
    NARROWER = "narrower"


class CoverageError(ValueError):
    """A coverage question that names something unknown."""


#: Domain outputs that realize only part of their category's series.
NARROWER: Final = frozenset({"external_fill", "backfill", "requisition_generation"})

FIDELITY: Final[Mapping[str, Fidelity]] = MappingProxyType(
    {
        consumer: Fidelity.NARROWER if consumer in NARROWER else Fidelity.EXACT
        for spec in CATEGORIES.values()
        for consumer in spec.consumers
    }
)


@dataclass(frozen=True)
class CategoryCoverage:
    category: GuidanceCategory
    state: CoverageState
    #: The carried category whose quota the category's domains read: itself, a fill-tree
    #: ancestor, or None when they run unguided.
    source: GuidanceCategory | None
    #: The category's consumers in this run, each with its mapping's fidelity.
    consumers: Mapping[str, Fidelity]

    @property
    def guided(self) -> bool:
        return self.state in (CoverageState.COMPOSED, CoverageState.UNCOMPOSED)


def coverage(
    carried: Iterable[GuidanceCategory],
    domains: Iterable[str],
    unguided: Iterable[str] = (),
) -> dict[GuidanceCategory, CategoryCoverage]:
    """Every category's coverage state, given what an artifact carries and what a run has."""
    have = frozenset(carried)
    run = frozenset(domains)
    declared = frozenset(unguided)
    known = frozenset(FIDELITY)
    unknown = sorted((run | declared) - known)
    if unknown:
        raise CoverageError(f"no category is consumed by {unknown}; known: {sorted(known)}")
    stray = sorted(declared - run)
    if stray:
        raise CoverageError(f"declared unguided but not in the run: {stray}")
    return {c: _cover(c, have, run, declared) for c in GuidanceCategory}


def _cover(
    category: GuidanceCategory,
    have: frozenset[GuidanceCategory],
    run: frozenset[str],
    declared: frozenset[str],
) -> CategoryCoverage:
    spec = CATEGORIES[category]
    present = [d for d in spec.consumers if d in run]
    consumers = MappingProxyType({d: FIDELITY[d] for d in present})
    guided = [d for d in present if d not in declared]
    if category == ANCHOR or category in FILL_TREE:
        parts = leaves(category)
        parts_carried = all(p in have for p in parts)
        if category in have:
            state = (
                CoverageState.FORECAST_INTERNAL
                if category == ANCHOR or parts_carried
                else CoverageState.UNCOMPOSED
            )
            return CategoryCoverage(category, state, category, consumers)
        if category != ANCHOR and parts_carried:
            return CategoryCoverage(category, CoverageState.COMPOSED, None, consumers)
        return CategoryCoverage(category, CoverageState.UNFITTED, None, consumers)
    source = (
        category if category in have else next((a for a in ancestors(category) if a in have), None)
    )
    if source is None:
        return CategoryCoverage(category, CoverageState.UNFITTED, None, consumers)
    if not present:
        return CategoryCoverage(category, CoverageState.UNMAPPED, source, consumers)
    if not guided:
        return CategoryCoverage(category, CoverageState.DECLARED_UNGUIDED, None, consumers)
    state = CoverageState.COMPOSED if source == category else CoverageState.UNCOMPOSED
    return CategoryCoverage(category, state, source, consumers)
