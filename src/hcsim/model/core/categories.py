"""Guidance categories: the series a guidance forecast can carry.

A category is a count per group per month. Its grain decides which identity it belongs to:
headcount-grain flows enter the forecast-coherency identity, requisition-grain series belong to
the opening identity, and the `headcount` stock anchors coherency. The grain is declared, not
inferred from the sign: `requisitions_opened` is an inflow exactly as `hires` is, yet it belongs
to a different identity.

This is the walking-skeleton subset. The full set, including the fill tree, arrives in M3.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class Grain(StrEnum):
    HEADCOUNT = "headcount"  # a flow of heads
    REQUISITION = "requisition"  # a flow of requisition openings
    STOCK = "stock"  # month-end headcount, the anchor


class GuidanceCategory(StrEnum):
    ATTRITION = "attrition"
    HIRES = "hires"
    REQUISITIONS_OPENED = "requisitions_opened"
    HEADCOUNT = "headcount"


@dataclass(frozen=True)
class CategorySpec:
    category: GuidanceCategory
    grain: Grain
    symbol: str
    #: Sign in the forecast-coherency identity: +1 inflow, -1 outflow, 0 for the anchor, and
    #: None when the category isn't a term of that identity.
    coherency_sign: int | None
    #: Domain outputs that consume the category's quota. Empty means forecast-internal.
    consumers: tuple[str, ...]


CATEGORIES: Final[Mapping[GuidanceCategory, CategorySpec]] = MappingProxyType(
    {
        spec.category: spec
        for spec in (
            CategorySpec(GuidanceCategory.ATTRITION, Grain.HEADCOUNT, "A", -1, ("attrition",)),
            CategorySpec(GuidanceCategory.HIRES, Grain.HEADCOUNT, "F_ext", +1, ("external_fill",)),
            CategorySpec(
                GuidanceCategory.REQUISITIONS_OPENED,
                Grain.REQUISITION,
                "G + B_att + B_tr",
                None,
                ("backfill", "requisition_generation"),
            ),
            CategorySpec(GuidanceCategory.HEADCOUNT, Grain.STOCK, "Ĥ", 0, ()),
        )
    }
)

ANCHOR: Final = GuidanceCategory.HEADCOUNT


class UnknownCategoryError(ValueError):
    """A name that isn't a guidance category."""


def parse_category(name: str) -> GuidanceCategory:
    """Resolve a category by name, refusing anything else with the known names."""
    try:
        return GuidanceCategory(name)
    except ValueError:
        known = ", ".join(c.value for c in GuidanceCategory)
        raise UnknownCategoryError(f"{name!r} isn't a guidance category; known: {known}") from None


def coherency_terms() -> dict[GuidanceCategory, int]:
    """The flow terms of the forecast-coherency identity and their signs, anchor excluded."""
    return {
        c: spec.coherency_sign
        for c, spec in CATEGORIES.items()
        if spec.coherency_sign not in (None, 0)
    }


def category_for_consumer(domain_output: str) -> GuidanceCategory | None:
    """The category a domain output consumes, or None when it's unguided by declaration."""
    for category, spec in CATEGORIES.items():
        if domain_output in spec.consumers:
            return category
    return None
