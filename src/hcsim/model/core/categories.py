"""Guidance categories: the series a guidance forecast can carry.

A category is a count per group per month. Its grain decides which identity it belongs to:
headcount-grain flows enter the forecast-coherency identity, requisition-grain series belong to
the opening identity, and the `headcount` stock anchors coherency. The grain is declared, not
inferred from the sign: `requisitions_opened` is an inflow exactly as `external_hires` is, yet it
belongs to a different identity.

There are 11 categories:

| category | grain | symbol | in the coherency identity |
|---|---|---|---|
| `headcount` | stock | `Ĥ` | the anchor |
| `attrition` | headcount | `A` | an outflow |
| `external_hires` | headcount | `F_ext` | an inflow |
| `conversion_hires` | headcount | `F_conv` | an inflow |
| `internal_starts` | headcount | `F_tr` | an inflow |
| `conversions_out` | headcount | `V_conv` | an outflow |
| `hires` | headcount | `F_ext + F_conv` | through its parts |
| `fills` | headcount | `F_ext + F_conv + F_tr` | through its parts |
| `promotions` | headcount | `Π` | no: a promotion moves no one |
| `requisitions_opened` | requisition | `G + B_att + B_tr` | no |
| `requisitions_closed_unfilled` | requisition | `U` | no |

`hires` and `fills` are the inner nodes of the fill tree. They enter the identity through
their parts, so it never counts a hire twice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class Grain(StrEnum):
    HEADCOUNT = "headcount"  # a flow of heads
    REQUISITION = "requisition"  # a flow of requisitions or their openings
    STOCK = "stock"  # month-end headcount, the anchor


class GuidanceCategory(StrEnum):
    HEADCOUNT = "headcount"
    ATTRITION = "attrition"
    EXTERNAL_HIRES = "external_hires"
    CONVERSION_HIRES = "conversion_hires"
    INTERNAL_STARTS = "internal_starts"
    CONVERSIONS_OUT = "conversions_out"
    HIRES = "hires"
    FILLS = "fills"
    PROMOTIONS = "promotions"
    REQUISITIONS_OPENED = "requisitions_opened"
    REQUISITIONS_CLOSED_UNFILLED = "requisitions_closed_unfilled"


@dataclass(frozen=True)
class CategorySpec:
    category: GuidanceCategory
    grain: Grain
    symbol: str
    #: Sign in the forecast-coherency identity: +1 inflow, -1 outflow, 0 for the anchor, and
    #: None when the category isn't a term of that identity, or enters it through its parts.
    coherency_sign: int | None
    #: Domain outputs that consume the category's quota. Empty means forecast-internal, or
    #: served through the fill tree.
    consumers: tuple[str, ...]


_H, _R, _S = Grain.HEADCOUNT, Grain.REQUISITION, Grain.STOCK
_C = GuidanceCategory

CATEGORIES: Final[Mapping[GuidanceCategory, CategorySpec]] = MappingProxyType(
    {
        spec.category: spec
        for spec in (
            CategorySpec(_C.HEADCOUNT, _S, "Ĥ", 0, ()),
            CategorySpec(_C.ATTRITION, _H, "A", -1, ("attrition",)),
            CategorySpec(_C.EXTERNAL_HIRES, _H, "F_ext", +1, ("external_fill",)),
            CategorySpec(_C.CONVERSION_HIRES, _H, "F_conv", +1, ("conversion_fill",)),
            CategorySpec(_C.INTERNAL_STARTS, _H, "F_tr", +1, ("transfer_fill",)),
            CategorySpec(_C.CONVERSIONS_OUT, _H, "V_conv", -1, ("conversion_out",)),
            CategorySpec(_C.HIRES, _H, "F_ext + F_conv", None, ()),
            CategorySpec(_C.FILLS, _H, "F_ext + F_conv + F_tr", None, ()),
            CategorySpec(_C.PROMOTIONS, _H, "Π", None, ("promotion",)),
            CategorySpec(
                _C.REQUISITIONS_OPENED,
                _R,
                "G + B_att + B_tr",
                None,
                ("backfill", "requisition_generation"),
            ),
            CategorySpec(_C.REQUISITIONS_CLOSED_UNFILLED, _R, "U", None, ("cancellation",)),
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
