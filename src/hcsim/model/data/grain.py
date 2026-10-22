"""Grain relations: how a subset relates to an artifact's grouping.

A guidance artifact holds quotas per group. Before a subset run can read them, each group is
placed against the subset: wholly inside, wholly outside, or crossing its boundary. The groups
together give the relation:

- `COVERED`: no group crosses the boundary, so the subset is a union of whole groups and their
  quotas can be read as they are;
- `CROSSING`: at least one group is partly inside, so its quota has to be shared out;
- `UNREACHABLE`: no group reaches into the subset at all, so the artifact can't guide it.

An org grouping (`cost_center` or an `org_level<k>`) compares each group's leaf cost centers
with the subset's. A grouping by another buffer attribute compares codes. Any subset rule on a
column other than the grouping's cuts through every group it doesn't exclude, since the rows
inside a group differ in that column. `TOPLINE` is one group: covered only by a subset with no
rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from hcsim.model.core.aligner import TOPLINE
from hcsim.model.core.schema import COST_CENTER
from hcsim.model.data.subset import SUBSET_COLUMNS, LoweredSubset
from hcsim.model.sql.catalog import is_org_level


class GrainRelation(StrEnum):
    COVERED = "covered"
    CROSSING = "crossing"
    UNREACHABLE = "unreachable"


class GrainError(ValueError):
    """A grouping the subset can't be placed against."""


@dataclass(frozen=True)
class Placement:
    relation: GrainRelation
    inside: tuple[str, ...]
    crossing: tuple[str, ...]
    outside: tuple[str, ...]


def relate(subset: LoweredSubset, grouping: str, groups: Sequence[str]) -> Placement:
    """Place every group of an artifact grouped by `grouping` against the subset."""
    codes = subset.codes
    if subset.empty:
        return Placement(GrainRelation.UNREACHABLE, (), (), tuple(groups))
    if grouping == TOPLINE:
        place = "inside" if not codes else "crossing"
        return _placement(dict.fromkeys(groups, place))
    if grouping == COST_CENTER or is_org_level(grouping):
        others = set(codes) - {COST_CENTER}
        leaves = codes.get(COST_CENTER)
        schema = subset.schema
        if grouping == COST_CENTER:
            member = {g: {schema.vocabulary(COST_CENTER).code(g)} for g in groups}
        else:
            if grouping not in schema.rollups:
                raise GrainError(f"the schema has no rollup {grouping!r}")
            vocab = schema.vocabulary(grouping)
            rollup = schema.rollups[grouping]
            member = {g: {int(i) for i in np.flatnonzero(rollup == vocab.code(g))} for g in groups}
        places = {}
        for group, under in member.items():
            inside = under if leaves is None else under & leaves
            if not inside:
                places[group] = "outside"
            elif inside == under and not others:
                places[group] = "inside"
            else:
                places[group] = "crossing"
        return _placement(places)
    if grouping not in SUBSET_COLUMNS:
        raise GrainError(f"a subset can't be placed against a grouping by {grouping!r}")
    vocab = subset.schema.vocabulary(grouping)
    others = set(codes) - {grouping}
    admitted = codes.get(grouping)
    places = {}
    for group in groups:
        if admitted is not None and vocab.code(group) not in admitted:
            places[group] = "outside"
        else:
            places[group] = "crossing" if others else "inside"
    return _placement(places)


def _placement(places: dict[str, str]) -> Placement:
    inside = tuple(g for g, p in places.items() if p == "inside")
    crossing = tuple(g for g, p in places.items() if p == "crossing")
    outside = tuple(g for g, p in places.items() if p == "outside")
    if crossing:
        relation = GrainRelation.CROSSING
    elif inside:
        relation = GrainRelation.COVERED
    else:
        relation = GrainRelation.UNREACHABLE
    return Placement(relation, inside, crossing, outside)
