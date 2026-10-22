"""Disaggregation: a forecast at a coarse grouping split onto a finer leaf grain.

A forecast fitted at, say, org level 1 can be split onto the cost centers inside each org
node: the leaf grain, which must nest in the forecast's grouping, each leaf in one group. Each
group's forecast of a category is split among its leaves by shares that sum to one, so the
leaves always add up to their group. The shares come from a **basis**: the leaves' own counts
over the last `window_months` of history, by one of five strategies:

- `average_proportions`: each leaf's share of its group, month by month, averaged over the
  window;
- `proportions_of_averages`: each leaf's total over the window, over its group's;
- `last_proportions`: each leaf's share in the last month of history;
- `exposure`: each leaf's share of its group's headcount at the snapshot, for every category,
  as a flow tends to follow the people it happens to;
- `equal`: the group's leaves share equally.

A group whose basis is all zero for a category, as a leaf-level history can be for a rare flow,
falls back to `exposure`, and to `equal` when its headcount is zero too; the report counts the
fallbacks.

**The chunked basis query.** The basis is counted in one query per chunk of `chunk_size`
groups, over the window's months only, so a large world's leaf-level history is never held
whole.

Sample paths split by the same shares, so their quantiles split exactly as the values do.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.guidance.ingest import CATEGORY, MONTH, VALUE, IngestionError, basis_counts
from hcsim.model.core.categories import ANCHOR, GuidanceCategory

Floats = NDArray[np.float64]
Forecasts = dict[tuple[str, GuidanceCategory], Floats]
Strategy = Literal[
    "average_proportions", "proportions_of_averages", "last_proportions", "exposure", "equal"
]
STRATEGIES: Final = (
    "average_proportions",
    "proportions_of_averages",
    "last_proportions",
    "exposure",
    "equal",
)


@dataclass(frozen=True)
class Basis:
    """Each group's leaves, and their counts in the window."""

    leaves: Mapping[str, tuple[str, ...]]
    frame: pd.DataFrame
    chunks: int


def basis(
    con: duckdb.DuckDBPyConnection,
    grouping: str,
    leaf: str,
    groups: Sequence[str],
    categories: Sequence[GuidanceCategory],
    snapshot: pd.Period,
    window_months: int,
    chunk_size: int = 500,
) -> Basis:
    """The leaves' counts for `groups` over the window ending at `snapshot`, counted a chunk
    of groups at a time. Refuses a leaf grain that doesn't nest in the grouping."""
    first = snapshot - window_months + 1
    wanted = list(dict.fromkeys([ANCHOR, *categories]))
    chunks = [list(groups[k : k + chunk_size]) for k in range(0, len(groups), chunk_size)]
    frames = [basis_counts(con, grouping, leaf, wanted, first, snapshot, c) for c in chunks]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frame.empty:
        raise IngestionError(f"no {leaf} leaf has history in the window for these groups")
    parents = frame.groupby("leaf")["group"].nunique()
    straddling = sorted(parents[parents > 1].index)
    if straddling:
        raise IngestionError(
            f"{leaf} doesn't nest in {grouping}: {len(straddling)} leaves, such as "
            f"{straddling[0]!r}, fall in more than one group"
        )
    leaves = {str(g): tuple(sorted(rows["leaf"].unique())) for g, rows in frame.groupby("group")}
    return Basis(leaves, frame, len(chunks))


@dataclass
class Shares:
    """Each (group, category)'s shares over its leaves, and how many fell back."""

    by_key: dict[tuple[str, GuidanceCategory], pd.Series] = field(default_factory=dict)
    fallbacks: dict[str, int] = field(default_factory=dict)


def _normalized(values: pd.Series) -> pd.Series | None:
    total = float(values.sum())
    return values / total if total > 0 and np.isfinite(total) else None


def shares(
    found: Basis,
    keys: Sequence[tuple[str, GuidanceCategory]],
    strategy: Strategy,
    snapshot: pd.Period,
) -> Shares:
    """The shares each forecast key splits by, under `strategy`, with its fallbacks."""
    result = Shares()
    frame = found.frame
    for group, category in keys:
        leaves = list(found.leaves.get(group, ()))
        if not leaves:
            continue
        rows = frame[(frame["group"] == group)]
        table = (
            rows[rows[CATEGORY] == category.value]
            .pivot_table(index=MONTH, columns="leaf", values=VALUE, aggfunc="sum", fill_value=0.0)
            .reindex(columns=leaves, fill_value=0.0)
        )
        stock = rows[(rows[CATEGORY] == ANCHOR.value) & (rows[MONTH] == snapshot)]
        exposure = _normalized(stock.groupby("leaf")[VALUE].sum().reindex(leaves, fill_value=0.0))
        share = _strategy(table, strategy, exposure)
        if share is None:
            fallback = "exposure" if exposure is not None else "equal"
            result.fallbacks[fallback] = result.fallbacks.get(fallback, 0) + 1
            share = exposure
        if share is None:
            share = pd.Series(1.0 / len(leaves), index=leaves)
        result.by_key[group, category] = share.reindex(leaves, fill_value=0.0)
    return result


def _strategy(
    table: pd.DataFrame, strategy: Strategy, exposure: pd.Series | None
) -> pd.Series | None:
    if strategy == "equal":
        return pd.Series(1.0 / len(table.columns), index=table.columns)
    if strategy == "exposure":
        return exposure
    if table.empty:
        return None
    if strategy == "proportions_of_averages":
        return _normalized(table.sum(axis=0))
    if strategy == "last_proportions":
        return _normalized(table.iloc[-1])
    totals = table.sum(axis=1)
    monthly = table[totals > 0].div(totals[totals > 0], axis=0)
    return None if monthly.empty else _normalized(monthly.mean(axis=0))


def split(
    forecasts: Mapping[tuple[str, GuidanceCategory], Floats],
    paths: Mapping[tuple[str, GuidanceCategory], Floats],
    found: Shares,
) -> tuple[Forecasts, Forecasts, list[tuple[str, GuidanceCategory]]]:
    """Each key's forecast and paths split onto its leaves, and the keys no basis covered."""
    leaf_forecasts: Forecasts = {}
    leaf_paths: Forecasts = {}
    uncovered = []
    for key, values in forecasts.items():
        share = found.by_key.get(key)
        if share is None:
            uncovered.append(key)
            continue
        _, category = key
        for leaf, weight in share.items():
            leaf_forecasts[str(leaf), category] = float(weight) * np.asarray(values, float)
            if key in paths:
                leaf_paths[str(leaf), category] = float(weight) * np.asarray(paths[key], float)
    return leaf_forecasts, leaf_paths, uncovered
