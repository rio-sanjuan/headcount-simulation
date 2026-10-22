"""Coherence: each fill-tree parent the sum of its children, in every group and month.

A forecast that carries a fill-tree parent and all its children, such as `hires` with
`external_hires` and `conversion_hires`, must have the parent equal the children's sum: the
artifact refuses anything else. Each series is forecast on its own, so a coherence pass makes
them agree, by one of three methods:

- `bottom_up`: each parent becomes its children's sum, deepest parents first, so the leaves'
  information carries up the tree;
- `top_down`: each parent's children are scaled to sum to it, shallowest first, keeping their
  shares, and shared equally when they sum to zero;
- `ols`: the smallest change, in least squares, to every carried node of the group's tree
  that makes each parent its children's sum, solved for all parents at once, month by month,
  with no value below zero.

**The four passes.** The pipeline runs a pass at each stage that can leave the tree
incoherent:

1. after generation, by the configured method, since each series was forecast alone;
2. after reconciliation, bottom up, since reconciliation adjusts the identity's terms, the
   deepest carried cover, and the parents above them must follow;
3. after alignment, bottom up, for the same reason (alignment arrives with M5.9);
4. at format, bottom up, after negative cells are clamped to zero, which can shift a sum.

A bottom-up pass of non-negative values is non-negative, so the final pass leaves nothing for
the clamp to undo. Each pass reports, per parent, the largest gap it closed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from hcsim.guidance.reconcile import nonnegative_projection
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.fill_tree import FILL_TREE, ancestors

Floats = NDArray[np.float64]
Forecasts = dict[tuple[str, GuidanceCategory], Floats]
Method = Literal["bottom_up", "top_down", "ols"]
METHODS: Final = ("bottom_up", "top_down", "ols")
#: The pipeline's coherence passes, in the order they run.
PASSES: Final = ("generated", "reconciled", "aligned", "final")


def _depth(node: GuidanceCategory) -> int:
    return len(ancestors(node))


def cohere(
    forecasts: Mapping[tuple[str, GuidanceCategory], Floats], method: Method = "bottom_up"
) -> tuple[Forecasts, dict[str, float]]:
    """`forecasts` made coherent by `method`, and per parent, the largest gap closed."""
    if method not in METHODS:
        raise ValueError(f"unknown coherence method {method!r}; known: {', '.join(METHODS)}")
    result: Forecasts = {
        key: np.asarray(v, dtype=np.float64).copy() for key, v in forecasts.items()
    }
    groups = sorted({group for group, _ in result})
    parents = sorted(FILL_TREE, key=_depth, reverse=method != "top_down")
    closed: dict[str, float] = {}
    if method == "ols":
        for group in groups:
            _ols(result, group, closed)
        return result, closed
    for parent in parents:
        children = FILL_TREE[parent]
        for group in groups:
            keys = [(group, parent), *((group, c) for c in children)]
            if not all(key in result for key in keys):
                continue
            whole = result[group, parent]
            parts = [result[group, c] for c in children]
            gap = whole - np.sum(parts, axis=0)
            closed[parent.value] = max(closed.get(parent.value, 0.0), float(np.abs(gap).max()))
            if method == "bottom_up":
                result[group, parent] = np.sum(parts, axis=0)
            else:
                total = np.sum(parts, axis=0)
                for child, part in zip(children, parts, strict=True):
                    share = np.where(
                        total > 0, part / np.where(total > 0, total, 1.0), 1 / len(parts)
                    )
                    result[group, child] = share * whole
    return result, closed


def _ols(result: Forecasts, group: str, closed: dict[str, float]) -> None:
    """Project the group's carried tree onto coherence, month by month, in place."""
    constrained = [
        parent
        for parent, children in FILL_TREE.items()
        if all((group, node) in result for node in (parent, *children))
    ]
    if not constrained:
        return
    nodes = sorted({n for p in constrained for n in (p, *FILL_TREE[p])}, key=lambda n: n.value)
    index = {node: k for k, node in enumerate(nodes)}
    a = np.zeros((len(constrained), len(nodes)))
    for row, parent in enumerate(constrained):
        a[row, index[parent]] = 1.0
        for child in FILL_TREE[parent]:
            a[row, index[child]] = -1.0
        gap = result[group, parent] - sum(result[group, c] for c in FILL_TREE[parent])
        closed[parent.value] = max(closed.get(parent.value, 0.0), float(np.abs(gap).max()))
    months = len(result[group, nodes[0]])
    for t in range(months):
        base = np.array([result[group, node][t] for node in nodes])
        x = nonnegative_projection(base, np.eye(len(nodes)), a, np.zeros(len(constrained)))
        for node in nodes:
            result[group, node][t] = x[index[node]]
