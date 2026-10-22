"""The SBC publication rule for subsets.

A subset run publishes SBC only for the people inside it. Because people move in and out, the
rule is per month: a tranche's expense in a month is published when its holder is inside the
subset that month, and held in run-scoped scratch otherwise. Nothing is dropped. The two
parts sum to the tranche's whole expense, so the accounting identity still holds for every
tranche once both are counted, and the scratch part is what a run reconciles against when
it's combined with others.

The Financial Layer applies the rule (M6.6 and M6.10). Here it's a pure function on arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Floats = NDArray[np.float64]


class PublicationError(ValueError):
    """Expense, holders, or membership that don't line up."""


@dataclass(frozen=True)
class Publication:
    #: Expense per tranche and month while the holder is inside the subset.
    published: Floats
    #: The rest, held in run-scoped scratch.
    scratch: Floats

    @property
    def total(self) -> Floats:
        result: Floats = self.published + self.scratch
        return result


def membership(inside: ArrayLike, holder: ArrayLike) -> NDArray[np.bool_]:
    """Per tranche and month, whether the holder is inside the subset. `inside` is (holders,
    months), and `holder` gives each tranche's row in it."""
    grid = np.asarray(inside, dtype=bool)
    index = np.asarray(holder, dtype=np.int64)
    if grid.ndim != 2 or index.ndim != 1:
        raise PublicationError("inside is (holders, months), and holder one row per tranche")
    if index.size and (index.min() < 0 or index.max() >= len(grid)):
        raise PublicationError(f"holders must lie in 0..{len(grid) - 1}")
    result: NDArray[np.bool_] = grid[index]
    return result


def publish(expense: ArrayLike, holder: ArrayLike, inside: ArrayLike) -> Publication:
    """Split each tranche's monthly expense by its holder's membership."""
    values = np.asarray(expense, dtype=np.float64)
    mask = membership(inside, holder)
    if values.shape != mask.shape:
        raise PublicationError(
            f"expense is {values.shape}, but holders and months give {mask.shape}"
        )
    return Publication(np.where(mask, values, 0.0), np.where(mask, 0.0, values))
