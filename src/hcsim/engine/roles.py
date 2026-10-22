"""Roles for generated requisitions, drawn from an attribute generator.

Growth and campus both open requisitions with no role of their own and draw one from an
attribute generator. A drawn role is *degenerate* when its cost center had too few rows of its
own, so the generator drew from its pooled rows instead, or when it carries a code the schema
lacks. Both domains count their degenerate rows per cost center, as the `degenerate_rows`
diagnostic.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from hcsim.model.core.layouts import Buffer
from hcsim.model.ml.generative import ATTRIBUTES, EmpiricalAttributeGenerator

Ints = NDArray[np.int64]

#: The diagnostic counting generated requisitions with a degenerate role.
DEGENERATE_ROWS: Final = "degenerate_rows"


def fill_roles(
    rows: Buffer, counts: Ints, generator: EmpiricalAttributeGenerator, rng: np.random.Generator
) -> Ints:
    """Draw each new requisition's role from `generator`, in place, `counts[c]` of the rows
    (in cost-center order) for each cost center `c`. Returns the degenerate rows per cost
    center: those drawn from the generator's pooled rows because their cost center had too
    few of its own, or carrying a code the schema lacks."""
    degenerate = np.zeros(len(counts), dtype=np.int64)
    start = 0
    for cc in np.flatnonzero(counts):
        k = int(counts[cc])
        drawn = generator.sample(int(cc), k, rng)
        rows[start : start + k][:, list(ATTRIBUTES)] = drawn
        pooled = not generator.covers(int(cc))
        degenerate[cc] = k if pooled else int((np.asarray(drawn) < 0).any(axis=1).sum())
        start += k
    return degenerate
