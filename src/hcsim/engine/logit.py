"""The logit shift a planner control applies to a probability, `expit(logit p + δ)`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.float64]


def logit_shift(p: Floats, delta: Floats) -> Floats:
    """`expit(logit p + δ)`, in odds form so a probability of 0 or 1 stays where it is."""
    p = np.asarray(p, dtype=np.float64)
    scaled = p * np.exp(delta)
    out: Floats = np.divide(scaled, 1.0 - p + scaled, out=np.zeros_like(p), where=p > 0)
    return out
