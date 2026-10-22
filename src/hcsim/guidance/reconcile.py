"""Greedy reconciliation: hold the headcount anchor fixed and adjust the flows.

Per group and month, the forecast-coherency identity asks that

    Ĥ(t) - Ĥ(t-1) = Σ sign_c · flow_c(t)

with Ĥ(-1) the last actual headcount. Where it doesn't hold within the tolerance, the gap goes
to the smallest flow first. A flow that would turn negative is clamped at zero, and what the
clamp absorbed moves on to the next flow. Whatever no flow can absorb is the residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from hcsim.guidance.registry import Entry, Registry

Floats = NDArray[np.float64]


@dataclass(frozen=True)
class Reconciled:
    flows: dict[str, Floats]
    residual: Floats


def reconcile_greedy(
    headcount: Floats,
    last_actual: float,
    flows: dict[str, Floats],
    signs: dict[str, int],
    tolerance: float = 1.0,
) -> Reconciled:
    anchor = np.asarray(headcount, dtype=np.float64)
    adjusted = {
        name: np.maximum(np.asarray(v, dtype=np.float64), 0.0).copy() for name, v in flows.items()
    }
    residual = np.zeros(len(anchor))
    previous = float(last_actual)
    for t, level in enumerate(anchor):
        gap = (level - previous) - sum(signs[c] * adjusted[c][t] for c in adjusted)
        previous = float(level)
        if abs(gap) <= tolerance:
            continue
        for name in sorted(adjusted, key=lambda c: (adjusted[c][t], c)):
            old = adjusted[name][t]
            new = max(0.0, old + signs[name] * gap)
            adjusted[name][t] = new
            gap -= signs[name] * (new - old)
            if abs(gap) < 1e-12:
                break
        residual[t] = gap
    return Reconciled(adjusted, residual)


class Reconciler(Protocol):
    """Flows reconciled to an anchor, by the forecast-coherency identity."""

    def __call__(
        self,
        headcount: Floats,
        last_actual: float,
        flows: dict[str, Floats],
        signs: dict[str, int],
        tolerance: float = 1.0,
    ) -> Reconciled: ...


#: The reconciliation methods a configuration can name.
RECONCILERS: Registry[Reconciler] = Registry(
    "reconciliation method", (Entry("greedy", lambda: reconcile_greedy),)
)
