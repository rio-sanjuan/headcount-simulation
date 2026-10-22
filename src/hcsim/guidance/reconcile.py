"""Reconciliation: making each group's forecasts satisfy the forecast-coherency identity.

Per group and month, the identity asks that

    Ĥ(t) - Ĥ(t-1) = Σ sign_c · flow_c(t)

with Ĥ(-1) the last actual headcount, over the identity's terms (see `fill_tree`). The
methods:

- `greedy` holds the anchor fixed and adjusts the flows. Where a month's identity doesn't hold
  within the tolerance, the gap goes to the smallest flow first; a flow that would turn
  negative is clamped at zero, and what the clamp absorbed moves on to the next. With
  `sign_aware`, the gap first raises the flows that move in its direction, inflows for a rise
  and outflows for a fall, in proportion to their size, so no flow is cut while another could
  grow; only when none moves that way does it fall back to cutting.
- `ols`, `wls`, and `mint` project the forecasts onto the identity: the reconciled forecasts
  are the closest to the base ones, in the metric `W⁻¹`, that satisfy it exactly. OLS weighs
  every value equally; WLS by the inverse of each series' backtest error variance; MinT by the
  inverse of the series' error covariance, shrunk towards its diagonal (Schäfer and Strimmer's
  estimate of the intensity, unless the configuration fixes it). With `anchor_fixed`, the
  default, a projection adjusts only the flows; otherwise the anchor moves too. No value goes
  below zero: one the projection drives negative is fixed at zero and the rest projected
  again, so the identity holds wherever another value can absorb the gap.

Where a method needs backtest errors that a series doesn't have, it falls back: MinT to WLS,
and WLS to OLS, and the method that ran is reported.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from hcsim.guidance.registry import Entry, Registry

Floats = NDArray[np.float64]
#: The key the anchor's backtest errors go under, beside the flows'.
ANCHOR_KEY: Final = "headcount"


@dataclass(frozen=True)
class Problem:
    """One group's forecasts and what reconciling them needs."""

    anchor: Floats
    last_actual: float
    flows: Mapping[str, Floats]
    signs: Mapping[str, int]
    tolerance: float = 1.0
    #: Each series' backtest errors, aligned month by month, the anchor's under `ANCHOR_KEY`.
    residuals: Mapping[str, Floats] = field(default_factory=dict)
    sign_aware: bool = False
    anchor_fixed: bool = True
    #: MinT's shrinkage intensity in [0, 1]; none estimates it.
    shrinkage: float | None = None


@dataclass(frozen=True)
class Reconciled:
    flows: dict[str, Floats]
    #: Per month, what the identity still misses after reconciling.
    residual: Floats
    #: The reconciled anchor; none when it was held fixed.
    anchor: Floats | None = None
    #: The method that ran, which a fallback can make differ from the one asked for.
    method: str = "greedy"


class Reconciler(Protocol):
    def __call__(self, problem: Problem) -> Reconciled: ...


def gaps(
    anchor: Floats, last_actual: float, flows: Mapping[str, Floats], signs: Mapping[str, int]
) -> Floats:
    """Per month, the anchor's change less the signed flows: zero where the identity holds."""
    change = np.diff(np.r_[float(last_actual), np.asarray(anchor, dtype=np.float64)])
    implied = sum(
        (signs[c] * np.asarray(v, dtype=np.float64) for c, v in flows.items()),
        start=np.zeros(len(change)),
    )
    result: Floats = change - implied
    return result


# --- greedy ------------------------------------------------------------------------------------


def reconcile_greedy(
    headcount: Floats,
    last_actual: float,
    flows: dict[str, Floats],
    signs: dict[str, int],
    tolerance: float = 1.0,
    sign_aware: bool = False,
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
        if sign_aware:
            gap = _raise_with(adjusted, signs, t, gap)
        for name in sorted(adjusted, key=lambda c: (adjusted[c][t], c)):
            if abs(gap) < 1e-12:
                break
            old = adjusted[name][t]
            new = max(0.0, old + signs[name] * gap)
            adjusted[name][t] = new
            gap -= signs[name] * (new - old)
        residual[t] = gap
    return Reconciled(adjusted, residual)


def _raise_with(adjusted: dict[str, Floats], signs: Mapping[str, int], t: int, gap: float) -> float:
    """Close `gap` by raising the flows that move in its direction, in proportion to their
    size, or equally when they're all zero. Returns what's left: nothing, unless no flow moves
    that way."""
    direction = 1 if gap > 0 else -1
    movers = [c for c in adjusted if signs[c] == direction]
    if not movers:
        return gap
    sizes = np.array([adjusted[c][t] for c in movers])
    shares = sizes / sizes.sum() if sizes.sum() > 0 else np.full(len(movers), 1 / len(movers))
    for name, share in zip(movers, shares.tolist(), strict=True):
        adjusted[name][t] += share * abs(gap)
    return 0.0


def greedy(problem: Problem) -> Reconciled:
    return reconcile_greedy(
        problem.anchor,
        problem.last_actual,
        dict(problem.flows),
        dict(problem.signs),
        problem.tolerance,
        problem.sign_aware,
    )


# --- projections -------------------------------------------------------------------------------


def _shrunk(errors: Floats, intensity: float | None) -> tuple[Floats, float]:
    """The error covariance shrunk towards its diagonal, and the intensity used: Schäfer and
    Strimmer's estimate when `intensity` is none."""
    months = errors.shape[1]
    centered = errors - errors.mean(axis=1, keepdims=True)
    sample = centered @ centered.T / months
    if intensity is None:
        sd = np.sqrt(np.diag(sample))
        sd = np.where(sd > 0, sd, 1.0)
        standardized = centered / sd[:, None]
        correlation = standardized @ standardized.T / months
        products = np.einsum("it,jt->ijt", standardized, standardized)
        variance = products.var(axis=2) * months / max(months - 1, 1)
        off = ~np.eye(len(errors), dtype=bool)
        denominator = float((correlation[off] ** 2).sum())
        intensity = (
            float(np.clip(variance[off].sum() / denominator, 0.0, 1.0)) if denominator else 1.0
        )
    target = np.diag(np.diag(sample))
    return intensity * target + (1 - intensity) * sample, intensity


def _projected(problem: Problem, weighting: str) -> Reconciled:
    names = list(problem.flows)
    horizon = len(problem.anchor)
    series = names if problem.anchor_fixed else [ANCHOR_KEY, *names]
    base = np.concatenate(
        [np.asarray(problem.flows[n], dtype=np.float64) for n in names]
        if problem.anchor_fixed
        else [np.asarray(problem.anchor, dtype=np.float64)]
        + [np.asarray(problem.flows[n], dtype=np.float64) for n in names]
    )
    # Each month's identity as a row of A x = b over the stacked series, months within series.
    k = len(series)
    a = np.zeros((horizon, k * horizon))
    b = np.zeros(horizon)
    for t in range(horizon):
        for s, name in enumerate(series):
            if name == ANCHOR_KEY:
                a[t, s * horizon + t] = 1.0
                if t > 0:
                    a[t, s * horizon + t - 1] = -1.0
            else:
                a[t, s * horizon + t] = -problem.signs[name]
        if problem.anchor_fixed:
            previous = problem.last_actual if t == 0 else problem.anchor[t - 1]
            b[t] = -(problem.anchor[t] - previous)
        elif t == 0:
            b[t] = problem.last_actual
    method = weighting
    covariance = np.eye(k)
    errors = [problem.residuals.get(name) for name in series]
    usable = all(e is not None and len(e) >= 2 for e in errors)
    if weighting == "mint" and usable:
        stacked = np.vstack([np.asarray(e, dtype=np.float64) for e in errors if e is not None])
        lengths = {len(e) for e in errors if e is not None}
        if len(lengths) == 1:
            covariance, _ = _shrunk(stacked, problem.shrinkage)
        else:
            method = "wls"
    elif weighting == "mint":
        method = "wls"
    if method == "wls":
        if usable:
            variances = np.array(
                [max(float(np.var(np.asarray(e))), 1e-9) for e in errors if e is not None]
            )
            covariance = np.diag(variances)
        else:
            method = "ols"
    w = np.kron(covariance, np.eye(horizon)).astype(np.float64)
    w += 1e-9 * np.eye(len(w))  # keeps W A' invertible when a series never erred
    x = nonnegative_projection(base, w, a, b)
    values = {name: x[s * horizon : (s + 1) * horizon] for s, name in enumerate(series)}
    anchor = None if problem.anchor_fixed else values.pop(ANCHOR_KEY)
    flows = {name: np.maximum(values[name], 0.0) for name in names}
    final = problem.anchor if anchor is None else anchor
    residual = gaps(final, problem.last_actual, flows, problem.signs)
    return Reconciled(flows, residual, anchor, method)


def nonnegative_projection(base: Floats, w: Floats, a: Floats, b: Floats) -> Floats:
    """The projection of `base` onto `a x = b` in the metric `w⁻¹`, with no value below zero.
    A value the projection drives negative is fixed at zero, and the rest are projected again
    onto what the identity then asks of them, until none is negative. A month whose values are
    all fixed keeps the gap it can't close."""
    x = base.astype(np.float64).copy()
    free = np.ones(len(x), dtype=bool)
    for _ in range(len(x) + 1):
        f = np.flatnonzero(free)
        rows = np.flatnonzero(np.abs(a[:, f]).sum(axis=1) > 0)
        if not len(f) or not len(rows):
            break
        af, wf = a[np.ix_(rows, f)], w[np.ix_(f, f)]
        target = b[rows] - a[np.ix_(rows, ~free)] @ x[~free]
        x[f] = base[f] - wf @ af.T @ np.linalg.pinv(af @ wf @ af.T) @ (af @ base[f] - target)
        negative = free & (x < 0)
        if not negative.any():
            break
        x[negative], free[negative] = 0.0, False
    return np.maximum(x, 0.0)


def ols(problem: Problem) -> Reconciled:
    return _projected(problem, "ols")


def wls(problem: Problem) -> Reconciled:
    return _projected(problem, "wls")


def mint(problem: Problem) -> Reconciled:
    return _projected(problem, "mint")


#: The reconciliation methods a configuration can name.
RECONCILERS: Registry[Reconciler] = Registry(
    "reconciliation method",
    (
        Entry("greedy", lambda: greedy),
        Entry("ols", lambda: ols),
        Entry("wls", lambda: wls),
        Entry("mint", lambda: mint),
    ),
)
