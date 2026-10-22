"""Plan vintages: published headcount plans, derived from the latent plan (see `PlanTruth`).

Each vintage is published at the start of a publication month and runs for the horizon. A
cost center's regime in a vintage is drawn afresh, so successive vintages disagree. `planned`
cells are the latent plan times the vintage's noise; `flat` cells hold the regular headcount
at the month-end before publication; `decaying` cells start from the latent plan and decay
toward that headcount.

Then each vintage reallocates targets within each level-1 org, fans each cost center out by
job family (fractional cells, a few of them negative), and adds a hedge per org: a pseudo cost
center, `HEDGE-<org>`, that belongs to no real cost center.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.synth.columns import categorical as label
from hcsim.synth.config import WorldConfig
from hcsim.synth.growth import LatentPlan

REGIMES: Final = ("planned", "flat", "decaying")


def publications(config: WorldConfig) -> list[pd.Period]:
    """Every vintage's publication month, from the world's second year on."""
    first = pd.Period(config.start_month, freq="M")
    months = [first + k for k in range(12, config.months)]
    return [m for m in months if m.month in config.plans.publication_months]


def _reallocate(
    target: NDArray[np.float64], org: NDArray[np.int64], sigma: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Move targets between the cost centers of each org: one factor per cost center, scaled
    so each org's total over the horizon is kept. A factor holds for every month, so each
    cost center's regime keeps its shape."""
    factor = np.exp(rng.normal(0, sigma, target.shape[1]))
    before = np.bincount(org, weights=target.sum(axis=0), minlength=org.max() + 1)
    after = np.bincount(org, weights=(target * factor).sum(axis=0), minlength=org.max() + 1)
    scale = np.divide(before, after, out=np.ones_like(before), where=after > 0)
    result: NDArray[np.float64] = target * (factor * scale[org])
    return result


def _fan_out(
    target: NDArray[np.float64],
    mix: NDArray[np.int64],
    dominant: NDArray[np.int64],
    negative_share: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Split each cost center's target by family, in proportion to its staff at publication
    (all to its dominant family when it has none). In a share of cost centers, one family's
    cell turns negative: more than its whole share moves to another family, and the cost
    center's total stays put."""
    months, n = target.shape
    families = mix.shape[1]
    totals = mix.sum(axis=1, keepdims=True)
    share = np.where(totals > 0, mix / np.maximum(totals, 1), 0.0)
    share[totals[:, 0] == 0, :] = 0.0
    share[totals[:, 0] == 0, dominant[totals[:, 0] == 0]] = 1.0
    cells = target[:, :, None] * share[None, :, :]
    present = (share > 0).sum(axis=1)
    flip = np.flatnonzero((rng.random(n) < negative_share) & (present >= 2))
    for c in flip.tolist():
        donor, receiver = rng.choice(np.flatnonzero(share[c] > 0), size=2, replace=False)
        excess = cells[:, c, donor] * (1 + rng.uniform(0.1, 0.5))
        cells[:, c, donor] -= excess
        cells[:, c, receiver] += excess
    return cells.reshape(months, n * families)


def vintages(
    config: WorldConfig,
    plan: LatentPlan,
    headcount: NDArray[np.int64],
    family_headcount: dict[int, NDArray[np.int64]],
    cost_centers: list[str],
    org_of_cost_center: NDArray[np.int64],
    org_names: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """`plan_headcount`: one row per vintage, cost center, job family, and month, plus each
    level-1 org's hedge row per vintage and month."""
    t = config.plans
    first = pd.Period(config.start_month, freq="M")
    n, families = len(cost_centers), len(config.families)
    dominant = np.arange(n) % families
    # Labels are categoricals built from codes: at scale, strings per row would dwarf the plan.
    names = [str(p) for p in publications(config)]
    places = [*cost_centers, *(f"HEDGE-{o}" for o in org_names)]

    frames = []
    for v, published in enumerate(publications(config)):
        k0 = (published - first).n
        span = np.arange(k0, k0 + t.horizon_months)
        regime = rng.choice(len(REGIMES), size=n, p=list(t.regime_shares))
        noise = np.exp(rng.normal(0, t.vintage_noise_sigma, n))
        latent = plan.target[span]  # (horizon, cost centers)
        base = headcount[k0 - 1].astype(np.float64)  # the month-end before publication
        decay = np.exp(-np.arange(t.horizon_months) / t.decay_months)[:, None]
        target = np.where(
            regime == 0,
            latent * noise,
            np.where(regime == 1, base, base + (latent * noise - base) * decay),
        )
        target = _reallocate(target, org_of_cost_center, t.reallocation_sigma, rng)
        cells = _fan_out(target, family_headcount[k0 - 1], dominant, t.negative_cell_share, rng)
        months = np.array(
            [(first + int(k)).end_time.normalize().date() for k in span], dtype="datetime64[D]"
        )
        rows = n * families * len(span)
        frames.append(
            pd.DataFrame(
                {
                    "vintage": label(np.full(rows, v), names),
                    "published_date": np.full(rows, published.start_time.to_datetime64()),
                    "cost_center": label(
                        np.tile(np.repeat(np.arange(n), families), len(span)), places
                    ),
                    "job_family": label(
                        np.tile(np.arange(families), n * len(span)), list(config.families)
                    ),
                    "month_end": np.repeat(months, n * families),
                    "regime": label(
                        np.tile(np.repeat(regime, families), len(span)), [*REGIMES, "hedge"]
                    ),
                    "is_hedge": False,
                    "target_headcount": cells.ravel().round(4),
                }
            )
        )
        # Hedges: one pseudo cost center per org, holding a share of the org's plan.
        org_total = np.stack(
            [target[:, org_of_cost_center == o].sum(axis=1) for o in range(len(org_names))], axis=1
        )
        hedge = org_total * rng.normal(t.hedge_share, t.hedge_sigma, len(org_names))
        rows = len(org_names) * len(span)
        frames.append(
            pd.DataFrame(
                {
                    "vintage": label(np.full(rows, v), names),
                    "published_date": np.full(rows, published.start_time.to_datetime64()),
                    "cost_center": label(np.tile(n + np.arange(len(org_names)), len(span)), places),
                    "job_family": label(np.full(rows, -1), list(config.families)),
                    "month_end": np.repeat(months, len(org_names)),
                    "regime": label(np.full(rows, len(REGIMES)), [*REGIMES, "hedge"]),
                    "is_hedge": True,
                    "target_headcount": hedge.ravel().round(4),
                }
            )
        )
        # Cells with no plan in them carry nothing, and the fact leaves them out.
        frames = [*frames[:-2], *(f[f["target_headcount"] != 0] for f in frames[-2:])]
    return pd.concat(frames, ignore_index=True)
