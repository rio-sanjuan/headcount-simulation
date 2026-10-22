"""Fill channels: each open requisition's monthly hazard of filling by each channel.

An opening fills, in a month, by at most one of three channels:

- **external**: an external hire (`F_ext`), scored by the external-fill model;
- **conversion**: a contingent worker converted into the role (`F_conv`), scored by the
  conversion-fill model;
- **transfer**: an employee moving in from another role (`F_tr`), scored by the transfer-fill
  model.

`channel_hazards` scores every requisition row by each channel's model, a channel without one
scoring zero, and zeroes the rows each channel's eligibility rules exclude (the domains
`external_fill`, `conversion_fill`, and `transfer_fill`). A **planner-mix hook** may then reshape
the hazards, as a planner's fill-share targets do for the channels guidance leaves free (M7).
Last, a row whose hazards sum past 1 is **rescaled** to sum to exactly 1, keeping its channels'
proportions, since an opening can't fill twice in a month.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

import numpy as np
from numpy.typing import NDArray

from hcsim.model.core.layouts import Buffer

if TYPE_CHECKING:
    from hcsim.engine.domains import StepContext

Floats = NDArray[np.float64]

CHANNELS: Final = ("external", "conversion", "transfer")
EXTERNAL, CONVERSION, TRANSFER = range(3)
#: Each channel's model field and eligibility domain.
CHANNEL_DOMAINS: Final = ("external_fill", "conversion_fill", "transfer_fill")


class FillMix(Protocol):
    """A planner's reshaping of the channel hazards: (rows, channels) in, the same out."""

    def __call__(self, hazards: Floats, openings: Buffer, step: int) -> Floats: ...


def rescale(hazards: Floats) -> Floats:
    """Rows whose hazards sum past 1 scaled down to sum to 1; the rest as they are."""
    total = hazards.sum(axis=1, keepdims=True)
    result: Floats = np.where(total > 1.0, hazards / np.where(total > 0, total, 1.0), hazards)
    return result


def channel_hazards(openings: Buffer, ctx: StepContext) -> Floats:
    """(rows, channels): each requisition row's monthly hazard of filling by each channel."""
    hazards = np.zeros((len(openings), len(CHANNELS)))
    for k, domain in enumerate(CHANNEL_DOMAINS):
        model = getattr(ctx.config.models, domain)
        if model is None or not len(openings):
            continue
        p = np.asarray(model.compute_probability(openings), dtype=np.float64)
        hazards[:, k] = ctx.config.resolved_eligibility.restrict(domain, openings, p)
    if ctx.config.fill_mix is not None:
        hazards = np.asarray(ctx.config.fill_mix(hazards, openings, ctx.step), dtype=np.float64)
        if hazards.shape != (len(openings), len(CHANNELS)):
            raise ValueError(f"a fill mix must return {(len(openings), len(CHANNELS))} hazards")
        if not np.isfinite(hazards).all() or (hazards < 0).any():
            raise ValueError("a fill mix must return finite, non-negative hazards")
    return rescale(hazards)
