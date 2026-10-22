"""The run-mode ladder, in its own module so the configuration and gates can both read it."""

from __future__ import annotations

from enum import StrEnum


class RunMode(StrEnum):
    """The run-mode ladder, from most pinned to least."""

    REPLAY = "replay"
    GUIDED_ACTUALS = "guided_actuals"
    GUIDED_FORECAST = "guided_forecast"
    UNGUIDED = "unguided"
