"""The guidance pipeline's shared errors."""

from __future__ import annotations


class ForecastError(ValueError):
    """A forecaster asked to work without the history it needs, or one that fails on it."""
