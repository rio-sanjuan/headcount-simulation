"""Alignment targets: the plan headcount a run is aligned to, `H*(g, t)`.

An `AlignmentTarget` holds one or more marginals. A marginal gives targets for the groups of one
grouping, such as cost centers or org nodes, at target dates. Between dates, the target's
interpolation says what it is:

- `none`: a target only at its dates;
- `hold_last`: the last target at or before the month;
- `linear`: a straight line between the dates around the month, held after the last.

Before a group's first date it has no target. A target may carry a band, its stated uncertainty
in heads at that date: zero or absent means exact.

The target covers a window of months. Its planned depth is how many of them, from the start,
are actively planned, and the evaluation month `t*` is the last of those: the plan's later
months are for supply, not for scoring.

A target is saved as a manifest and one label-valued file per marginal: a row per group and
target month, with its target and band, in labels rather than codes. Its digest covers what the
target is, not how it was built, so the same targets in any group order share one digest. The
counts of what building normalized away, such as summed duplicates or excluded groups, ride in
the manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Floats = NDArray[np.float64]
MANIFEST_FILE: Final = "manifest.json"
FORMAT_VERSION: Final = 1


class Interpolation(StrEnum):
    NONE = "none"
    HOLD_LAST = "hold_last"
    LINEAR = "linear"


class AlignmentError(ValueError):
    """A target that can't be trusted."""


@dataclass(frozen=True)
class Marginal:
    """Targets for one grouping: `values` and `bands` are (groups, dates), NaN where a group
    has no target at a date."""

    grouping: str
    groups: tuple[str, ...]
    dates: tuple[pd.Period, ...]
    values: Floats
    bands: Floats | None = None
    _months: NDArray[np.int64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        problems = []
        if len(set(self.groups)) != len(self.groups):
            problems.append("groups repeat")
        months = [pd.Period(d, freq="M") for d in self.dates]
        if months != sorted(set(months)):
            problems.append("dates must be distinct months in order")
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != (len(self.groups), len(months)):
            problems.append(f"values are {values.shape}, not {(len(self.groups), len(months))}")
        elif ((values < 0) | np.isinf(values)).any():
            problems.append("targets must be finite and non-negative")
        bands = None if self.bands is None else np.asarray(self.bands, dtype=np.float64)
        if bands is not None:
            if bands.shape != values.shape:
                problems.append("bands need the values' shape")
            elif ((bands < 0) | np.isinf(bands)).any():
                problems.append("bands must be finite and non-negative")
            elif (np.isnan(values) & ~np.isnan(bands)).any():
                problems.append("a band needs a target")
        if problems:
            raise AlignmentError(f"marginal {self.grouping!r}: " + "; ".join(problems))
        object.__setattr__(self, "dates", tuple(months))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "bands", bands)
        ordinal = np.array([m.ordinal for m in months], dtype=np.int64)
        object.__setattr__(self, "_months", ordinal)

    def at(self, month: pd.Period, interpolation: Interpolation) -> Floats:
        """Each group's target in `month`, NaN where it has none."""
        m = pd.Period(month, freq="M").ordinal
        out = np.full(len(self.groups), np.nan)
        for g in range(len(self.groups)):
            known = ~np.isnan(self.values[g])
            dates, values = self._months[known], self.values[g, known]
            if not len(dates) or m < dates[0]:
                continue
            exact = np.flatnonzero(dates == m)
            if exact.size:
                out[g] = values[exact[0]]
            elif interpolation is Interpolation.HOLD_LAST or m > dates[-1]:
                if interpolation is not Interpolation.NONE:
                    out[g] = values[np.searchsorted(dates, m, side="right") - 1]
            elif interpolation is Interpolation.LINEAR:
                out[g] = float(np.interp(m, dates, values))
        return out

    def band_at(self, month: pd.Period) -> Floats:
        """Each group's band at a target date: 0 for exact targets, NaN off the dates."""
        m = pd.Period(month, freq="M").ordinal
        column = np.flatnonzero(self._months == m)
        if not column.size:
            return np.full(len(self.groups), np.nan)
        values = self.values[:, column[0]]
        bands = np.zeros(len(self.groups)) if self.bands is None else self.bands[:, column[0]]
        result: Floats = np.where(np.isnan(values), np.nan, np.nan_to_num(bands, nan=0.0))
        return result


@dataclass(frozen=True)
class AlignmentTarget:
    marginals: tuple[Marginal, ...]
    start: pd.Period
    months: int
    planned_depth: int
    interpolation: Interpolation = Interpolation.HOLD_LAST
    #: What building the target normalized away, by kind, such as summed duplicates.
    normalization: Mapping[str, int] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        start = pd.Period(self.start, freq="M")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "interpolation", Interpolation(self.interpolation))
        object.__setattr__(
            self,
            "normalization",
            MappingProxyType({k: int(v) for k, v in self.normalization.items()}),
        )
        problems = []
        if self.months < 1:
            problems.append("the window needs at least one month")
        if not 1 <= self.planned_depth <= max(self.months, 1):
            problems.append(f"planned depth must lie in 1..{self.months}")
        if not self.marginals:
            problems.append("a target needs at least one marginal")
        groupings = [m.grouping for m in self.marginals]
        if len(set(groupings)) != len(groupings):
            problems.append(f"marginals repeat a grouping: {groupings}")
        end = start + self.months - 1
        for m in self.marginals:
            outside = [str(d) for d in m.dates if not start <= d <= end]
            if outside:
                problems.append(f"marginal {m.grouping!r} has dates outside the window: {outside}")
        if problems:
            raise AlignmentError("bad alignment target: " + "; ".join(problems))

    @property
    def window(self) -> tuple[pd.Period, pd.Period]:
        return self.start, self.start + self.months - 1

    @property
    def evaluation_month(self) -> pd.Period:
        """`t*`: the last actively planned month."""
        return self.start + self.planned_depth - 1

    def is_planned(self, month: pd.Period) -> bool:
        m = pd.Period(month, freq="M")
        return self.start <= m <= self.evaluation_month

    def marginal(self, grouping: str) -> Marginal:
        for m in self.marginals:
            if m.grouping == grouping:
                return m
        known = ", ".join(m.grouping for m in self.marginals)
        raise AlignmentError(f"no marginal for {grouping!r}; known: {known}")

    def target(self, grouping: str, month: pd.Period) -> Floats:
        """`H*(g, month)` for every group of the grouping's marginal."""
        m = pd.Period(month, freq="M")
        first, last = self.window
        if not first <= m <= last:
            raise AlignmentError(f"{m} is outside the window {first}..{last}")
        return self.marginal(grouping).at(m, self.interpolation)

    def digest(self) -> str:
        """`at1-` and 16 hex characters of a SHA-256 over the canonical targets."""
        return (
            "at1-"
            + hashlib.sha256(json.dumps(self._canonical(), sort_keys=True).encode()).hexdigest()[
                :16
            ]
        )

    def _canonical(self) -> dict[str, Any]:
        def number(x: float) -> float | None:
            return None if np.isnan(x) else float(x)

        marginals = []
        for m in sorted(self.marginals, key=lambda m: m.grouping):
            order = sorted(range(len(m.groups)), key=lambda g: m.groups[g])
            bands = m.bands if m.bands is not None else np.full(m.values.shape, np.nan)
            marginals.append(
                {
                    "grouping": m.grouping,
                    "dates": [str(d) for d in m.dates],
                    "groups": [m.groups[g] for g in order],
                    "values": [[number(x) for x in m.values[g]] for g in order],
                    "bands": [[number(x) for x in bands[g]] for g in order],
                }
            )
        return {
            "start": str(self.start),
            "months": self.months,
            "planned_depth": self.planned_depth,
            "interpolation": self.interpolation.value,
            "marginals": marginals,
        }

    def save(self, directory: Path) -> None:
        """Write the manifest and one label-valued file per marginal."""
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        for m in self.marginals:
            name = f"targets_{m.grouping}.csv"
            rows = [
                {
                    "group": m.groups[g],
                    "month": str(m.dates[d]),
                    "target": m.values[g, d],
                    "band": np.nan if m.bands is None else m.bands[g, d],
                }
                for g in range(len(m.groups))
                for d in range(len(m.dates))
                if not np.isnan(m.values[g, d])
            ]
            frame = pd.DataFrame(rows, columns=["group", "month", "target", "band"])
            frame.to_csv(directory / name, index=False, float_format="%.17g")
            files.append({"grouping": m.grouping, "file": name, "rows": len(frame)})
        manifest = {
            "format_version": FORMAT_VERSION,
            "start": str(self.start),
            "months": self.months,
            "planned_depth": self.planned_depth,
            "interpolation": self.interpolation.value,
            "marginals": files,
            "normalization": dict(self.normalization),
            "digest": self.digest(),
        }
        (directory / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

    @classmethod
    def load(cls, directory: Path) -> AlignmentTarget:
        """Read a saved target, refusing another format or a digest that doesn't match."""
        path = directory / MANIFEST_FILE
        if not path.is_file():
            raise AlignmentError(f"no alignment target at {directory}")
        manifest = json.loads(path.read_text())
        if manifest.get("format_version") != FORMAT_VERSION:
            raise AlignmentError(f"unsupported format_version {manifest.get('format_version')}")
        marginals = []
        for entry in manifest["marginals"]:
            frame = pd.read_csv(
                directory / entry["file"],
                dtype={"group": str, "month": str},
                float_precision="round_trip",  # the parser's default can miss by an ulp
            )
            marginals.append(_from_rows(entry["grouping"], frame))
        target = cls(
            tuple(marginals),
            pd.Period(manifest["start"], freq="M"),
            int(manifest["months"]),
            int(manifest["planned_depth"]),
            Interpolation(manifest["interpolation"]),
            manifest.get("normalization", {}),
        )
        if target.digest() != manifest["digest"]:
            raise AlignmentError(f"{directory} doesn't match its recorded digest")
        return target

    def path(self, grouping: str) -> Floats:
        """(groups, months) targets over the whole window."""
        return np.stack([self.target(grouping, self.start + k) for k in range(self.months)], axis=1)


def marginal(
    grouping: str,
    groups: Sequence[str],
    dates: Sequence[str | pd.Period],
    values: Sequence[Sequence[float]],
    bands: Sequence[Sequence[float]] | None = None,
) -> Marginal:
    return Marginal(
        grouping,
        tuple(groups),
        tuple(pd.Period(d, freq="M") for d in dates),
        np.asarray(values, dtype=np.float64),
        None if bands is None else np.asarray(bands, dtype=np.float64),
    )


def _from_rows(grouping: str, frame: pd.DataFrame) -> Marginal:
    """A marginal from label-valued rows: group, month, target, and band."""
    missing = sorted({"group", "month", "target"} - set(frame.columns))
    if missing:
        raise AlignmentError(f"marginal {grouping!r} file lacks columns {missing}")
    duplicated = frame.duplicated(["group", "month"])
    if duplicated.any():
        raise AlignmentError(f"marginal {grouping!r} repeats {int(duplicated.sum())} cells")
    groups = sorted(frame["group"].unique())
    dates = sorted({pd.Period(m, freq="M") for m in frame["month"]})
    values = np.full((len(groups), len(dates)), np.nan)
    bands = np.full_like(values, np.nan)
    row = {g: i for i, g in enumerate(groups)}
    col = {d: j for j, d in enumerate(dates)}
    i = frame["group"].map(row).to_numpy(dtype=np.int64)
    j = np.array([col[pd.Period(m, freq="M")] for m in frame["month"]], dtype=np.int64)
    values[i, j] = frame["target"].to_numpy(dtype=np.float64)
    if "band" in frame.columns:
        bands[i, j] = frame["band"].to_numpy(dtype=np.float64)
    return Marginal(
        grouping, tuple(groups), tuple(dates), values, bands if not np.isnan(bands).all() else None
    )
