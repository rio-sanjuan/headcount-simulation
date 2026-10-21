"""The guidance forecast artifact: event counts per group and month, read as quotas.

The artifact is long-form, one row per month, group, and category, with a JSON sidecar. It's
written by the offline pipeline, which never sees the simulation schema, and validated against
the schema when a run builds its aligner.

Three refusals keep a malformed artifact from loading and then silently realizing nothing:
a date-typed `forecast_month` (every step lookup would miss), a name that isn't a guidance
category (the series would never be read), and a negative cell (writers clamp at write time,
so a negative value means a defect upstream).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.model.core.aligner import GroupAligner
from hcsim.model.core.categories import GuidanceCategory, parse_category
from hcsim.model.core.schema import SimulationSchema

FORECAST_MONTH: Final = "forecast_month"
CATEGORY: Final = "category"
VALUE: Final = "value"
FRAME_FILE: Final = "forecast.parquet"
SIDECAR_FILE: Final = "metadata.json"
FORMAT_VERSION: Final = 1


class GuidanceArtifactError(ValueError):
    """A guidance artifact that can't be trusted."""


@dataclass(frozen=True)
class GuidanceForecast:
    """A validated guidance forecast for one grouping column."""

    frame: pd.DataFrame
    grouping_column: str
    snapshot_date: str
    horizon: int
    population: str | None = None
    groups: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        frame = self.frame.copy()
        required = [FORECAST_MONTH, CATEGORY, VALUE, self.grouping_column]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise GuidanceArtifactError(f"missing columns: {missing}")
        if not pd.api.types.is_integer_dtype(frame[FORECAST_MONTH]):
            raise GuidanceArtifactError(
                f"{FORECAST_MONTH} must hold integer step offsets; a date-typed column would "
                "make every quota silently zero"
            )
        for name in frame[CATEGORY].unique():
            parse_category(str(name))
        values = frame[VALUE].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise GuidanceArtifactError("values must be finite")
        if (values < 0).any():
            raise GuidanceArtifactError(
                f"{int((values < 0).sum())} negative cells; writers clamp at write time"
            )
        if self.horizon < 1:
            raise GuidanceArtifactError("horizon must be at least 1")
        steps = frame[FORECAST_MONTH]
        if len(frame) and (steps.min() < 0 or steps.max() >= self.horizon):
            raise GuidanceArtifactError(f"{FORECAST_MONTH} must lie in 0..{self.horizon - 1}")
        frame[self.grouping_column] = frame[self.grouping_column].astype(str)
        frame[CATEGORY] = frame[CATEGORY].astype(str)
        frame[FORECAST_MONTH] = frame[FORECAST_MONTH].astype(np.int64)
        frame[VALUE] = values
        object.__setattr__(self, "frame", frame[required].reset_index(drop=True))
        object.__setattr__(self, "groups", tuple(sorted(frame[self.grouping_column].unique())))

    # --- reading ------------------------------------------------------------------------------

    @property
    def categories(self) -> frozenset[GuidanceCategory]:
        return frozenset(parse_category(c) for c in self.frame[CATEGORY].unique())

    def covers(self, category: GuidanceCategory) -> bool:
        return category in self.categories

    def get_quota(self, step: int, category: GuidanceCategory) -> NDArray[np.float64]:
        """One count per group, in group order. Duplicate rows sum and absent cells read 0."""
        rows = self.frame[
            (self.frame[FORECAST_MONTH] == step) & (self.frame[CATEGORY] == category.value)
        ]
        summed = rows.groupby(self.grouping_column)[VALUE].sum()
        return summed.reindex(list(self.groups), fill_value=0.0).to_numpy(dtype=np.float64)

    def group_coverage(self, category: GuidanceCategory) -> NDArray[np.bool_]:
        """Which groups carry a series for `category`. An uncovered group runs unguided for it,
        rather than being guided to zero."""
        carried = set(self.frame.loc[self.frame[CATEGORY] == category.value, self.grouping_column])
        return np.array([g in carried for g in self.groups], dtype=bool)

    def aligner(self, schema: SimulationSchema) -> GroupAligner:
        """Validate the artifact's groups against `schema` and build the aligner."""
        return GroupAligner(schema, self.grouping_column, self.groups)

    # --- identity -----------------------------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "grouping_columns": [self.grouping_column],
            "categories": sorted(c.value for c in self.categories),
            "snapshot_date": self.snapshot_date,
            "horizon": self.horizon,
            "rows": len(self.frame),
        }
        if self.population is not None:
            data["population"] = self.population
        return data

    def fingerprint(self) -> str:
        """`gf1-` and 16 hex characters of a SHA-256 over the metadata and an order-insensitive
        digest of the rows. The population enters only when recorded."""
        # Every column is a sort key, so rows that share a cell still sort the same way.
        ordered = self.frame.sort_values(
            [CATEGORY, self.grouping_column, FORECAST_MONTH, VALUE], kind="stable"
        )
        rows = ordered.to_csv(index=False, float_format="%.17g")
        payload = json.dumps(self.metadata(), sort_keys=True) + "\n" + rows
        return "gf1-" + hashlib.sha256(payload.encode()).hexdigest()[:16]

    # --- persistence --------------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        duckdb.from_df(self.frame).write_parquet(str(directory / FRAME_FILE))
        sidecar = self.metadata() | {"fingerprint": self.fingerprint()}
        (directory / SIDECAR_FILE).write_text(json.dumps(sidecar, indent=2, sort_keys=True))

    @classmethod
    def load(cls, directory: Path) -> GuidanceForecast:
        meta: Mapping[str, Any] = json.loads((directory / SIDECAR_FILE).read_text())
        if meta.get("format_version") != FORMAT_VERSION:
            raise GuidanceArtifactError(f"unsupported format_version {meta.get('format_version')}")
        (column,) = meta["grouping_columns"]
        frame = duckdb.read_parquet(str(directory / FRAME_FILE)).df()
        artifact = cls(
            frame, column, meta["snapshot_date"], int(meta["horizon"]), meta.get("population")
        )
        if artifact.fingerprint() != meta["fingerprint"]:
            raise GuidanceArtifactError("the frame doesn't match its recorded fingerprint")
        return artifact
