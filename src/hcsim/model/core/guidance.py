"""The guidance forecast artifact: event counts per group and month, read as quotas.

The artifact is long-form, one row per month, group, and category, with a JSON sidecar. It's
written by the offline pipeline, which never sees the simulation schema, and validated against
the schema when a run builds its aligner.

Beyond the point value, an artifact may carry what its models know of uncertainty, and only
what they know: the pipeline never fabricates it.

- **Quantiles**: columns `q<level>`, for levels 1 to 99, such as `q10`, `q50`, and `q90`. A
  row's quantiles never decrease as the level rises.
- **A distribution**: a `distribution` family per row, with its `mean` and, for families that
  have one, its `dispersion`. Poisson has none, the negative binomial's is its size, and the
  normal's is its standard deviation.
- **Sample paths**: an array of shape (series, paths, horizon), saved beside the frame, with
  an index giving each series' group and category.

An artifact records the population it was built for, as a stamp (`name@digest`), and the
alignment target it was aligned to, if any. Wherever it carries a fill-tree parent and all of
the parent's parts, they must agree cell by cell: a load that finds them apart is refused,
naming the worst cells, since domains reading the parts and the parent would realize different
counts. `combine` joins artifacts that share their identity and cover different cells.

Three refusals keep a malformed artifact from loading and then silently realizing nothing:
a date-typed `forecast_month` (every step lookup would miss), a name that isn't a guidance
category (the series would never be read), and a negative cell (writers clamp at write time,
so a negative value means a defect upstream).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.model.core.aligner import GroupAligner
from hcsim.model.core.categories import GuidanceCategory, parse_category
from hcsim.model.core.fill_tree import FILL_TREE
from hcsim.model.core.schema import SimulationSchema
from hcsim.model.data.population import (
    PopulationError,
    PopulationSpec,
    PopulationStamp,
    guard,
    population,
)
from hcsim.model.interface import ProbabilisticOutput

FORECAST_MONTH: Final = "forecast_month"
CATEGORY: Final = "category"
VALUE: Final = "value"
DISTRIBUTION: Final = "distribution"
MEAN: Final = "mean"
DISPERSION: Final = "dispersion"
QUANTILE: Final = re.compile(r"q([1-9][0-9]?)")
#: Each distribution family, and whether it takes a dispersion.
FAMILIES: Final[Mapping[str, bool]] = {"poisson": False, "negative_binomial": True, "normal": True}
FRAME_FILE: Final = "forecast.parquet"
PATHS_FILE: Final = "paths.npy"
SIDECAR_FILE: Final = "metadata.json"
FORMAT_VERSION: Final = 2
#: How far a fill-tree parent may sit from the sum of its parts, relative to its size.
COHERENCE_TOLERANCE: Final = 1e-6
#: Cells a coherence failure names.
WORST_CELLS: Final = 5


class GuidanceArtifactError(ValueError):
    """A guidance artifact that can't be trusted."""


@dataclass(frozen=True)
class GuidanceForecast:
    """A validated guidance forecast for one grouping column."""

    frame: pd.DataFrame
    grouping_column: str
    snapshot_date: str
    horizon: int
    #: The population's stamp, `name@digest`, or a built-in population's name.
    population: str | None = None
    #: (series, paths, horizon) sample paths, and each series' (group, category).
    sample_paths: NDArray[np.float64] | None = None
    path_index: tuple[tuple[str, str], ...] = ()
    #: The digest of the alignment target the forecast was aligned to.
    aligned_to: str | None = None
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
                f"{int((values < 0).sum())} negative cells, such as "
                f"{_cells(frame, values < 0, values, self.grouping_column)}; writers clamp at "
                "write time, so a negative cell means a defect upstream"
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
        extras = self._check_quantiles(frame) + self._check_distribution(frame)
        object.__setattr__(self, "frame", frame[required + extras].reset_index(drop=True))
        object.__setattr__(self, "groups", tuple(sorted(frame[self.grouping_column].unique())))
        self._check_paths()
        object.__setattr__(self, "population", _stamp_text(self.population))
        if self.aligned_to is not None and (
            not self.aligned_to or self.aligned_to.split() != [self.aligned_to]
        ):
            raise GuidanceArtifactError(f"aligned_to must be a digest, not {self.aligned_to!r}")
        self._check_coherence()

    def _check_coherence(self) -> None:
        carried = set(self.frame[CATEGORY])
        gaps = []
        for parent, children in FILL_TREE.items():
            names = [parent.value, *(c.value for c in children)]
            if not set(names) <= carried:
                continue
            rows = self.frame[self.frame[CATEGORY].isin(names)]
            table = rows.pivot_table(
                index=[FORECAST_MONTH, self.grouping_column],
                columns=CATEGORY,
                values=VALUE,
                aggfunc="sum",
                fill_value=0.0,
            )
            whole = table[parent.value]
            parts = table[names[1:]].sum(axis=1)
            gap = (whole - parts).abs()
            bad = gap > COHERENCE_TOLERANCE * np.maximum(1.0, whole.abs())
            flagged = gap[bad]
            for (step, group), size, w, part in zip(
                flagged.index,
                flagged.to_numpy(),
                whole[bad].to_numpy(),
                parts[bad].to_numpy(),
                strict=True,
            ):
                gaps.append(
                    (float(size), parent.value, int(step), str(group), float(w), float(part))
                )
        if gaps:
            worst = sorted(gaps, reverse=True)[:WORST_CELLS]
            named = "; ".join(
                f"{p} at step {t} in {g}: {w:g} against {s:g} from its parts"
                for _, p, t, g, w, s in worst
            )
            raise GuidanceArtifactError(
                f"{len(gaps)} cells where a fill-tree parent isn't the sum of its parts. "
                f"The worst: {named}"
            )

    @property
    def population_stamp(self) -> PopulationStamp | None:
        return None if self.population is None else PopulationStamp.parse(self.population)

    def check_population(self, expected: PopulationSpec) -> None:
        """Refuse the artifact for a run of another population."""
        try:
            guard(expected, self.population, "guidance forecast")
        except PopulationError as error:
            raise GuidanceArtifactError(str(error)) from None

    @classmethod
    def combine(cls, artifacts: Sequence[GuidanceForecast]) -> GuidanceForecast:
        """One artifact from several that share grouping, snapshot, horizon, population, and
        alignment, and cover different (group, category) series."""
        if not artifacts:
            raise GuidanceArtifactError("nothing to combine")
        first = artifacts[0]
        identity = (
            first.grouping_column,
            first.snapshot_date,
            first.horizon,
            first.population,
            first.aligned_to,
        )
        problems = []
        for k, other in enumerate(artifacts[1:], start=1):
            theirs = (
                other.grouping_column,
                other.snapshot_date,
                other.horizon,
                other.population,
                other.aligned_to,
            )
            if theirs != identity:
                problems.append(
                    f"artifact {k} differs in grouping, snapshot, horizon, population, or alignment"
                )
            if list(other.frame.columns) != list(first.frame.columns):
                problems.append(f"artifact {k} carries other columns")
            if (other.sample_paths is None) != (first.sample_paths is None):
                problems.append(f"artifact {k} differs in carrying sample paths")
        series = [
            set(zip(a.frame[a.grouping_column], a.frame[CATEGORY], strict=True)) for a in artifacts
        ]
        seen: set[tuple[str, str]] = set()
        overlap: set[tuple[str, str]] = set()
        for found in series:
            overlap |= seen & found
            seen |= found
        if overlap:
            problems.append(f"series carried twice: {sorted(overlap)[:5]}")
        if problems:
            raise GuidanceArtifactError("can't combine: " + "; ".join(problems))
        paths = None
        if first.sample_paths is not None:
            try:
                paths = np.concatenate(
                    [a.sample_paths for a in artifacts if a.sample_paths is not None]
                )
            except ValueError:
                raise GuidanceArtifactError("can't combine: sample paths differ in count") from None
        return cls(
            pd.concat([a.frame for a in artifacts], ignore_index=True),
            first.grouping_column,
            first.snapshot_date,
            first.horizon,
            first.population,
            paths,
            tuple(pair for a in artifacts for pair in a.path_index),
            first.aligned_to,
        )

    def _check_quantiles(self, frame: pd.DataFrame) -> list[str]:
        columns = sorted(
            (c for c in frame.columns if QUANTILE.fullmatch(str(c))),
            key=lambda c: int(str(c)[1:]),
        )
        if not columns:
            return []
        values = frame[columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise GuidanceArtifactError("quantiles must be finite")
        if (values < 0).any():
            lowest = values.min(axis=1)
            raise GuidanceArtifactError(
                f"{int((values < 0).sum())} negative quantiles, such as "
                f"{_cells(frame, lowest < 0, lowest, self.grouping_column)}; writers clamp at "
                "write time"
            )
        crossing = int((np.diff(values, axis=1) < 0).any(axis=1).sum())
        if crossing:
            raise GuidanceArtifactError(f"{crossing} rows have quantiles that fall as levels rise")
        frame[columns] = values
        return columns

    def _check_distribution(self, frame: pd.DataFrame) -> list[str]:
        if DISTRIBUTION not in frame.columns:
            extra = [c for c in (MEAN, DISPERSION) if c in frame.columns]
            if extra:
                raise GuidanceArtifactError(f"{extra} need a {DISTRIBUTION!r} column")
            return []
        if MEAN not in frame.columns:
            raise GuidanceArtifactError(f"a {DISTRIBUTION!r} needs a {MEAN!r} column")
        family = frame[DISTRIBUTION].astype(str)
        unknown = sorted(set(family) - set(FAMILIES))
        if unknown:
            raise GuidanceArtifactError(
                f"unknown distributions {unknown}; known: {', '.join(FAMILIES)}"
            )
        mean = frame[MEAN].to_numpy(dtype=np.float64)
        if not np.isfinite(mean).all() or (mean < 0).any():
            raise GuidanceArtifactError("distribution means must be finite and non-negative")
        spread = (
            frame[DISPERSION].to_numpy(dtype=np.float64)
            if DISPERSION in frame.columns
            else np.full(len(frame), np.nan)
        )
        needs = family.map(FAMILIES).to_numpy(dtype=bool)
        if needs.any() and not (np.isfinite(spread[needs]) & (spread[needs] > 0)).all():
            raise GuidanceArtifactError("negative binomial and normal rows need a dispersion > 0")
        if (~needs & ~np.isnan(spread)).any():
            raise GuidanceArtifactError("poisson rows take no dispersion")
        frame[DISTRIBUTION], frame[MEAN], frame[DISPERSION] = family, mean, spread
        return [DISTRIBUTION, MEAN, DISPERSION]

    def _check_paths(self) -> None:
        if self.sample_paths is None:
            if self.path_index:
                raise GuidanceArtifactError("a path index needs sample paths")
            return
        paths = np.asarray(self.sample_paths, dtype=np.float64)
        if paths.ndim != 3 or paths.shape[2] != self.horizon:
            raise GuidanceArtifactError(
                f"sample paths must be (series, paths, {self.horizon}), not {paths.shape}"
            )
        index = tuple((str(g), str(c)) for g, c in self.path_index)
        if len(index) != paths.shape[0] or len(set(index)) != len(index):
            raise GuidanceArtifactError(
                "the path index needs one distinct (group, category) per series"
            )
        for _, category in index:
            parse_category(category)
        if not np.isfinite(paths).all():
            raise GuidanceArtifactError("sample paths must be finite")
        if (paths < 0).any():
            raise GuidanceArtifactError(
                f"{int((paths < 0).sum())} negative path values; writers clamp at write time"
            )
        object.__setattr__(self, "sample_paths", paths)
        object.__setattr__(self, "path_index", index)

    # --- reading ------------------------------------------------------------------------------

    @property
    def categories(self) -> frozenset[GuidanceCategory]:
        return frozenset(parse_category(c) for c in self.frame[CATEGORY].unique())

    def covers(self, category: GuidanceCategory) -> bool:
        return category in self.categories

    @property
    def outputs(self) -> frozenset[ProbabilisticOutput]:
        """What the artifact carries beyond its point values."""
        found = set()
        if self.quantile_levels:
            found.add(ProbabilisticOutput.QUANTILES)
        if DISTRIBUTION in self.frame.columns:
            found.add(ProbabilisticOutput.PARAMETRIC)
        if self.sample_paths is not None:
            found.add(ProbabilisticOutput.SAMPLE_PATHS)
        return frozenset(found) or frozenset({ProbabilisticOutput.NONE})

    @property
    def quantile_levels(self) -> tuple[int, ...]:
        return tuple(int(str(c)[1:]) for c in self.frame.columns if QUANTILE.fullmatch(str(c)))

    def get_quantile(
        self, step: int, category: GuidanceCategory, level: int
    ) -> NDArray[np.float64]:
        """One quantile per group, in group order. Absent cells read 0."""
        column = f"q{level}"
        if column not in self.frame.columns:
            raise GuidanceArtifactError(
                f"no quantile {level}; carried: {list(self.quantile_levels)}"
            )
        return self._per_group(step, category, column)

    def get_distribution(self, step: int, category: GuidanceCategory) -> pd.DataFrame:
        """Each group's distribution family, mean, and dispersion, in group order."""
        if DISTRIBUTION not in self.frame.columns:
            raise GuidanceArtifactError("the artifact carries no distributions")
        rows = self.frame[
            (self.frame[FORECAST_MONTH] == step) & (self.frame[CATEGORY] == category.value)
        ]
        return rows.set_index(self.grouping_column)[[DISTRIBUTION, MEAN, DISPERSION]].reindex(
            list(self.groups)
        )

    def get_paths(self, group: str, category: GuidanceCategory) -> NDArray[np.float64]:
        """One series' sample paths, (paths, horizon)."""
        if self.sample_paths is None:
            raise GuidanceArtifactError("the artifact carries no sample paths")
        try:
            row = self.path_index.index((group, category.value))
        except ValueError:
            raise GuidanceArtifactError(
                f"no sample paths for {group!r}, {category.value}"
            ) from None
        result: NDArray[np.float64] = self.sample_paths[row]
        return result

    def _per_group(self, step: int, category: GuidanceCategory, column: str) -> NDArray[np.float64]:
        rows = self.frame[
            (self.frame[FORECAST_MONTH] == step) & (self.frame[CATEGORY] == category.value)
        ]
        summed = rows.groupby(self.grouping_column)[column].sum()
        return summed.reindex(list(self.groups), fill_value=0.0).to_numpy(dtype=np.float64)

    def get_quota(self, step: int, category: GuidanceCategory) -> NDArray[np.float64]:
        """One count per group, in group order. Duplicate rows sum and absent cells read 0."""
        return self._per_group(step, category, VALUE)

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
            "outputs": sorted(o.value for o in self.outputs),
        }
        if self.quantile_levels:
            data["quantiles"] = list(self.quantile_levels)
        if self.sample_paths is not None:
            data["path_index"] = [list(pair) for pair in self.path_index]
            data["paths_shape"] = list(self.sample_paths.shape)
        if self.population is not None:
            data["population"] = self.population
        if self.aligned_to is not None:
            data["aligned_to"] = self.aligned_to
        return data

    def fingerprint(self) -> str:
        """`gf2-` and 16 hex characters of a SHA-256 over the metadata and an order-insensitive
        digest of the rows. The population enters only when recorded."""
        # Every column is a sort key, so rows that share a cell still sort the same way.
        ordered = self.frame.sort_values(list(self.frame.columns), kind="stable")
        rows = ordered.to_csv(index=False, float_format="%.17g")
        payload = json.dumps(self.metadata(), sort_keys=True) + "\n" + rows
        h = hashlib.sha256(payload.encode())
        if self.sample_paths is not None:
            h.update(np.ascontiguousarray(self.sample_paths, dtype="<f8").tobytes())
        return "gf2-" + h.hexdigest()[:16]

    # --- persistence --------------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        duckdb.from_df(self.frame).write_parquet(str(directory / FRAME_FILE))
        if self.sample_paths is not None:
            np.save(directory / PATHS_FILE, self.sample_paths, allow_pickle=False)
        sidecar = self.metadata() | {"fingerprint": self.fingerprint()}
        (directory / SIDECAR_FILE).write_text(json.dumps(sidecar, indent=2, sort_keys=True))

    @classmethod
    def load(cls, directory: Path) -> GuidanceForecast:
        meta: Mapping[str, Any] = json.loads((directory / SIDECAR_FILE).read_text())
        if meta.get("format_version") != FORMAT_VERSION:
            raise GuidanceArtifactError(f"unsupported format_version {meta.get('format_version')}")
        (column,) = meta["grouping_columns"]
        frame = duckdb.read_parquet(str(directory / FRAME_FILE)).df()
        paths = (
            np.load(directory / PATHS_FILE, allow_pickle=False) if "path_index" in meta else None
        )
        index: Sequence[Sequence[str]] = meta.get("path_index", ())
        artifact = cls(
            frame,
            column,
            meta["snapshot_date"],
            int(meta["horizon"]),
            meta.get("population"),
            paths,
            tuple((str(g), str(c)) for g, c in index),
            meta.get("aligned_to"),
        )
        if artifact.fingerprint() != meta["fingerprint"]:
            raise GuidanceArtifactError("the frame doesn't match its recorded fingerprint")
        return artifact


def _stamp_text(value: str | None) -> str | None:
    """A population as its stamp: `name@digest` as given, or a built-in name resolved."""
    if value is None:
        return None
    try:
        if "@" in value:
            return str(PopulationStamp.parse(value))
        return str(population(value).stamp)
    except PopulationError as error:
        raise GuidanceArtifactError(str(error)) from None


def _cells(
    frame: pd.DataFrame, mask: NDArray[np.bool_], values: NDArray[np.float64], group: str
) -> str:
    """The most negative cells under `mask`, by step, group, and category."""
    rows = np.flatnonzero(mask)
    worst = rows[np.argsort(values[rows], kind="stable")][:WORST_CELLS]
    return ", ".join(
        f"{frame[CATEGORY].iloc[i]} at step {frame[FORECAST_MONTH].iloc[i]} in "
        f"{frame[group].iloc[i]} ({values[i]:g})"
        for i in worst
    )
