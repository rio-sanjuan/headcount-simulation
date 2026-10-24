"""Metric baselines: compare a scorecard with a committed one.

A baseline keeps the scorecard's metrics and the provenance that fixes them: the world, the
profile, the mode, the seed, and the schema. It leaves out what may change without changing
the run, such as the package version. The guidance fingerprint is left out too: it hashes
floats at full precision, which can differ in the last bit across platforms while every metric
stays put.

Any difference is a failure: a metric added, dropped, or moved beyond the tolerance, a changed
verdict, or changed provenance. An intentional change regenerates the baseline in the same PR.
"""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from hcsim.studies import scorecard

KEY: Final = ["pass", "arm", "evaluator", "metric"]
METRIC_COLUMNS: Final = ["value", "threshold", "passed"]
PROVENANCE_COLUMNS: Final = [
    "world_generator",
    "world_preset",
    "world_seed",
    "profile",
    "mode",
    "scope",
    "seed",
    "run_date",
    "schema_fingerprint",
]
COLUMNS: Final = [*KEY, *METRIC_COLUMNS, *PROVENANCE_COLUMNS]
#: Integer metrics match exactly. Float metrics may differ in their last bits across platforms.
RTOL: Final = 1e-9
ATOL: Final = 1e-9


def baseline_frame(card: pd.DataFrame) -> pd.DataFrame:
    return card[COLUMNS].sort_values(KEY, ignore_index=True)


def write(card: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline_frame(card).to_csv(path, index=False, float_format=scorecard.FLOAT_FORMAT)


def load(path: Path) -> pd.DataFrame:
    strings: dict[Hashable, Any] = dict.fromkeys([*PROVENANCE_COLUMNS, "pass", "arm"], str)
    return pd.read_csv(path, dtype=strings, keep_default_na=False, na_values=[""])


def compare(card: pd.DataFrame, baseline: pd.DataFrame) -> list[str]:
    """Every difference between a scorecard and its baseline, one line each."""
    left = baseline_frame(card).set_index(KEY)
    right = baseline.set_index(KEY)

    def name(key: tuple[str, ...]) -> str:
        return "/".join(key)

    differences = [f"{name(k)}: not in the baseline" for k in left.index.difference(right.index)]
    differences += [f"{name(k)}: missing from the run" for k in right.index.difference(left.index)]
    common = left.index.intersection(right.index, sort=True)
    a, b = left.loc[common], right.loc[common]
    for column in ("value", "threshold"):
        now = a[column].to_numpy(dtype=np.float64)
        then = b[column].to_numpy(dtype=np.float64)
        off = ~np.isclose(now, then, rtol=RTOL, atol=ATOL, equal_nan=True)
        differences += [
            f"{name(k)}: {column} is {x!r}, baseline {y!r}"
            for k, x, y in zip(common[off], now[off], then[off], strict=True)
        ]
    for column in ("passed", *PROVENANCE_COLUMNS):
        kind = bool if column == "passed" else str
        now_s, then_s = a[column].astype(kind), b[column].astype(kind)
        off_s = (now_s != then_s).to_numpy()
        differences += [
            f"{name(k)}: {column} is {x!r}, baseline {y!r}"
            for k, x, y in zip(common[off_s], now_s[off_s], then_s[off_s], strict=True)
        ]
    return differences
