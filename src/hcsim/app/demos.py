"""The explorer's demos: each one a small run on a world, shaped for a chart.

- **Quota versus independent sampling** (`sampling`): rows with unequal probabilities, drawn
  many times under quota sampling, which realizes exactly the quota, and under independent
  draws whose expected count is the quota. It shows how many rows each draws, and how often each
  row is drawn.
- **The three valves** (`valves`): one plan, driven by each alignment method a run can take:
  none; supply, through the requisitions growth opens; flow, through guidance aligned to the
  plan; and rate, through calibrated domain rates. It shows each run's headcount beside the
  plan, and its bias where the plan is scored.
- **The requisition lifecycle** (`lifecycle`): the open-openings stock, month by month, and
  every flow in and out of it: the requisitions opened by growth, campus, and backfill; those
  that exit unfilled; and the openings each fill channel takes. The opening identity holds
  exactly, and its residual says so.
- **The SBC layer** (`sbc`): the expense a run recognizes each month, by part (the seed's
  equity, the run's own grants, and the offline stream), the grants it makes by award type, and
  one new-hire grant's vesting schedule.

Every demo is a seeded run, so the same request gives the same answer, and each answer is kept,
so asking again is free. Runs read the world through its data, as studies do.
"""

from __future__ import annotations

import datetime as dt
import threading
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Any, Final

import duckdb
import numpy as np
import pandas as pd

from hcsim.engine.loop import SimulationResult
from hcsim.engine.seeding import schema_from_world, seed_state, snapshot_before
from hcsim.guidance.config import GuidanceConfig
from hcsim.guidance.pipeline import run as fit_guidance
from hcsim.model.core.alignment_builders import synthetic
from hcsim.model.core.schema import AWARD_TYPE, COST_CENTER
from hcsim.model.ml.sampling import bernoulli, quota_select
from hcsim.studies.harness import run_study
from hcsim.studies.profile import StudyProfile

#: The alignment methods `valves` runs, and the valve each drives.
VALVES: Final = {
    "none": "no valve",
    "supply": "supply: the requisitions growth opens",
    "demand": "flow: guidance aligned to the plan",
    "rate_calibration": "rate: calibrated domain rates",
}
#: The months `valves` runs and its plan spans.
PLAN_MONTHS: Final = 6
#: The flows into and out of the open-openings stock.
OPENED: Final = ("G", "G_camp", "B_att", "B_tr")
FILLED: Final = ("F_ext", "F_conv", "F_tr", "F_camp", "F_pend")


class DemoError(ValueError):
    """A demo that can't run on its world."""


def _mean(values: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in values]


def sampling(
    rows: int = 200, quota: int = 20, trajectories: int = 500, spread: float = 1.0, seed: int = 0
) -> dict[str, Any]:
    """`trajectories` draws of `rows` rows, whose probabilities vary by `spread` (their
    coefficient of variation) and sum to `quota`, under quota sampling and independent draws."""
    if not 1 <= quota <= rows:
        raise DemoError(f"a quota lies between 1 and the rows, {rows}, not {quota}")
    rng = np.random.default_rng(seed)
    weights = rng.gamma(1 / spread**2, spread**2, rows) if spread > 0 else np.ones(rows)
    p = np.sort(np.minimum(weights * quota / weights.sum(), 1.0))[::-1]
    groups = np.zeros(rows, dtype=np.int64)
    k = np.array([quota], dtype=np.int64)
    counts = {
        "quota": np.zeros(trajectories, np.int64),
        "independent": np.zeros(trajectories, np.int64),
    }
    hits = {"quota": np.zeros(rows), "independent": np.zeros(rows)}
    for t in range(trajectories):
        draws = {"quota": quota_select(p, groups, k, rng).mask, "independent": bernoulli(p, rng)}
        for name, mask in draws.items():
            counts[name][t] = int(mask.sum())
            hits[name] += mask
    low = int(min(c.min() for c in counts.values()))
    high = int(max(c.max() for c in counts.values()))
    bins = list(range(low, high + 1))
    out: dict[str, Any] = {"probabilities": _mean(p), "expected": round(float(p.sum()), 4)}
    out["bins"] = bins
    for name in counts:
        c = counts[name]
        out[name] = {
            "histogram": [int((c == b).sum()) for b in bins],
            "mean": round(float(c.mean()), 4),
            "sd": round(float(c.std(ddof=1)) if trajectories > 1 else 0.0, 4),
            "rates": _mean(hits[name] / trajectories),
        }
    return out


class Explorer:
    """The demos that run on a world, each answer kept. `scratch` holds the plans and guidance
    the valves demo writes."""

    def __init__(self, world: Path, scratch: Path) -> None:
        if not world.is_file():
            raise DemoError(f"no world at {world}")
        self.world = world
        self.scratch = scratch
        self._kept: dict[Hashable, Any] = {}
        self._lock = threading.Lock()
        with duckdb.connect(str(world), read_only=True) as con:
            last = snapshot_before(con, dt.date.max)
        self.run_date = last + dt.timedelta(days=1)

    def _keep(self, key: Hashable, make: Callable[[], Any]) -> Any:
        with self._lock:
            if key not in self._kept:
                self._kept[key] = make()
            return self._kept[key]

    def _run(self, guidance: Path | None = None, **fields: Any) -> SimulationResult:
        profile = StudyProfile.from_mapping(
            {"name": "explorer", "seed": 7, "mode": "unguided", "scope": "hcf"} | fields
        )
        return run_study(profile, self.world, guidance).simulated

    # --- the three valves ---------------------------------------------------------------------

    def _factors(self, growth: float) -> list[float]:
        return [round(1 + growth * (k + 1) / PLAN_MONTHS, 6) for k in range(PLAN_MONTHS)]

    def _aligned(self, growth: float) -> tuple[Path, Path]:
        """A plan rising evenly to `growth` above the seed, saved, and guidance aligned to it."""
        where = self.scratch / f"valves-{growth:g}"
        plan_path, guidance_path = where / "plan", where / "guidance"
        if guidance_path.exists():
            return plan_path, guidance_path
        start = pd.Period(self.run_date, freq="M")
        with duckdb.connect(str(self.world), read_only=True) as con:
            schema = schema_from_world(con)
            labels = schema.vocabulary(COST_CENTER).labels
            state, _ = seed_state(con, schema, self.run_date)
            heads = {labels[c]: float(n) for c, n in enumerate(state.headcount) if n > 0}
            plan = synthetic(
                "relative",
                COST_CENTER,
                heads,
                start=start,
                months=PLAN_MONTHS,
                dates=[str(start + k) for k in range(PLAN_MONTHS)],
                factors=self._factors(growth),
            )
            plan.save(plan_path)
            config = GuidanceConfig.from_mapping(
                {"horizon": PLAN_MONTHS, "alignment": {"method": "greedy"}}
            )
            fit_guidance(con, config, plan).artifact.save(guidance_path)
        return plan_path, guidance_path

    def valves(self, growth: float = 0.05, trajectories: int = 4) -> dict[str, Any]:
        """One plan, rising to `growth` above the seed over six months, driven by each method."""
        return dict(
            self._keep(("valves", growth, trajectories), lambda: self._valves(growth, trajectories))
        )

    def _valves(self, growth: float, trajectories: int) -> dict[str, Any]:
        plan_path, guidance = self._aligned(growth)
        runs: dict[str, dict[str, Any]] = {}
        start_total = 0.0
        for method in VALVES:
            demand = method == "demand"
            fields: dict[str, Any] = {
                "trajectories": trajectories,
                "months": PLAN_MONTHS,
                "mode": "guided_forecast" if demand else "unguided",
                "plan": {"path": str(plan_path)},
                "alignment": {"method": method},
                "evaluators": ["invariants", "plan_alignment"],
            }
            profile = StudyProfile.from_mapping(
                {"name": f"valves-{method}", "seed": 7, "scope": "hcf"} | fields
            )
            run = run_study(profile, self.world, guidance if demand else None)
            heads = run.simulated.headcount.sum(axis=2).mean(axis=0)
            start_total = float(heads[0])
            bias = {m.name: m.value for m in run.metrics if m.evaluator == "plan_alignment"}
            runs[method] = {
                "valve": VALVES[method],
                "headcount": _mean(heads),
                "bias_mean": round(bias.get("bias_mean", float("nan")), 4),
            }
        # A relative plan scales each cost center's seed, so its total scales the seed's total.
        plan = [round(start_total * f, 4) for f in self._factors(growth)]
        return {
            "months": PLAN_MONTHS,
            "run_date": self.run_date.isoformat(),
            "plan": plan,
            "runs": runs,
        }

    # --- the requisition lifecycle ------------------------------------------------------------

    def lifecycle(self, months: int = 12, trajectories: int = 4) -> dict[str, Any]:
        """The open-openings stock and every flow in and out of it, month by month."""
        return dict(
            self._keep(
                ("lifecycle", months, trajectories), lambda: self._lifecycle(months, trajectories)
            )
        )

    def _lifecycle(self, months: int, trajectories: int) -> dict[str, Any]:
        r = self._run(months=months, trajectories=trajectories)
        stock = r.open_openings.sum(axis=2)  # (trajectories, months + 1)

        def per_month(symbol: str) -> np.ndarray:
            flow: np.ndarray = r.flow(symbol).sum(axis=2)
            return flow

        opened = {s: per_month(s) for s in OPENED}
        filled = {s: per_month(s) for s in FILLED}
        unfilled = per_month("U")
        change = stock[:, 1:] - stock[:, :-1]
        residual = change - sum(opened.values()) + unfilled + sum(filled.values())
        return {
            "months": months,
            "run_date": self.run_date.isoformat(),
            "open": _mean(stock.mean(axis=0)),
            "opened": {s: _mean(v.mean(axis=0)) for s, v in opened.items()},
            "filled": {s: _mean(v.mean(axis=0)) for s, v in filled.items()},
            "unfilled": _mean(unfilled.mean(axis=0)),
            "identity_residual": int(np.abs(residual).max(initial=0)),
        }

    # --- the SBC layer ------------------------------------------------------------------------

    def sbc(self, months: int = 12, trajectories: int = 2) -> dict[str, Any]:
        """The expense a run recognizes each month by part, its grants by award type, and one
        new-hire grant's vesting schedule."""
        return dict(
            self._keep(("sbc", months, trajectories), lambda: self._sbc(months, trajectories))
        )

    def _sbc(self, months: int, trajectories: int) -> dict[str, Any]:
        r = self._run(months=months, trajectories=trajectories, scope="hcf+sbc")
        parts: dict[str, np.ndarray] = {}
        grants: dict[str, float] = {}
        example: dict[str, Any] = {}
        labels = r.config.schema.vocabulary(AWARD_TYPE).labels
        for t in r.trajectories:
            if t.sbc is None or t.tranches is None:
                raise DemoError("an SBC run kept no SBC")
            window = t.sbc[t.sbc["month"].between(0, months - 1)]
            by = window.groupby(["part", "month"])["recognized"].sum()
            for part in by.index.get_level_values(0).unique():
                series = by.xs(part).reindex(range(months), fill_value=0.0).to_numpy()
                parts[str(part)] = parts.get(str(part), np.zeros(months)) + series / len(
                    r.trajectories
                )
            tr = t.tranches
            awarded = (tr.award_month >= 0) & (tr.award_month < months) & ~tr.cancelled
            keys = pd.DataFrame(
                {
                    "holder": tr.holder[awarded],
                    "month": tr.award_month[awarded],
                    "type": tr.award_type[awarded],
                }
            ).drop_duplicates()
            counts = keys["type"].value_counts()
            for code, n in zip(
                counts.index.to_numpy(dtype=np.int64), counts.to_numpy(), strict=True
            ):
                name = labels[int(code)]
                grants[name] = grants.get(name, 0.0) + float(n) / len(r.trajectories)
            if not example:
                new_hire = labels.index("new_hire") if "new_hire" in labels else -1
                mine = np.flatnonzero(awarded & (tr.award_type == new_hire))
                if len(mine):
                    first = mine[0]
                    same = (tr.holder == tr.holder[first]) & (
                        tr.award_month == tr.award_month[first]
                    )
                    same &= tr.award_type == new_hire
                    rows = np.flatnonzero(same)
                    example = {
                        "award_month": int(tr.award_month[first]),
                        "vest_months": [int(v) for v in tr.vest_month[rows]],
                        "shares": [round(float(s), 4) for s in tr.shares[rows]],
                    }
        return {
            "months": months,
            "run_date": self.run_date.isoformat(),
            "recognized": {p: _mean(v) for p, v in sorted(parts.items())},
            "grants": {k: round(v, 4) for k, v in sorted(grants.items())},
            "example_grant": example,
        }
