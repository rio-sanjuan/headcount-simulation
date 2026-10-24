"""The study catalog: the studies that measure the simulator, declared once.

Each entry (`Entry`) is a study profile that's written out for a world (`write`). Writing sets
the profile's run date from the world's last snapshot: a closed window that ends there, for a
study that scores against history, or the month after it, for a forecast. It also writes
beside the profile whatever inputs it reads: a plan, the guidance fitted through the world's
end, and guidance aligned to the plan. The manifest it writes names each study's profile, the
guidance to pass, and the exit code `hcsim study run` documents for it (see `studies.cli`) on
the reference world, the `tiny` preset at seed 5. `run` runs every study through the command
and checks each exit code against the documented one.

The studies:

- `fidelity_ceiling`: a replay of a closed window. It reproduces history, so it bounds how
  closely any run can match it.
- `sbc_accuracy_replay`: a replay at hcf+sbc, scored against the reference ledger. A checkpoint
  (`studies.checkpoint`) records its accuracy and checks later runs against it.
- `variation`: the study's own seed against three others, which shows how far results move with
  the seed alone.
- `dispersion`: how widely many trajectories spread, unguided, beside a guided forecast whose
  quotas fix its counts.
- `regime_comparison`: supply alignment under each supply rule, scored by plan alignment, whose
  table breaks every target down by valve and route.
- `backtest`: an unguided forecast of a closed window, scored against history, with a replay
  for reference.
- `domain_sweeps`: promotion's gate probability and transfers' availability swept, scored by
  each domain's realized rate.
- `subsets`: two cost centers run on their own.
- seven alignment studies: one under each method (`alignment_none`, `alignment_supply`,
  `alignment_demand`, `alignment_hybrid`, and `alignment_rate_calibration`), one whose arms
  take different methods (`alignment_per_arm`), and one shaped by fill shares and a tilt
  (`alignment_planner`).
- `award_sizing`: the grants a run sizes beside history's recorded awards, with a replay for
  reference.
- `why_quota_sampling`: quota sampling against the independent-sampling baseline, under guided
  actuals, scored against history.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import duckdb
import pandas as pd
import yaml

from hcsim.engine.seeding import schema_from_world, seed_state, snapshot_before
from hcsim.guidance.config import GuidanceConfig
from hcsim.guidance.pipeline import run as fit_guidance
from hcsim.model.core.alignment_builders import synthetic
from hcsim.model.core.schema import COST_CENTER
from hcsim.studies.profile import StudyProfile

MANIFEST: Final = "catalog.json"
#: The awards of the reference world follow their events by up to two months.
REPLAY_LAG: Final = {"new_hire": 2, "promotion": 2, "transfer": 2}
#: The months a forecast study runs, and a closed window spans.
MONTHS: Final = 6


class Window(StrEnum):
    """Where a study's months lie: a closed window that ends at the world's last snapshot, or
    a forecast from the month after it."""

    CLOSED = "closed"
    FORECAST = "forecast"


class Guidance(StrEnum):
    """The guidance a study is given: none, the guidance fitted through the world's end, or
    guidance aligned to the study's plan."""

    NONE = "none"
    FITTED = "fitted"
    ALIGNED = "aligned"


@dataclass(frozen=True)
class Entry:
    """A catalog study: its profile, less what writing it for a world sets, and what it
    needs."""

    name: str
    doc: str
    profile: Mapping[str, Any]
    window: Window = Window.FORECAST
    guidance: Guidance = Guidance.NONE
    #: Whether its plan is the saved one guidance is aligned to, rather than its own.
    saved_plan: bool = False
    #: How far above the seed its own plan rises by the window's end.
    growth: float = 0.05
    #: The exit code `hcsim study run` documents for it on the reference world.
    expect: int = 0
    #: Fields that need the world: each given its labels by cost center, in code order.
    fields: Callable[[Sequence[str]], Mapping[str, Any]] = field(default=lambda _: {})


def _plan(start: pd.Period, growth: float = 0.05) -> dict[str, Any]:
    """A plan with a target every month, rising evenly from the seed to `growth` above it by
    the window's last month."""
    dates = [str(start + k) for k in range(MONTHS)]
    factors = [round(1 + growth * (k + 1) / MONTHS, 6) for k in range(MONTHS)]
    return {"synthetic": {"shape": "relative", "dates": dates, "factors": factors}}


def _alignment(method: str, **fields: Any) -> dict[str, Any]:
    return {
        "scope": "hcf",
        "mode": "unguided",
        "alignment": {"method": method},
        "evaluators": ["invariants", "plan_alignment"],
    } | fields


_SCORED = ["invariants", "fidelity", "domain", "dispersion"]
#: Every study in the catalog.
ENTRIES: Final = (
    Entry(
        "fidelity_ceiling",
        "a replay of a closed window, which reproduces history: the ceiling on fidelity",
        {"mode": "replay", "scope": "hcf", "trajectories": 2, "evaluators": _SCORED},
        Window.CLOSED,
    ),
    Entry(
        "sbc_accuracy_replay",
        "a replay at hcf+sbc, its SBC scored against the reference ledger",
        {
            "mode": "replay",
            "scope": "hcf+sbc",
            "trajectories": 1,
            "replay_lag": REPLAY_LAG,
            "evaluators": ["invariants", "fidelity", "sbc", "award_grants"],
        },
        Window.CLOSED,
    ),
    Entry(
        "variation",
        "the study's own seed against three others: how far results move with the seed alone",
        {
            "mode": "unguided",
            "scope": "hcf",
            "trajectories": 5,
            "months": 12,
            "evaluators": ["invariants", "summary", "dispersion"],
            "arms": [{"name": "seed", "null": True}],
            "arm_groups": [{"name": "reseeded", "vary": {"run.seed": [101, 102, 103]}}],
            "rankings": [{"metric": "summary.headcount_end_mean", "better": "higher"}],
        },
    ),
    Entry(
        "dispersion",
        "how widely many trajectories spread, unguided and, for reference, guided",
        {
            "mode": "unguided",
            "scope": "hcf+sbc",
            "trajectories": 20,
            "months": 12,
            "evaluators": ["invariants", "summary", "dispersion"],
            "arms": [
                {"name": "unguided", "null": True},
                {"name": "guided", "reference": True, "mode": "guided_forecast"},
            ],
        },
        guidance=Guidance.FITTED,
    ),
    Entry(
        "regime_comparison",
        "supply alignment under each supply rule, scored by plan alignment's regimes",
        _alignment(
            "supply",
            arms=[
                {
                    "name": "strength",
                    "null": True,
                    "overrides": {"requisitions.supply_rule": "strength"},
                },
                {
                    "name": "expected_fill",
                    "overrides": {"requisitions.supply_rule": "expected_fill"},
                },
                {"name": "soft_bridge", "overrides": {"requisitions.supply_rule": "soft_bridge"}},
            ],
            rankings=[{"metric": "plan_alignment.bias_mean"}],
            # Half the gain, so the rules part: strength closes half each gap, expected fill
            # opens for the yield, and the soft bridge closes the gap to next month's target.
            settings={"requisitions.supply_gain": 0.5},
        ),
        growth=0.2,  # steep enough that the rules open different counts on a small world
    ),
    Entry(
        "backtest",
        "an unguided forecast of a closed window, against history, with a replay for reference",
        {
            "mode": "unguided",
            "scope": "hcf",
            "trajectories": 5,
            "evaluators": _SCORED,
            "arms": [
                {"name": "forecast", "null": True},
                {"name": "history", "reference": True, "mode": "replay"},
            ],
        },
        Window.CLOSED,
    ),
    Entry(
        "domain_sweeps",
        "promotion's gate probability and transfers' availability swept, scored by domain",
        {
            "mode": "unguided",
            "scope": "hcf",
            "trajectories": 3,
            "evaluators": ["invariants", "domain", "summary"],
            "arms": [{"name": "base", "null": True}],
            "arm_groups": [
                {"name": "gate", "vary": {"promotion.gate_probability": [0.25, 0.5]}},
                {"name": "availability", "vary": {"transfers.availability_factor": [0.25, 0.75]}},
            ],
        },
    ),
    Entry(
        "subsets",
        "two cost centers run on their own",
        {"mode": "unguided", "scope": "hcf+sbc", "evaluators": ["invariants", "summary"]},
        fields=lambda labels: {"subset": {"cost_center": list(labels[:2])}},
    ),
    Entry(
        "alignment_none",
        "a plan scored with no valve driving it",
        _alignment("none"),
    ),
    Entry("alignment_supply", "the plan driving supply", _alignment("supply")),
    Entry(
        "alignment_demand",
        "the plan driving flow, through guidance aligned to it",
        _alignment("demand", mode="guided_forecast"),
        guidance=Guidance.ALIGNED,
        saved_plan=True,
    ),
    Entry(
        "alignment_hybrid",
        "the plan driving supply and flow both, through guidance aligned to it",
        _alignment("hybrid", mode="guided_forecast"),
        guidance=Guidance.ALIGNED,
        saved_plan=True,
    ),
    Entry(
        "alignment_rate_calibration",
        "the plan calibrating the domain rates",
        _alignment("rate_calibration"),
    ),
    Entry(
        "alignment_per_arm",
        "arms that align by different methods, the null by supply",
        _alignment(
            "supply",
            alignment={"method": "supply", "overrides": {"calibrated": "rate_calibration"}},
            arms=[{"name": "supply", "null": True}, {"name": "calibrated"}],
            rankings=[{"metric": "plan_alignment.bias_mean"}],
        ),
    ),
    Entry(
        "alignment_planner",
        "supply alignment shaped by fill shares and an attrition tilt",
        _alignment(
            "supply",
            tilts=[{"domain": "attrition", "multiplier": 0.8, "reason": "a retention program"}],
        ),
        fields=lambda labels: {
            "fill_shares": [
                {
                    "scope": {"cost_center": list(labels[:3])},
                    "shares": {"external": 0.8, "conversion": 0.1, "transfer": 0.1},
                }
            ]
        },
    ),
    Entry(
        "award_sizing",
        "the grants a run sizes beside history's recorded awards, with a replay for reference",
        {
            "mode": "unguided",
            "scope": "hcf+sbc",
            "trajectories": 2,
            "replay_lag": REPLAY_LAG,
            "evaluators": ["invariants", "award_grants", "award_sizing", "sbc"],
            "arms": [
                {"name": "sized", "null": True},
                {"name": "history", "reference": True, "mode": "replay"},
            ],
        },
        Window.CLOSED,
    ),
    Entry(
        "why_quota_sampling",
        "quota sampling against the independent-sampling baseline, under guided actuals",
        {
            "mode": "guided_actuals",
            "scope": "hcf",
            "trajectories": 10,
            "evaluators": ["invariants", "fidelity", "dispersion"],
            "arms": [
                {
                    "name": "independent",
                    "null": True,
                    "overrides": {"sampling.method": "independent"},
                },
                {"name": "quota"},
            ],
            "rankings": [{"metric": "fidelity.attrition_wape"}],
        },
        Window.CLOSED,
    ),
)
#: Each entry by name.
BY_NAME: Final = {e.name: e for e in ENTRIES}


@dataclass(frozen=True)
class Written:
    """One study written for a world: its profile's path, the guidance to pass, if any, and
    its documented exit code."""

    name: str
    profile: Path
    guidance: Path | None
    expect: int

    def args(self, world: Path, out: Path) -> list[str]:
        """`hcsim study run`'s arguments for it."""
        args = ["study", "run", "--profile", str(self.profile), "--world", str(world)]
        if self.guidance is not None:
            args += ["--guidance", str(self.guidance)]
        return [*args, "--out", str(out / self.name)]


def _months_from(start: pd.Period) -> dt.date:
    return start.start_time.date()


def write(world: Path, out: Path, names: Sequence[str] = ()) -> list[Written]:
    """Write every catalog study, or those named, for `world` under `out`: each profile, the
    inputs it reads, and the manifest. Returns what it wrote, in catalog order."""
    unknown = sorted(set(names) - set(BY_NAME))
    if unknown:
        raise ValueError(f"no catalog studies {unknown}; the catalog holds {', '.join(BY_NAME)}")
    entries = [e for e in ENTRIES if not names or e.name in names]
    profiles, inputs = out / "profiles", out / "inputs"
    profiles.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(world), read_only=True) as con:
        last = pd.Period(snapshot_before(con, dt.date.max), freq="M")
        schema = schema_from_world(con)
        labels = schema.vocabulary(COST_CENTER).labels
        forecast, closed = last + 1, last - MONTHS + 1
        needs = {e.guidance for e in entries}
        fitted = inputs / "guidance"
        if Guidance.FITTED in needs:
            fit_guidance(con, GuidanceConfig()).artifact.save(fitted)
        plan_path, aligned = inputs / "plan", inputs / "aligned_guidance"
        if Guidance.ALIGNED in needs:
            state, _ = seed_state(con, schema, _months_from(forecast))
            heads = {labels[c]: float(n) for c, n in enumerate(state.headcount) if n > 0}
            spec = _plan(forecast)["synthetic"]
            plan = synthetic(
                spec["shape"],
                COST_CENTER,
                heads,
                start=forecast,
                months=MONTHS,
                dates=spec["dates"],
                factors=spec["factors"],
            )
            plan.save(plan_path)
            config = GuidanceConfig.from_mapping(
                {"horizon": MONTHS, "alignment": {"method": "greedy"}}
            )
            fit_guidance(con, config, plan).artifact.save(aligned)
    written = []
    for entry in entries:
        start = closed if entry.window is Window.CLOSED else forecast
        data: dict[str, Any] = {"name": entry.name, "seed": 7, "trajectories": 3, "months": MONTHS}
        data |= dict(entry.profile) | dict(entry.fields(labels))
        data["run_date"] = _months_from(start)
        if "alignment" in data and "plan" not in data:
            own = _plan(start, entry.growth)
            data["plan"] = {"path": str(plan_path)} if entry.saved_plan else own
        StudyProfile.from_mapping(data)  # every written profile validates
        path = profiles / f"{entry.name}.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        guidance = {Guidance.NONE: None, Guidance.FITTED: fitted, Guidance.ALIGNED: aligned}
        written.append(Written(entry.name, path, guidance[entry.guidance], entry.expect))
    manifest = [
        {
            "name": w.name,
            "doc": BY_NAME[w.name].doc,
            "profile": str(w.profile),
            "guidance": str(w.guidance) if w.guidance else None,
            "expect": w.expect,
        }
        for w in written
    ]
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return written


def load(out: Path) -> list[Written]:
    """The studies a manifest under `out` names."""
    rows = json.loads((out / MANIFEST).read_text())
    return [
        Written(
            r["name"],
            Path(r["profile"]),
            Path(r["guidance"]) if r["guidance"] else None,
            r["expect"],
        )
        for r in rows
    ]


@dataclass(frozen=True)
class Outcome:
    """A study's run: its documented exit code and the one it gave."""

    name: str
    expect: int
    got: int

    @property
    def matched(self) -> bool:
        return self.expect == self.got


def run(
    written: Sequence[Written], world: Path, out: Path, main: Callable[[Sequence[str]], int]
) -> list[Outcome]:
    """Run each written study through `main`, the `hcsim` command, into `out`."""
    return [Outcome(w.name, w.expect, main(w.args(world, out))) for w in written]
