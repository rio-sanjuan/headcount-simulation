"""The simulation configuration: one frozen dataclass that holds every parameter of a run.

Validation collects every problem before raising, so one run shows everything to fix. The
configuration is never changed during a run: it's frozen, and live objects inside it (the
schema, the guidance, the models) are treated as read-only.

**Settings and objects.** A field is either a *setting*, a scalar that loads in layers (see
`hcsim.engine.settings`), or an *object* passed in code: the schema, the models, the guidance,
the award settings, the gates, and the eligibility rules, which are resolved against the schema.
Every setting belongs to one *field group*, listed in `GROUPS`, and records it in its metadata.
A setting's name is its group's name, an underscore, and its key, so `promotion_cycle_months` is
the key `cycle_months` of the group `promotion`. The `run` group's settings go unprefixed. The
configuration stays one flat record (ADR 0007), so the whole tunable surface can be listed, and
a group with no settings yet is where a later domain puts its own.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.engine.alignment import AlignmentMethod, supplied
from hcsim.engine.campus import CampusSchedule
from hcsim.engine.ceiling import Ceiling, CeilingError
from hcsim.engine.eligibility import Eligibility, EligibilityError, EligibilityRule, resolve
from hcsim.engine.fills import FillMix
from hcsim.engine.gates import Gate
from hcsim.engine.gates import check as check_gates
from hcsim.engine.modes import RunMode
from hcsim.engine.transfers import ShortfallPolicy
from hcsim.financial.expense import ExpenseConfig
from hcsim.financial.prices import StockPriceModel, covers
from hcsim.financial.routing import AnticipatedAwards, HybridWindow, RecordedAwards
from hcsim.financial.sizing import SizingStrategy
from hcsim.financial.streams import ManualAdjustments, OfflineStream
from hcsim.financial.timing import AnnualCutoff, AwardTiming
from hcsim.model.core.aligner import GroupAligner
from hcsim.model.core.alignment import AlignmentTarget
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.guidance import GuidanceForecast
from hcsim.model.core.schema import (
    AWARD_TYPE,
    COST_CENTER,
    JOB_LEVEL,
    SchemaError,
    SimulationSchema,
)
from hcsim.model.interface import EntityModel, GroupRateModel
from hcsim.model.ml.generative import EmpiricalAttributeGenerator
from hcsim.model.ml.markov import MarkovModel, SimilarityMatrix


class ConfigError(ValueError):
    """A configuration with one or more problems, all listed."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("invalid simulation configuration:\n  - " + "\n  - ".join(problems))


class ForecastScope(StrEnum):
    HEADCOUNT = "hcf"
    HEADCOUNT_AND_SBC = "hcf+sbc"


class BackfillSource(StrEnum):
    """Where backfill's probability per departure comes from."""

    RATE = "rate"
    CLASSIFIER = "classifier"


#: Every field group, in the order a configuration lists them.
GROUPS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "run": "the run's start, length, size, seed, scope, and mode",
        "seeding": "how the seed state is read and how large its buffers are",
        "known_events": "anticipated movements, declared events, and their sanitizing",
        "gates": "registration gates and stochastic step gates",
        "performance": "rating cycles",
        "attrition": "departures",
        "backfill": "requisitions opened to replace departures and transfers out",
        "promotion": "promotion cycles",
        "pending_starts": "accepted offers that start during the run",
        "fills": "fill channels and their hazards",
        "transfers": "matching employees to transfer-assigned openings",
        "campus": "campus openings and starts",
        "cancellation": "requisitions that exit unfilled",
        "requisitions": "growth requisition generation",
        "alignment": "which valve the plan's targets drive",
        "replay": "replaying realized history",
        "results": "what a run records",
    }
)


def _conform(value: Any, hint: Any) -> Any:
    """`value` as `hint`: an enum's value becomes the enum, and a list of integers a tuple.
    Anything else of the wrong type is refused."""
    if isinstance(hint, types.UnionType):
        inner = [a for a in typing.get_args(hint) if a is not type(None)]
        return None if value is None else _conform(value, inner[0])
    if typing.get_origin(hint) is tuple:
        if not isinstance(value, tuple | list) or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        ):
            raise TypeError(f"expected a list of integers, got {value!r}")
        return tuple(value)
    if isinstance(hint, type) and issubclass(hint, StrEnum):
        try:
            return hint(value)
        except ValueError:
            known = ", ".join(m.value for m in hint)
            raise ValueError(f"expected one of {known}, got {value!r}") from None
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    wrong_kind = isinstance(value, bool) and hint is not bool
    wrong_kind |= isinstance(value, dt.datetime) and hint is dt.date
    if wrong_kind or not isinstance(value, hint):
        raise TypeError(f"expected {getattr(hint, '__name__', hint)}, got {value!r}")
    return value


@functools.cache
def _hints() -> dict[str, Any]:
    return typing.get_type_hints(SimulationConfig)


def _when(condition: bool, message: str) -> list[str]:
    return [message] if condition else []


def _months(name: str, values: tuple[int, ...], low: int, high: int) -> list[str]:
    problems = []
    if any(not low <= v <= high for v in values):
        problems.append(f"{name} must lie in {low}..{high}, got {list(values)}")
    if len(set(values)) != len(values):
        problems.append(f"{name} repeat: {list(values)}")
    return problems


def setting(group: str, default: Any, doc: str) -> Any:
    """A field that loads in layers, in `group`."""
    if group not in GROUPS:
        raise ValueError(f"no field group {group!r}")
    return field(default=default, metadata={"group": group, "doc": doc})


@dataclass(frozen=True)
class DomainModels:
    """The models behind the skeleton's domains, and the artifact version each was loaded from,
    where it was."""

    attrition: EntityModel  # scores employee rows
    external_fill: EntityModel  # scores opening rows: a monthly fill hazard
    cancellation: EntityModel  # scores opening rows: a monthly unfilled-exit hazard
    backfill: GroupRateModel | EntityModel  # a rate per cost center, or a classifier
    growth: GroupRateModel  # mean growth requisitions per month, by cost center code
    attributes: EmpiricalAttributeGenerator  # attributes of generated requisitions
    #: Rating transitions and first ratings. Without it, ratings stay as seeded.
    performance: MarkovModel | None = None
    #: Backfill for transfers out. Without it, the departure backfill model serves both.
    backfill_transfer: GroupRateModel | EntityModel | None = None
    #: Scores employee rows for promotion. Without it, only known promotions happen.
    promotion: EntityModel | None = None
    #: How alike job families are, for a promotion that changes family.
    family_similarity: SimilarityMatrix | None = None
    #: Score opening rows for filling by conversion and by transfer. Without one, its channel
    #: never fills.
    conversion_fill: EntityModel | None = None
    transfer_fill: EntityModel | None = None
    #: Roles for campus requisitions; a run with a campus schedule needs one.
    campus_attributes: EmpiricalAttributeGenerator | None = None
    #: Which campus requisitions fill in a start month. Without it, every one does.
    campus_fill: EntityModel | None = None
    #: Artifact versions, by model field.
    versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stray = sorted(set(self.versions) - set(MODEL_FIELDS))
        if stray:
            raise ConfigError([f"versions for models the configuration doesn't hold: {stray}"])
        object.__setattr__(self, "versions", dict(self.versions))

    @property
    def present(self) -> list[str]:
        """The model fields that hold a model."""
        return [name for name in MODEL_FIELDS if getattr(self, name) is not None]

    @property
    def unversioned(self) -> list[str]:
        """The models that weren't loaded from an artifact."""
        return [name for name in self.present if name not in self.versions]


#: The model fields, in order.
MODEL_FIELDS: Final = (
    "attrition",
    "external_fill",
    "cancellation",
    "backfill",
    "growth",
    "attributes",
    "performance",
    "backfill_transfer",
    "promotion",
    "family_similarity",
    "conversion_fill",
    "transfer_fill",
    "campus_attributes",
    "campus_fill",
)
#: The model fields a run may leave empty.
OPTIONAL_MODELS: Final = frozenset(
    {
        "performance",
        "backfill_transfer",
        "promotion",
        "family_similarity",
        "conversion_fill",
        "transfer_fill",
        "campus_attributes",
        "campus_fill",
    }
)
#: The artifact domain a model field loads from, where it isn't the field's own name.
ARTIFACT_DOMAINS: Final[Mapping[str, str]] = MappingProxyType({"attributes": "job_family"})


@dataclass(frozen=True)
class AwardSettings:
    """Everything SBC needs. Required at `hcf+sbc` scope, refused at `hcf`."""

    new_hire: SizingStrategy
    price_model: StockPriceModel
    expense: ExpenseConfig = field(default_factory=ExpenseConfig)
    #: Sizes promotion grants. Without it, promotions emit grants nobody sizes.
    promotion: SizingStrategy | None = None
    #: Sizes each other award type's grants, keyed by the type: the annual award, a transfer's,
    #: a retention or off-cycle grant. A type without a sizer is emitted and counted unsized.
    sizers: Mapping[str, SizingStrategy] = field(default_factory=dict)
    #: The calendar month, 1 to 12, of the annual award, which grants the workforce. None grants
    #: no annual award.
    annual_month: int | None = None
    #: Awards a planner knows of in advance, each granted to its named employee at its step.
    anticipated: AnticipatedAwards | None = None
    #: The awards as recorded, for the grants they cover: a recipient holding one takes it,
    #: and the rest are sized.
    replayed: RecordedAwards | None = None
    #: The hybrid window's last step for recorded awards: grants after it are all sized. None
    #: routes recorded awards at every step.
    replay_through: int | None = None
    #: Per award type, how many months after a grant's dispatch its recorded award may fall, as
    #: a new hire's follows their start. A recipient takes their earliest recorded award in that
    #: window, in the award's own month. A type without one routes only in its dispatch month.
    replay_lag: Mapping[str, int] = field(default_factory=dict)
    #: Per award type, when its grants are awarded after dispatch. A type without one is
    #: awarded in its dispatch month. A recorded award is awarded in the month it was recorded.
    timing: Mapping[str, AwardTiming] = field(default_factory=dict)
    #: Per award type, who's eligible by hire date. A type without a cutoff grants everyone
    #: its domain dispatches.
    eligibility: Mapping[str, AnnualCutoff] = field(default_factory=dict)
    #: The offline subsidiaries' stream, and expense added by hand: SBC every trajectory
    #: carries beside its own. A grant's recipient who holds an award a replay's offline stream
    #: records takes it from the stream, and the run neither replays nor sizes their grant.
    offline: OfflineStream | None = None
    manual: ManualAdjustments | None = None

    @property
    def window(self) -> HybridWindow:
        return HybridWindow(self.replay_through, self.replay_lag)

    @functools.cached_property
    def streams(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """The offline stream's SBC frame and the manual adjustments', calculated once."""
        offline = None if self.offline is None else self.offline.sbc(self.expense)
        manual = None if self.manual is None else self.manual.sbc()
        return offline, manual

    @property
    def keyed(self) -> tuple[tuple[str, SizingStrategy], ...]:
        """Every sizer beside the new hires', keyed by the award type it's given for."""
        promotion = () if self.promotion is None else (("promotion", self.promotion),)
        return (*promotion, *self.sizers.items())

    def sizer(self, award_type: str) -> SizingStrategy | None:
        """The sizer for an award type, if the run has one."""
        sizers = {self.new_hire.award_type: self.new_hire, **dict(self.keyed)}
        return sizers.get(award_type)


@dataclass(frozen=True)
class SimulationConfig:
    # run: its settings go unprefixed
    run_date: dt.date = field(
        metadata={"group": "run", "doc": "the first day of the run's first month"}
    )
    # objects, passed in code
    schema: SimulationSchema
    models: DomainModels
    # run
    simulation_months: int = setting("run", 12, "how many months the run simulates")
    num_trajectories: int = setting("run", 1, "how many independent trajectories it runs")
    seed: int | None = setting("run", None, "the root seed; none draws fresh entropy")
    forecast_scope: ForecastScope = setting(
        "run", ForecastScope.HEADCOUNT_AND_SBC, "headcount alone, or headcount and SBC"
    )
    run_mode: RunMode | None = setting(
        "run", None, "the run-mode rung; none infers guided_forecast from guidance, else unguided"
    )
    # seeding
    seeding_include_opening_buffer: bool = setting(
        "seeding", True, "seed open requisitions; without them a run opens every requisition"
    )
    seeding_max_openings: int = setting(
        "seeding", 10, "the most openings a seeded requisition keeps; more are clipped"
    )
    seeding_employee_headroom: float = setting(
        "seeding", 0.5, "spare employee rows, as a share of the seed per year of the run"
    )
    seeding_opening_headroom: float = setting(
        "seeding", 0.5, "opening rows beyond the seed, as a share of the employee capacity"
    )
    seeding_min_spare_rows: int = setting(
        "seeding", 500, "the fewest spare rows either buffer gets"
    )
    # performance
    performance_cycle_month: int = setting(
        "performance", 3, "the calendar month a rating cycle takes effect"
    )
    performance_first_rating_min_tenure_months: int = setting(
        "performance", 6, "the tenure an employee needs at a cycle to receive a first rating"
    )
    # attrition
    attrition_executive_min_level: int | None = setting(
        "attrition",
        None,
        "the lowest executive level; executives leave only by known events, never by sampling",
    )
    # backfill
    backfill_source: BackfillSource = setting(
        "backfill", BackfillSource.RATE, "a rate per cost center, or a classifier per departure"
    )
    backfill_anticipated_terminations: bool = setting(
        "backfill",
        False,
        "whether a known termination opens a backfill; by default a planned exit isn't replaced",
    )
    # promotion
    promotion_cycle_months: tuple[int, ...] = setting(
        "promotion", (3, 9), "the calendar months promotions take effect"
    )
    promotion_gate_probability: float = setting(
        "promotion", 1.0, "the chance a cycle month's promotions happen at all"
    )
    promotion_family_change_probability: float = setting(
        "promotion", 0.0, "the chance a sampled promotion also moves to a similar job family"
    )
    # pending starts
    pending_starts_staleness_days: int = setting(
        "pending_starts", 90, "how long past its start date an unstarted offer is still expected"
    )
    # transfers
    transfers_availability_factor: float = setting(
        "transfers", 1.0, "the share of each employee cell available to transfer"
    )
    transfers_level_band: int = setting(
        "transfers", 1, "how many levels from an opening's own a transfer may come from"
    )
    transfers_shortfall_policy: ShortfallPolicy = setting(
        "transfers",
        ShortfallPolicy.KEEP_OPEN,
        "keep an unmatched transfer opening open, or spill it to another fill channel",
    )
    # requisitions
    requisitions_supply_gain: float = setting(
        "requisitions",
        1.0,
        "the share of the gap to the plan target that target-seeking growth closes each month",
    )
    # alignment
    alignment_method: AlignmentMethod = setting(
        "alignment",
        AlignmentMethod.SUPPLY,
        "the valve the plan drives: none, supply, demand, hybrid, or rate_calibration",
    )
    # replay
    replay_boundary_fill: bool = setting(
        "replay", True, "inject boundary fills for hires replay can't trace to a requisition"
    )
    # results
    results_snapshot_steps: tuple[int, ...] = setting(
        "results", (), "the steps whose buffers a run keeps as snapshots"
    )
    results_streaming: bool = setting(
        "results", False, "stream each trajectory's results out instead of holding them"
    )
    # objects, passed in code
    guidance: GuidanceForecast | None = None
    awards: AwardSettings | None = None
    #: Eligibility rules, declared in raw labels.
    eligibility: tuple[EligibilityRule, ...] = ()
    #: Registration and step gates on the domains.
    gates: tuple[Gate, ...] = ()
    #: `Cap(c)`, by cost-center label.
    capacity_caps: Mapping[str, int] = field(default_factory=dict)
    #: The plan whose cost-center targets bound committed strength, `H*(c, t)`.
    plan: AlignmentTarget | None = None
    #: A planner's reshaping of the fill channels' hazards.
    fill_mix: FillMix | None = None
    #: Campus requisitions by month and cost center, and the months campus hires start.
    campus: CampusSchedule | None = None
    aligner: GroupAligner | None = field(init=False, default=None)
    #: The rules, resolved against the schema.
    resolved_eligibility: Eligibility = field(init=False, default_factory=Eligibility)
    #: `Cap_eff(c, t)`, from the caps and the plan.
    ceiling: Ceiling = field(init=False, default_factory=lambda: Ceiling(np.zeros(0)))

    def __post_init__(self) -> None:
        problems: list[str] = []
        typed = self._normalize(problems)
        problems += self._check_settings(typed)
        if self.guidance is not None and {"simulation_months", "run_date"} <= typed:
            problems += self._check_guidance(self.guidance)
        object.__setattr__(self, "eligibility", tuple(self.eligibility))
        object.__setattr__(self, "gates", tuple(self.gates))
        problems += [f"gates: {p}" for p in check_gates(self.gates)]
        object.__setattr__(self, "capacity_caps", MappingProxyType(dict(self.capacity_caps)))
        if self.campus is not None:
            problems += [f"campus: {p}" for p in self.campus.problems(self.schema)]
            if self.models.campus_attributes is None:
                problems.append("campus: a campus schedule needs a campus attribute generator")
        try:
            method = (
                self.alignment_method if "alignment_method" in typed else AlignmentMethod.SUPPLY
            )
            supply = supplied(method, self._opened_coverage())
            ceiling = Ceiling.build(self.schema, self.capacity_caps, self.plan, supply)
            object.__setattr__(self, "ceiling", ceiling)
        except CeilingError as error:
            problems += [f"ceiling: {p}" for p in error.problems]
        try:
            rules = (*self.eligibility, *self._executive_rule())
            object.__setattr__(self, "resolved_eligibility", resolve(rules, self.schema))
        except EligibilityError as error:
            problems += [f"eligibility: {p}" for p in error.problems]
        if problems:
            raise ConfigError(problems)

    def _normalize(self, problems: list[str]) -> set[str]:
        """Check every setting holds its type, reading enum values and lists as their types.
        Returns the settings that do: a check runs only when every setting it reads does."""
        hints = _hints()
        typed = set()
        for f in dataclasses.fields(self):
            if "group" not in f.metadata:
                continue
            try:
                object.__setattr__(self, f.name, _conform(getattr(self, f.name), hints[f.name]))
            except (TypeError, ValueError) as error:
                problems.append(f"{f.name}: {error}")
            else:
                typed.add(f.name)
        if "run_mode" in typed and self.run_mode is None:
            mode = RunMode.UNGUIDED if self.guidance is None else RunMode.GUIDED_FORECAST
            object.__setattr__(self, "run_mode", mode)
        return typed

    def _check_settings(self, typed: set[str]) -> list[str]:
        mode = cast(RunMode, self.run_mode)
        guided = mode in {RunMode.GUIDED_ACTUALS, RunMode.GUIDED_FORECAST}
        sbc = self.forecast_scope is ForecastScope.HEADCOUNT_AND_SBC
        checks: list[tuple[tuple[str, ...], Callable[[], list[str]]]] = [
            (
                ("run_date",),
                lambda: _when(
                    self.run_date.day != 1,
                    f"run_date must be the first of a month, got {self.run_date}",
                ),
            ),
            (
                ("simulation_months",),
                lambda: _when(self.simulation_months < 1, "simulation_months must be at least 1"),
            ),
            (
                ("num_trajectories",),
                lambda: _when(self.num_trajectories < 1, "num_trajectories must be at least 1"),
            ),
            (
                ("seed",),
                lambda: _when(self.seed is not None and self.seed < 0, "seed must be non-negative"),
            ),
            (
                ("forecast_scope",),
                lambda: (
                    _when(
                        sbc and self.awards is None,
                        "forecast_scope 'hcf+sbc' needs awards settings",
                    )
                    + _when(
                        not sbc and self.awards is not None,
                        "forecast_scope 'hcf' computes no SBC, so it refuses awards settings",
                    )
                ),
            ),
            (
                ("run_mode",),
                lambda: (
                    _when(guided and self.guidance is None, f"run_mode {mode} needs guidance")
                    + _when(
                        not guided and self.guidance is not None,
                        f"run_mode {mode} reads no guidance; drop it",
                    )
                ),
            ),
            (
                ("backfill_source",),
                lambda: self._check_backfill_models(),
            ),
            (
                ("seeding_max_openings",),
                lambda: _when(
                    self.seeding_max_openings < 1, "seeding_max_openings must be at least 1"
                ),
            ),
            (
                ("seeding_employee_headroom", "seeding_opening_headroom"),
                lambda: _when(
                    min(self.seeding_employee_headroom, self.seeding_opening_headroom) < 0,
                    "seeding headrooms must be non-negative",
                ),
            ),
            (
                ("seeding_min_spare_rows",),
                lambda: _when(
                    self.seeding_min_spare_rows < 0, "seeding_min_spare_rows must be non-negative"
                ),
            ),
            (
                ("performance_cycle_month",),
                lambda: _when(
                    not 1 <= self.performance_cycle_month <= 12,
                    "performance_cycle_month must be a calendar month, 1 to 12",
                ),
            ),
            (
                ("performance_first_rating_min_tenure_months",),
                lambda: _when(
                    self.performance_first_rating_min_tenure_months < 0,
                    "performance_first_rating_min_tenure_months must be non-negative",
                ),
            ),
            (
                ("promotion_cycle_months",),
                lambda: _months("promotion_cycle_months", self.promotion_cycle_months, 1, 12),
            ),
            (
                ("promotion_gate_probability",),
                lambda: _when(
                    not 0.0 <= self.promotion_gate_probability <= 1.0,
                    "promotion_gate_probability must lie in [0, 1]",
                ),
            ),
            (
                ("promotion_family_change_probability",),
                lambda: self._check_family_change(),
            ),
            (
                ("forecast_scope", "simulation_months"),
                lambda: self._check_sizers(),
            ),
            (
                ("forecast_scope", "simulation_months"),
                lambda: (
                    []
                    if self.awards is None
                    else covers(self.awards.price_model, self.simulation_months)
                ),
            ),
            (
                ("pending_starts_staleness_days",),
                lambda: _when(
                    self.pending_starts_staleness_days < 0,
                    "pending_starts_staleness_days must be non-negative",
                ),
            ),
            (
                ("transfers_availability_factor",),
                lambda: _when(
                    not 0.0 <= self.transfers_availability_factor <= 1.0,
                    "transfers_availability_factor must lie in [0, 1]",
                ),
            ),
            (
                ("transfers_level_band",),
                lambda: _when(
                    self.transfers_level_band < 0, "transfers_level_band must be non-negative"
                ),
            ),
            (
                ("requisitions_supply_gain",),
                lambda: _when(
                    not 0.0 <= self.requisitions_supply_gain <= 1.0,
                    "requisitions_supply_gain must lie in [0, 1]",
                ),
            ),
            (
                ("results_snapshot_steps", "simulation_months"),
                lambda: _months(
                    "results_snapshot_steps", self.results_snapshot_steps, 0, self.simulation_months
                ),
            ),
        ]
        problems = []
        for needs, check in checks:
            if set(needs) <= typed:
                problems += check()
        return problems

    def _check_family_change(self) -> list[str]:
        chance = self.promotion_family_change_probability
        if not 0.0 <= chance <= 1.0:
            return ["promotion_family_change_probability must lie in [0, 1]"]
        if chance > 0 and self.models.family_similarity is None:
            return ["a promotion that changes family needs a family_similarity model"]
        return []

    def _check_sizers(self) -> list[str]:
        """Each sizer grants the award type it's given for, one the schema knows, and no type
        has two; the annual award falls in a calendar month; the replay lags are months from 0,
        of types the schema knows; and the anticipated awards are of types the schema knows,
        inside the run."""
        if self.awards is None:
            return []
        known = self.schema.vocabulary(AWARD_TYPE) if AWARD_TYPE in self.schema.vocabularies else ()
        problems = []
        if self.awards.new_hire.award_type in self.awards.sizers:
            problems.append(f"{self.awards.new_hire.award_type} grants have two sizers")
        if self.awards.promotion is not None and "promotion" in self.awards.sizers:
            problems.append("promotion grants have two sizers")
        for award_type, sizer in self.awards.keyed:
            if sizer.award_type != award_type:
                problems.append(
                    f"the {award_type} sizer grants {sizer.award_type!r}, not {award_type!r}"
                )
            elif award_type not in known:
                problems.append(
                    f"sizing {award_type} grants needs {award_type!r} among the award types"
                )
        month = self.awards.annual_month
        if month is not None and not 1 <= month <= 12:
            problems.append(f"the annual award's month is a calendar month, 1 to 12, not {month}")
        lag = self.awards.replay_lag
        unknown = sorted(t for t in lag if t not in known)
        if unknown:
            problems.append(f"replay lags for award types the schema doesn't know: {unknown}")
        negative = sorted(t for t, months in lag.items() if months < 0)
        if negative:
            problems.append(f"a replay lag is a number of months from 0, which {negative} isn't")
        anticipated = self.awards.anticipated
        if anticipated is not None:
            unknown = sorted(t for t in anticipated.award_types if t not in known)
            if unknown:
                problems.append(f"anticipated awards of types the schema doesn't know: {unknown}")
            if anticipated.last_step >= self.simulation_months:
                problems.append(
                    f"an anticipated award at step {anticipated.last_step} lies past the run's "
                    f"last, {self.simulation_months - 1}"
                )
        return problems

    def _check_backfill_models(self) -> list[str]:
        """Each backfill model must serve the configured source: a rate per cost center, or a
        classifier per departure."""
        problems = []
        wanted = GroupRateModel if self.backfill_source is BackfillSource.RATE else EntityModel
        for name in ("backfill", "backfill_transfer"):
            model = getattr(self.models, name)
            if model is not None and not isinstance(model, wanted):
                problems.append(
                    f"backfill_source {self.backfill_source} needs a {wanted.__name__} as "
                    f"the {name} model, not a {type(model).__name__}"
                )
        return problems

    def _executive_rule(self) -> list[EligibilityRule]:
        """Keep executives out of sampled attrition: every level from the lowest executive
        one up, as the schema's level vocabulary has them."""
        floor = self.attrition_executive_min_level
        if not isinstance(floor, int) or JOB_LEVEL not in self.schema.vocabularies:
            return []
        levels = [v for v in self.schema.vocabulary(JOB_LEVEL).labels if float(v) >= floor]
        if not levels:
            return []
        reason = "executives leave only by known events"
        return [EligibilityRule.of("attrition", job_level=levels, reason=reason)]

    def _check_guidance(self, guidance: GuidanceForecast) -> list[str]:
        problems = []
        if guidance.horizon < self.simulation_months:
            problems.append(
                f"guidance covers {guidance.horizon} months; the run needs {self.simulation_months}"
            )
        expected_origin = (self.run_date - dt.timedelta(days=1)).isoformat()
        if guidance.snapshot_date != expected_origin:
            problems.append(
                f"guidance was fitted through {guidance.snapshot_date}; a run from "
                f"{self.run_date} needs a snapshot of {expected_origin}"
            )
        if (
            GuidanceCategory.REQUISITIONS_OPENED in guidance.categories
            and guidance.grouping_column != COST_CENTER
        ):
            problems.append("requisition generation reads guidance only at cost-center grain")
        if guidance.aligned_to is not None and self.plan is not None:
            plan = self.plan.digest()
            if guidance.aligned_to != plan:
                problems.append(
                    f"guidance was aligned to the plan {guidance.aligned_to}, but the run's plan "
                    f"is {plan}: realign the guidance, or run against the plan it was aligned to"
                )
        try:
            object.__setattr__(self, "aligner", guidance.aligner(self.schema))
        except SchemaError as error:
            problems.append(f"guidance doesn't fit the schema: {error}")
        return problems

    def _opened_coverage(self) -> NDArray[np.bool_]:
        """Per cost-center code, whether guidance covers its requisitions opened."""
        n = len(self.schema.vocabulary(COST_CENTER))
        guidance, aligner = self.guidance, self.aligner
        opened = GuidanceCategory.REQUISITIONS_OPENED
        if guidance is None or aligner is None or not guidance.covers(opened):
            return np.zeros(n, dtype=bool)
        groups = aligner.groups_of_codes(np.arange(n, dtype=np.int64))
        cover = guidance.group_coverage(opened)
        covered: NDArray[np.bool_] = (groups >= 0) & cover[np.maximum(groups, 0)]
        return covered

    @property
    def computes_sbc(self) -> bool:
        return self.forecast_scope is ForecastScope.HEADCOUNT_AND_SBC
