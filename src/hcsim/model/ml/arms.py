"""The arm matrix: every domain's candidate models, how they're ordered, and how one is chosen.

An `Arm` is one model for one domain. Each domain's arms share a family (classifier,
generator, or count) and are ranked by simplicity, from the null arm, the naive benchmark every
other arm is measured against, to the richest. The authored matrix is validated as a whole,
every problem reported at once:

- arm names are unique;
- each domain has exactly one null arm, and it's the simplest;
- simplicity ranks within a domain are distinct, so the order is total;
- a domain's arms share one known family, and each algorithm belongs to it.

**Selection.** Starting from the null arm, the champion moves to the next richer arm only if
that arm clears the decision margin against it. Both arms are scored on the same held-out rows
by the family's primary metric, broken into per-row contributions: log loss for classifiers,
log-likelihood for generators, and each cell's share of WAPE for counts. A richer arm clears
the margin when its mean per-row gain is more than `z` standard errors of the paired gains. The
margin is statistical rather than a fixed amount per family, so it scales with how much a
domain's data can tell arms apart. `z` defaults to 2.

**Lift tables** show how a classifier's scores separate events: rows in bands of equal count,
from the highest scores down, with each band's event rate, its lift over the overall rate, and
the share of all events captured so far.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from hcsim.model.ml.metrics import FAMILY_METRICS, Direction, primary

Floats = NDArray[np.float64]
_EPS: Final = 1e-15
DEFAULT_Z: Final = 2.0

#: The algorithms each family's arms may use.
ALGORITHMS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "classifier": frozenset(
            {"constant", "level_only", "naive_rate", "logistic", "boosted", "backfill_rate"}
        ),
        "generator": frozenset({"empirical_attributes", "conditional_categorical", "markov"}),
        "count": frozenset({"poisson_growth", "negative_binomial_growth"}),
    }
)


class ArmError(ValueError):
    """An arm matrix, or a selection, that doesn't hold."""


@dataclass(frozen=True)
class Arm:
    name: str
    domain: str
    family: str
    algorithm: str
    simplicity: int
    null: bool = False
    features: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArmMatrix:
    arms: tuple[Arm, ...]

    def __post_init__(self) -> None:
        problems = []
        names = [a.name for a in self.arms]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            problems.append(f"arm names repeat: {repeated}")
        for domain in self.domains:
            arms = [a for a in self.arms if a.domain == domain]
            families = {a.family for a in arms}
            if len(families) > 1:
                problems.append(f"{domain} mixes families {sorted(families)}")
            for arm in arms:
                if arm.family not in ALGORITHMS:
                    known = ", ".join(ALGORITHMS)
                    problems.append(f"{arm.name} has family {arm.family!r}; known: {known}")
                elif arm.algorithm not in ALGORITHMS[arm.family]:
                    problems.append(f"{arm.name}: {arm.algorithm!r} isn't a {arm.family} algorithm")
            nulls = [a for a in arms if a.null]
            if len(nulls) != 1:
                problems.append(f"{domain} needs exactly one null arm, not {len(nulls)}")
            ranks = [a.simplicity for a in arms]
            if len(set(ranks)) != len(ranks):
                problems.append(f"{domain} has arms of equal simplicity")
            if len(nulls) == 1 and nulls[0].simplicity != min(ranks):
                problems.append(f"{domain}'s null arm isn't its simplest")
        if problems:
            raise ArmError("bad arm matrix: " + "; ".join(problems))

    @property
    def domains(self) -> list[str]:
        return sorted({a.domain for a in self.arms})

    def domain(self, name: str) -> list[Arm]:
        """A domain's arms, simplest first."""
        arms = sorted((a for a in self.arms if a.domain == name), key=lambda a: a.simplicity)
        if not arms:
            raise ArmError(f"no arms for domain {name!r}; known: {', '.join(self.domains)}")
        return arms

    def null(self, domain: str) -> Arm:
        return self.domain(domain)[0]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Sequence[Mapping[str, Any]]]) -> ArmMatrix:
        """`{domain: [{name, family, algorithm, null?, features?}, ...]}`, simplest first."""
        arms = []
        for domain, entries in data.items():
            for rank, entry in enumerate(entries):
                arms.append(
                    Arm(
                        str(entry["name"]),
                        domain,
                        str(entry["family"]),
                        str(entry["algorithm"]),
                        rank,
                        bool(entry.get("null", False)),
                        tuple(entry.get("features", ())),
                    )
                )
        return cls(tuple(arms))


def _classifier_arms(domain: str, richer: Iterable[str]) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = [
        {
            "name": f"{domain}.constant",
            "family": "classifier",
            "algorithm": "constant",
            "null": True,
        }
    ]
    arms += [{"name": f"{domain}.{a}", "family": "classifier", "algorithm": a} for a in richer]
    return arms


#: The authored arm matrix for the engine's sampled domains.
DEFAULT_MATRIX: Final = ArmMatrix.from_mapping(
    {
        "attrition": _classifier_arms("attrition", ["level_only", "logistic", "boosted"]),
        "promotion": _classifier_arms("promotion", ["level_only", "logistic", "boosted"]),
        "external_fill": _classifier_arms("external_fill", ["level_only", "boosted"]),
        "cancellation": _classifier_arms("cancellation", ["level_only", "boosted"]),
        "backfill": _classifier_arms("backfill", ["backfill_rate"]),
        "growth": [
            {
                "name": "growth.poisson",
                "family": "count",
                "algorithm": "poisson_growth",
                "null": True,
            },
            {
                "name": "growth.negative_binomial",
                "family": "count",
                "algorithm": "negative_binomial_growth",
            },
        ],
        "job_family": [
            {
                "name": "job_family.empirical",
                "family": "generator",
                "algorithm": "empirical_attributes",
                "null": True,
            },
            {
                "name": "job_family.conditional",
                "family": "generator",
                "algorithm": "conditional_categorical",
            },
        ],
        "performance": [
            {
                "name": "performance.overall",
                "family": "generator",
                "algorithm": "conditional_categorical",
                "null": True,
            },
            {"name": "performance.markov", "family": "generator", "algorithm": "markov"},
        ],
    }
)


def contributions(family: str, **inputs: ArrayLike) -> Floats:
    """Each held-out row's part of the family's primary metric.

    - classifier: `labels` and `probabilities`, giving each row's log loss;
    - generator: `probabilities` (rows, categories) and `outcomes`, giving each row's
      log-likelihood;
    - count: `actual` and `forecast`, giving each cell's share of WAPE.
    """
    if family == "classifier":
        y = np.asarray(inputs["labels"], dtype=np.float64)
        p = np.clip(np.asarray(inputs["probabilities"], dtype=np.float64), _EPS, 1 - _EPS)
        result: Floats = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return result
    if family == "generator":
        p = np.asarray(inputs["probabilities"], dtype=np.float64)
        k = np.asarray(inputs["outcomes"], dtype=np.int64)
        logged: Floats = np.log(np.clip(p[np.arange(len(p)), k], _EPS, None))
        return logged
    if family == "count":
        a = np.asarray(inputs["actual"], dtype=np.float64)
        f = np.asarray(inputs["forecast"], dtype=np.float64)
        scale = np.abs(a).sum()
        if scale == 0:
            raise ArmError("WAPE contributions need some non-zero actuals")
        shares: Floats = np.abs(a - f).ravel() / scale
        return shares
    raise ArmError(f"no model family {family!r}; known: {', '.join(FAMILY_METRICS)}")


@dataclass(frozen=True)
class Decision:
    """One step of a selection: whether `arm` cleared the margin against `against`."""

    arm: str
    against: str
    mean_gain: float
    standard_error: float
    cleared: bool


@dataclass(frozen=True)
class Selection:
    champion: str
    decisions: tuple[Decision, ...]


def select(
    matrix: ArmMatrix,
    domain: str,
    scored: Mapping[str, ArrayLike],
    *,
    z: float = DEFAULT_Z,
) -> Selection:
    """Choose a domain's champion from per-row contributions, scored on the same rows."""
    arms = matrix.domain(domain)
    missing = [a.name for a in arms if a.name not in scored]
    if missing:
        raise ArmError(f"{domain} arms without scores: {missing}")
    rows = {a.name: np.asarray(scored[a.name], dtype=np.float64) for a in arms}
    shapes = {v.shape for v in rows.values()}
    if len(shapes) != 1 or next(iter(shapes))[0] < 2:
        raise ArmError("every arm needs contributions for the same rows, at least two of them")
    sign = 1.0 if primary(arms[0].family).direction is Direction.HIGHER else -1.0
    champion = arms[0].name
    decisions = []
    for arm in arms[1:]:
        gain = sign * (rows[arm.name] - rows[champion])
        se = float(gain.std(ddof=1) / np.sqrt(len(gain)))
        mean = float(gain.mean())
        cleared = mean > 0 and mean > z * se
        decisions.append(Decision(arm.name, champion, mean, se, cleared))
        if cleared:
            champion = arm.name
    return Selection(champion, tuple(decisions))


def lift_table(labels: ArrayLike, probabilities: ArrayLike, bands: int = 10) -> pd.DataFrame:
    """Rows in bands of equal count, highest scores first, with each band's event rate, lift
    over the overall rate, and the cumulative share of events captured."""
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or y.size == 0:
        raise ArmError("labels and probabilities need one value each per row")
    if bands < 1:
        raise ArmError("a lift table needs at least one band")
    overall = y.mean()
    order = np.argsort(-p, kind="stable")
    records = []
    captured = 0.0
    for band, chunk in enumerate(np.array_split(order, min(bands, len(order))), start=1):
        events = float(y[chunk].sum())
        captured += events
        rate = events / len(chunk)
        records.append(
            {
                "band": band,
                "rows": len(chunk),
                "events": events,
                "event_rate": rate,
                "mean_probability": float(p[chunk].mean()),
                "lift": rate / overall if overall > 0 else np.nan,
                "captured": captured / y.sum() if y.sum() > 0 else np.nan,
            }
        )
    return pd.DataFrame(records)
