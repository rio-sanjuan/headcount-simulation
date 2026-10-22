"""Known-answer cases for the rate-aware alignment QP, each solved by hand before its first run.

One group, a last actual of 100, hires (+) and attrition (-). With equal weights and no rate
scaling, the QP minimizes `Δh² + Δa²`; the balance makes the anchor's change `h - a`.
"""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.guidance.alignment import AlignmentProblem, qp

TOLERANCE = 1e-4
SIGNS = {"hires": 1, "attrition": -1}
NAN = np.nan


def solve(
    hires: list[float],
    attrition: list[float],
    target: list[float],
    band: list[float] | None = None,
    anchor: list[float] | None = None,
    **kwargs: object,
) -> object:
    flows = {"hires": np.array(hires), "attrition": np.array(attrition)}
    months = len(hires)
    problem = AlignmentProblem(
        np.array(anchor or [100.0] * months),
        100.0,
        flows,
        SIGNS,
        np.array(target),
        np.zeros(months) if band is None else np.array(band),
        **{"rate_aware": False, **kwargs},  # type: ignore[arg-type]
    )
    return qp(problem)


def flows(result: object) -> tuple[float, ...]:
    values = result.flows  # type: ignore[attr-defined]
    return tuple(float(x) for name in ("hires", "attrition") for x in values[name])


def test_equal_weights_split_the_gap_evenly() -> None:
    # +10 wanted from h - a = 0: Δh = 5, Δa = -5.
    result = solve([5.0], [5.0], [110.0])
    assert flows(result) == pytest.approx((10.0, 0.0), abs=TOLERANCE)
    assert result.anchor.tolist() == pytest.approx([110.0], abs=TOLERANCE)  # type: ignore[attr-defined]
    assert result.shortfall.tolist() == pytest.approx([0.0], abs=TOLERANCE)  # type: ignore[attr-defined]


def test_weights_say_which_flow_gives() -> None:
    # min Δh² + 3Δa² with Δh - Δa = 10: Δh = 7.5, Δa = -2.5.
    result = solve([5.0], [5.0], [110.0], weights={"attrition": 3.0})
    assert flows(result) == pytest.approx((12.5, 2.5), abs=TOLERANCE)


def test_a_fixed_flow_leaves_the_gap_to_the_rest() -> None:
    result = solve([5.0], [5.0], [110.0], fixed=frozenset({"attrition"}))
    assert flows(result) == pytest.approx((15.0, 5.0), abs=TOLERANCE)


def test_a_band_asks_only_for_its_edge() -> None:
    # 110 within 4 needs only +6: Δh = 3, Δa = -3.
    result = solve([5.0], [5.0], [110.0], band=[4.0])
    assert flows(result) == pytest.approx((8.0, 2.0), abs=TOLERANCE)
    assert result.anchor.tolist() == pytest.approx([106.0], abs=TOLERANCE)  # type: ignore[attr-defined]


def test_a_flow_stops_at_zero_and_the_others_carry_the_rest() -> None:
    # +12 wanted, +4 forecast: an even split of the other 8 would take attrition to -3, so it
    # stops at 0, a change of -1, and hires gives the other 7.
    result = solve([5.0], [1.0], [112.0])
    assert flows(result) == pytest.approx((12.0, 0.0), abs=TOLERANCE)


def test_an_unreachable_target_is_a_shortfall() -> None:
    # 80 wanted, attrition fixed at 5: hires can fall only to 0, reaching 95, 15 short.
    result = solve([5.0], [5.0], [80.0], fixed=frozenset({"attrition"}))
    assert flows(result) == pytest.approx((0.0, 5.0), abs=TOLERANCE)
    assert result.shortfall.tolist() == pytest.approx([15.0], abs=TOLERANCE)  # type: ignore[attr-defined]


def test_rate_awareness_moves_the_months_with_more_exposure() -> None:
    # +10 by month 2, attrition fixed: min (x1/100)² + (x2/200)² with x1 + x2 = 10 puts the
    # change in proportion to the exposure squared, 1 : 4, so x1 = 2 and x2 = 8.
    result = solve(
        [5.0, 5.0],
        [5.0, 5.0],
        [NAN, 110.0],
        anchor=[100.0, 200.0],
        fixed=frozenset({"attrition"}),
        rate_aware=True,
    )
    assert flows(result) == pytest.approx((7.0, 13.0, 5.0, 5.0), abs=TOLERANCE)
    blind = solve([5.0, 5.0], [5.0, 5.0], [NAN, 110.0], fixed=frozenset({"attrition"}))
    assert flows(blind) == pytest.approx((10.0, 10.0, 5.0, 5.0), abs=TOLERANCE)


def test_declared_movements_enter_the_balance() -> None:
    # A movement of +4 brings 4 of the 10 wanted: the flows split the other 6, 3 and 3.
    result = solve([5.0], [5.0], [110.0], movements=np.array([4.0]))
    assert flows(result) == pytest.approx((8.0, 2.0), abs=TOLERANCE)
    assert result.anchor.tolist() == pytest.approx([110.0], abs=TOLERANCE)  # type: ignore[attr-defined]


def test_unaligned_months_keep_their_flows() -> None:
    result = solve([5.0, 5.0], [5.0, 5.0], [NAN, NAN])
    assert flows(result) == pytest.approx((5.0, 5.0, 5.0, 5.0), abs=TOLERANCE)
    assert result.method == "qp"  # type: ignore[attr-defined]
