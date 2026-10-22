"""Tests for the coherence validators: a world passes them all, and each catches its break."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.synth.cli import main
from hcsim.synth.config import QuirkTruth, preset
from hcsim.synth.validate import CHECKS, Result, validate
from hcsim.synth.world import World, generate_world, write_world

EVERY_QUIRK = QuirkTruth(
    backdated_share=0.2,
    correction_share=0.2,
    no_show_pair_share=1.0,
    pending_hire_rows=True,
    aggregate_only_share=0.2,
    snapshot_gap_share=0.2,
    state_seam_month="2023-07",
    cost_center_aliases=True,
)


@pytest.fixture(scope="module")
def world() -> World:
    return generate_world(dataclasses.replace(preset("tiny", seed=5), quirks=EVERY_QUIRK))


def checked(world: World, path: Path) -> dict[str, Result]:
    write_world(world, path)
    with duckdb.connect(str(path), read_only=True) as con:
        return {r.name: r for r in validate(con)}


def test_a_world_passes_every_check_with_every_quirk_on(world: World, tmp_path: Path) -> None:
    results = checked(world, tmp_path / "w.duckdb")
    assert list(results) == [c.name for c in CHECKS]
    assert all(r.passed and r.examples == [] for r in results.values())


def _first(frame: pd.DataFrame, mask: pd.Series) -> int:
    return int(frame.index[mask][0])


def _drop_flow(t: dict[str, pd.DataFrame]) -> None:
    txn = t["workforce_txn"]
    t["workforce_txn"] = txn.drop(index=_first(txn, txn["trans_type"] == "TERM"))


def _double_stock(t: dict[str, pd.DataFrame]) -> None:
    txn = t["workforce_txn"]
    row = txn.loc[[_first(txn, txn["trans_type"] == "ACTIVE")]]
    t["workforce_txn"] = pd.concat([txn, row], ignore_index=True)


def _unknown_code(t: dict[str, pd.DataFrame]) -> None:
    agg = t["workforce_agg"].astype({"trans_type": object})
    agg.loc[0, "trans_type"] = "NOT_A_CODE"
    t["workforce_agg"] = agg


def _bump_aggregate(t: dict[str, pd.DataFrame]) -> None:
    agg = t["workforce_agg"].copy()
    agg["headcount"] = agg["headcount"] + (agg.index == _first(agg, agg["trans_type"] == "HIRE"))
    t["workforce_agg"] = agg


def _bump_openings(t: dict[str, pd.DataFrame]) -> None:
    reqs = t["req_interval"].copy()
    reqs["openings"] = reqs["openings"] + (reqs.index == _first(reqs, reqs["interval_number"] == 2))
    t["req_interval"] = reqs


def _drop_transfer(t: dict[str, pd.DataFrame]) -> None:
    txn = t["workforce_txn"]
    t["workforce_txn"] = txn.drop(index=_first(txn, txn["trans_type"] == "XFER_IN"))


def _move_offer(t: dict[str, pd.DataFrame]) -> None:
    pending = t["pending_start"].copy()
    pending["offer_date"] = pending["offer_date"].mask(
        pending.index == 0, pending["offer_date"] - np.timedelta64(1, "D")
    )
    t["pending_start"] = pending


def _bump_tranche(t: dict[str, pd.DataFrame]) -> None:
    vests = t["equity_vest"].copy()
    vests["shares"] = vests["shares"] + (vests.index == 0)
    t["equity_vest"] = vests


def _misstate_status(t: dict[str, pd.DataFrame]) -> None:
    vests = t["equity_vest"].copy()
    vests.loc[_first(vests, vests["status"] == "vested"), "status"] = "unvested"
    t["equity_vest"] = vests


def _bump_recognized(t: dict[str, pd.DataFrame]) -> None:
    ledger = t["equity_expense"].copy()
    ledger["recognized"] = ledger["recognized"] + np.where(ledger.index == 0, 1.0, 0.0)
    t["equity_expense"] = ledger


def _bump_net(t: dict[str, pd.DataFrame]) -> None:
    """Rows still add up, but a settled award's total no longer matches its vested value."""
    ledger = t["equity_expense"].copy()
    for column in ("net", "recognized"):
        ledger[column] = ledger[column] + np.where(ledger.index == 0, 100.0, 0.0)
    t["equity_expense"] = ledger


BREAKS: dict[str, Callable[[dict[str, pd.DataFrame]], None]] = {
    "stock_flow": _drop_flow,
    "one_stock_row": _double_stock,
    "known_codes": _unknown_code,
    "aggregate_ties": _bump_aggregate,
    "requisition_chain": _bump_openings,
    "fills_tie_to_moves": _drop_transfer,
    "pending_starts": _move_offer,
    "vests_sum_to_awards": _bump_tranche,
    "vest_status": _misstate_status,
    "ledger_rows": _bump_recognized,
    "ledger_totals": _bump_net,
}


def test_every_check_has_a_break() -> None:
    assert set(BREAKS) == {c.name for c in CHECKS}


@pytest.mark.parametrize("name", list(BREAKS))
def test_each_check_catches_its_break(world: World, tmp_path: Path, name: str) -> None:
    tables = dict(world.tables)
    BREAKS[name](tables)
    broken = dataclasses.replace(world, tables=tables)
    result = checked(broken, tmp_path / "b.duckdb")[name]
    assert not result.passed
    assert 0 < len(result.examples) <= 5


def test_the_cli_validates_a_world(
    world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "w.duckdb"
    write_world(world, path)
    assert main(["validate", "--db", str(path)]) == 0
    assert f"{len(CHECKS)} of {len(CHECKS)} checks passed" in capsys.readouterr().out
    tables = dict(world.tables)
    _drop_transfer(tables)
    write_world(dataclasses.replace(world, tables=tables), path)
    assert main(["validate", "--db", str(path)]) == 1
    out = capsys.readouterr().out
    assert "FAIL fills_tie_to_moves" in out
    assert main(["validate", "--db", str(tmp_path / "missing.duckdb")]) == 2


def test_the_small_world_passes_every_check(small_world: World, tmp_path: Path) -> None:
    results = checked(small_world, tmp_path / "small.duckdb")
    assert [name for name, r in results.items() if not r.passed] == []
