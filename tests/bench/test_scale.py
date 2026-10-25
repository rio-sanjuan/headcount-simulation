"""Tests for the scale harness, the capacity model, and benchmark baselines."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from hcsim.bench import cli, scale
from hcsim.bench.baseline import (
    BASELINE_PATH,
    MEASURES,
    Baseline,
    BaselineError,
    Flag,
    compare,
    machine,
)
from hcsim.bench.capacity import CapacityError, CapacityModel, Fit, nonnegative_fit
from hcsim.bench.scale import (
    CASES,
    GUIDANCE,
    LARGE,
    SIMULATED,
    SUITE,
    Case,
    Measurement,
    ensure_world,
    measure,
)
from hcsim.bench.world import BenchError

ROOT = Path(__file__).resolve().parents[2]
TINY = Case("tiny-sbc-2", "tiny", "hcf+sbc", 2, months=3)


def synthetic(
    scope: str, employees: int, trajectories: int, streaming: bool = False
) -> Measurement:
    """A measurement made by an exactly linear machine."""
    held = 1 if streaming else trajectories
    work = employees * 12 * trajectories
    return Measurement(
        case=f"{scope}-{employees}-{trajectories}-{streaming}",
        preset="x",
        scope=scope,
        trajectories=trajectories,
        months=12,
        streaming=streaming,
        employees=employees,
        openings=employees // 20,
        tranches=0,
        prepare_seconds=0.5 + 2e-5 * employees,
        simulate_seconds=1.0 + 1e-6 * work,
        peak_mib=150 + 0.01 * employees + 0.002 * employees * held,
    )


MEASURED = [
    synthetic("hcf", 1_000, 2),
    synthetic("hcf", 10_000, 2),
    synthetic("hcf", 10_000, 8),
    synthetic("hcf", 50_000, 4, streaming=True),
]


@pytest.fixture(scope="module")
def tiny_world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return ensure_world(TINY, tmp_path_factory.mktemp("worlds"))


# --- the harness ------------------------------------------------------------------------------


def test_the_suites_name_their_cases_once() -> None:
    assert len(CASES) == len(SUITE) + len(LARGE)
    assert {c.preset for c in SUITE} == {"tiny", "small", "medium"}
    assert {c.preset for c in LARGE} == {"large"}
    for scope in SIMULATED:
        assert len([c for c in SUITE if c.scope == scope]) >= 3  # enough to fit each scope
    assert any(c.streaming for c in SUITE)
    assert {c.preset for c in SUITE if c.scope == GUIDANCE} == {"small", "medium"}


def test_a_case_measures_its_size_and_costs(tiny_world: Path) -> None:
    held = measure(TINY, tiny_world)
    streamed = measure(dataclasses.replace(TINY, streaming=True), tiny_world)
    assert held.employees == streamed.employees > 0
    assert held.tranches == streamed.tranches > 0  # streaming counts every trajectory's
    assert (held.held, streamed.held) == (2, 1)
    assert held.employee_months == held.employees * 3 * 2
    assert min(held.prepare_seconds, held.simulate_seconds) >= 0
    assert held.peak_mib > 0


def test_a_guidance_case_times_the_fit_and_the_alignment(tiny_world: Path) -> None:
    m = measure(Case("tiny-guidance", "tiny", GUIDANCE, 1, months=6), tiny_world)
    assert m.employees > 0
    assert (m.openings, m.tranches) == (0, 0)
    assert m.prepare_seconds > 0
    assert m.simulate_seconds >= 0
    assert 0 < m.prepare_peak_mib <= m.peak_mib
    # It doesn't simulate, so the capacity model leaves it out.
    assert set(CapacityModel.fit([*MEASURED, m]).fits) == {"hcf"}


# 27 MiB on macOS at M9, imports included. The memory job runs on Linux, where the skeleton's
# allocations run about four times macOS's.
@pytest.mark.limit_memory("160 MB")
def test_a_tiny_case_stays_within_its_memory(tiny_world: Path) -> None:
    measure(dataclasses.replace(TINY, scope="hcf"), tiny_world)


def test_each_case_runs_in_its_own_process_and_reuses_its_world(tmp_path: Path) -> None:
    case = dataclasses.replace(TINY, name="tiny-hcf-1", scope="hcf", trajectories=1)
    (first,) = scale.run([case], tmp_path)
    world = scale.world_path(case, tmp_path)
    made = world.stat().st_mtime_ns
    (again,) = scale.run([case], tmp_path)
    assert world.stat().st_mtime_ns == made
    assert (first.case, first.employees) == (again.case, again.employees)


def test_the_child_prints_its_world_and_its_measurement(
    tiny_world: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = json.dumps(dataclasses.asdict(dataclasses.replace(TINY, scope="hcf", trajectories=1)))
    assert scale.main(["world", spec, str(tiny_world.parent)]) == 0
    assert capsys.readouterr().out.strip() == str(tiny_world)
    assert scale.main(["case", spec, str(tiny_world)]) == 0
    assert json.loads(capsys.readouterr().out)["case"] == TINY.name


# --- the capacity model -----------------------------------------------------------------------


def test_a_nonnegative_fit_drops_a_term_that_would_go_negative() -> None:
    x = np.column_stack([np.ones(4), np.array([1.0, 2.0, 3.0, 4.0])])
    assert nonnegative_fit(x, 2 + 3 * x[:, 1]) == pytest.approx([2.0, 3.0])
    falling = 10 - x[:, 1]  # the slope wants to be negative
    fitted = nonnegative_fit(x, falling)
    assert fitted[1] == 0
    assert fitted[0] == pytest.approx(falling.mean())


def test_the_model_recovers_an_exactly_linear_machine() -> None:
    model = CapacityModel.fit(MEASURED)
    fit = model.fits["hcf"]
    assert fit.prepare == pytest.approx((0.5, 2e-5))
    assert fit.simulate == pytest.approx((1.0, 1e-6))
    assert fit.memory == pytest.approx((150, 0.01, 0.002))
    assert max(fit.worst.values()) < 1e-6
    p = model.predict(100_000, 10, 12, "hcf")
    assert p.peak_mib == pytest.approx(150 + 1_000 + 2_000)
    assert p.simulate_seconds == pytest.approx(1.0 + 12.0)
    assert model.predict(100_000, 10, 12, "hcf", streaming=True).peak_mib == pytest.approx(1_350)
    assert CapacityModel.from_dict(model.to_dict()) == model


def test_a_plan_says_whether_a_run_fits_must_stream_or_cant() -> None:
    model = CapacityModel.fit(MEASURED)
    # 100k employees: 1,150 MiB before holding anything, then 200 MiB per trajectory held.
    assert model.most_held(100_000, "hcf", 2_000) == 4
    assert model.plan(100_000, 4, 12, "hcf", 2_000).advice == "fits as it is"
    must = model.plan(100_000, 10, 12, "hcf", 2_000)
    assert not must.fits
    assert must.advice == "stream its trajectories: it can hold 4 at once"
    cant = model.plan(100_000, 10, 12, "hcf", 1_200)
    assert cant.most_held == 0
    assert cant.advice == "doesn't fit, even streaming its trajectories"


def test_the_model_refuses_what_it_cant_fit_or_predict() -> None:
    with pytest.raises(CapacityError, match="three cases or more"):
        Fit.of(MEASURED[:2])
    model = CapacityModel.fit(MEASURED)
    with pytest.raises(CapacityError, match="no fit for the scope 'hcf\\+sbc'"):
        model.predict(10, 1, 1, "hcf+sbc")
    with pytest.raises(CapacityError, match="at least one employee"):
        model.predict(0, 1, 1, "hcf")
    with pytest.raises(CapacityError, match="version 1"):
        CapacityModel.from_dict({"version": 0})
    flat = CapacityModel({"hcf": dataclasses.replace(model.fits["hcf"], memory=(150, 0.01, 0.0))})
    assert flat.most_held(1_000, "hcf", 1_000) == 2**31 - 1


# --- baselines --------------------------------------------------------------------------------


def test_a_baseline_round_trips(tmp_path: Path) -> None:
    baseline = Baseline.of(MEASURED)
    assert baseline.machine == machine()
    baseline.save(tmp_path / "b.json")
    assert Baseline.load(tmp_path / "b.json") == baseline
    with pytest.raises(BaselineError, match="can't read the baseline"):
        Baseline.load(tmp_path / "missing.json")
    (tmp_path / "old.json").write_text(json.dumps({"version": 0}))
    with pytest.raises(BaselineError, match="version 1"):
        Baseline.load(tmp_path / "old.json")


def test_a_regression_past_twenty_percent_is_flagged() -> None:
    baseline = Baseline.of(MEASURED)
    here = dict(baseline.machine)
    first = MEASURED[1]
    slower = dataclasses.replace(first, simulate_seconds=first.simulate_seconds * 1.3)
    bigger = dataclasses.replace(MEASURED[2], peak_mib=MEASURED[2].peak_mib * 1.21)
    jitter = dataclasses.replace(MEASURED[0], prepare_seconds=MEASURED[0].prepare_seconds * 1.4)
    faster = dataclasses.replace(MEASURED[3], simulate_seconds=0.1)
    new = dataclasses.replace(first, case="unseen")
    result = compare([slower, bigger, jitter, faster, new], baseline, here=here)
    assert [(f.case, f.measure) for f in result.flags] == [
        (first.case, "simulate_seconds"),
        (MEASURED[2].case, "peak_mib"),
    ]
    assert result.new == ("unseen",)
    assert result.same_machine
    assert result.flagged
    assert str(result.flags[0]).endswith("(+30%)")
    assert jitter.prepare_seconds - MEASURED[0].prepare_seconds < MEASURES["prepare_seconds"]
    assert not compare(MEASURED, baseline, here=here).flagged
    assert not compare(MEASURED, baseline, here={"system": "other"}).same_machine
    assert compare([slower], baseline, threshold=0.5, here=here).flags == ()
    with pytest.raises(BaselineError, match="positive fraction"):
        compare(MEASURED, baseline, threshold=0)
    assert Flag("c", "peak_mib", 10, 0).growth == float("inf")


def test_the_stored_baseline_covers_the_suite_and_fits_itself() -> None:
    stored = Baseline.load(ROOT / BASELINE_PATH)
    assert [m.case for m in stored.measurements] == [c.name for c in (*SUITE, *LARGE)]
    assert set(stored.capacity.fits) == {"hcf", "hcf+sbc"}
    assert {"system", "machine", "cpus", "memory_gib", "python"} <= set(stored.machine)
    for m in (m for m in stored.measurements if m.scope in SIMULATED):
        fit = stored.capacity.fits[m.scope]
        p = stored.capacity.predict(m.employees, m.trajectories, m.months, m.scope, m.streaming)
        error = abs(p.peak_mib - m.peak_mib) / max(m.peak_mib, 32.0)
        assert error <= fit.worst["memory"] + 1e-3, m.case


# --- the commands -----------------------------------------------------------------------------


def test_capacity_plans_a_run_from_the_stored_baseline(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    baseline = str(ROOT / BASELINE_PATH)
    args = ["capacity", "--baseline", baseline, "--employees", "7000", "--trajectories", "4"]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "7,000 employees, 4 trajectories, 12 months, hcf+sbc:" in out
    assert "within 16 GiB: fits as it is" in out
    huge = ["capacity", "--baseline", baseline, "--employees", "5000000", "--memory-gib", "8"]
    assert cli.main(huge) == 0
    assert "doesn't fit, even streaming" in capsys.readouterr().out
    assert cli.main(["capacity", "--baseline", str(tmp_path / "x.json"), "--employees", "5"]) == 2


def test_compare_exits_3_on_a_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = tmp_path / "b.json"
    Baseline.of(MEASURED).save(baseline)
    run = tmp_path / "run.json"
    rows = [dataclasses.asdict(m) for m in MEASURED]
    run.write_text(json.dumps(rows))
    args = ["compare", "--run", str(run), "--baseline", str(baseline)]
    assert cli.main(args) == 0
    assert "no regressions past 20%" in capsys.readouterr().out
    rows[1]["simulate_seconds"] *= 2
    rows.append(rows[0] | {"case": "unseen"})
    run.write_text(json.dumps(rows))
    assert cli.main(args) == 3
    out = capsys.readouterr().out
    assert "FLAGGED" in out
    assert "new      unseen" in out
    run.write_text("not json")
    assert cli.main(args) == 2


def test_scale_writes_measurements_and_a_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ran: list[list[str]] = []

    def fake(cases: list[Case], cache: Path, limit: float | None) -> list[Measurement]:
        ran.append([c.name for c in cases])
        return MEASURED

    monkeypatch.setattr(scale, "run", fake)
    out, base = tmp_path / "run.json", tmp_path / "base.json"
    args = ["scale", "--cache-dir", str(tmp_path), "--out", str(out), "--baseline-out", str(base)]
    assert cli.main(args) == 0
    assert ran == [[c.name for c in SUITE]]
    assert len(json.loads(out.read_text())) == len(MEASURED)
    assert Baseline.load(base).measurements == tuple(MEASURED)
    assert cli.main(["scale", "--cache-dir", str(tmp_path), "--suite", "large"]) == 0
    assert ran[-1] == [c.name for c in LARGE]
    assert cli.main(["scale", "--cache-dir", str(tmp_path), "--suite", "all"]) == 0
    assert ran[-1] == [c.name for c in (*SUITE, *LARGE)]
    assert cli.main(["scale", "--cache-dir", str(tmp_path), "--case", "small-hcf-2"]) == 0
    assert ran[-1] == ["small-hcf-2"]

    monkeypatch.setattr(scale, "run", lambda *_: MEASURED[:2])
    assert cli.main(["scale", "--cache-dir", str(tmp_path), "--baseline-out", str(base)]) == 1
    assert "can't fit a capacity model" in capsys.readouterr().out

    def broken(*_: object) -> list[Measurement]:
        raise BenchError("the 'tiny-hcf-2' case passed its memory limit of 1 GiB")

    monkeypatch.setattr(scale, "run", broken)
    assert cli.main(["scale", "--cache-dir", str(tmp_path)]) == 1
    assert "memory limit" in capsys.readouterr().out


# --- profiling --------------------------------------------------------------------------------


def test_profiling_writes_statistics_and_a_report(tiny_world: Path, tmp_path: Path) -> None:
    from hcsim.bench.profiling import profile_here

    m = profile_here(TINY, tiny_world, tmp_path, top=5)
    assert 0 < m.prepare_peak_mib <= m.peak_mib
    report = (tmp_path / f"{TINY.name}.txt").read_text()
    assert report.startswith(f"case {TINY.name}: tiny, hcf+sbc, 2 trajectories of 3 months")
    assert "set it" in report
    assert "## hcsim's functions, by cumulative time" in report
    assert "## every function, by its own time" in report
    assert "loop.py" in report  # the simulation, among the top five
    assert (tmp_path / f"{TINY.name}.prof").stat().st_size > 0


def test_a_case_is_profiled_in_its_own_process(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = dataclasses.replace(TINY, name="small-hcf-2", preset="tiny", scope="hcf")
    monkey = pytest.MonkeyPatch()
    monkey.setitem(CASES, "small-hcf-2", case)  # the tiny world, under a suite case's name
    try:
        args = ["profile", "--case", "small-hcf-2", "--cache-dir", str(tmp_path / "w")]
        assert cli.main([*args, "--out", str(tmp_path / "p"), "--top", "3"]) == 0
    finally:
        monkey.undo()
    out = capsys.readouterr().out
    assert "case small-hcf-2: tiny, hcf" in out
    assert (tmp_path / "p" / "small-hcf-2.prof").exists()


def test_a_failed_profile_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from hcsim.bench import profiling

    def broken(*_: object) -> object:
        raise BenchError("the 'tiny-hcf-2' profile failed:\nboom")

    monkeypatch.setattr(profiling, "profile", broken)
    args = ["profile", "--case", "tiny-hcf-2", "--cache-dir", str(tmp_path), "--out", str(tmp_path)]
    assert cli.main(args) == 1
    assert "boom" in capsys.readouterr().out


def test_an_older_measurement_reads_without_its_prepared_peak() -> None:
    row = dataclasses.asdict(MEASURED[0])
    del row["prepare_peak_mib"]
    assert Measurement(**row).prepare_peak_mib == 0.0
