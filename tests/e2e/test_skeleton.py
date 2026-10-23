"""The walking skeleton end to end: a tiny world, the guidance fit, and a study run, through the
installed `hcsim` command, then the milestone's acceptance checks.

- Quotas realize exactly, the H and O identities are exact, and every tranche ties to within
  0.01, over a 12-month, 20-trajectory run.
- Two runs with the same seed, in separate processes, give bit-identical results.
- Each scorecard matches its committed baseline. After an intentional change, regenerate the
  baselines with `make baselines` and commit them in the same PR.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from hcsim.cli import main
from hcsim.studies import baseline
from hcsim.studies.harness import run_study
from hcsim.studies.profile import StudyProfile
from hcsim.studies.scorecard import results_digest

pytestmark = pytest.mark.e2e

BASELINES = Path(__file__).parent / "baselines"
HCSIM = str(Path(sys.executable).parent / "hcsim")
PROFILES = {
    # The milestone's acceptance run: guided, with SBC.
    "skeleton": "name: skeleton\nseed: 2026\ntrajectories: 20\nmonths: 12\n"
    "mode: guided_forecast\nscope: hcf+sbc\n",
    # The reference study: the same run, unguided, so every domain draws independently.
    "reference_unguided": "name: reference_unguided\nseed: 2026\ntrajectories: 20\n"
    "months: 12\nmode: unguided\nscope: hcf+sbc\n",
}


def hcsim(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([HCSIM, *map(str, args)], capture_output=True, text=True, check=False)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("e2e")
    generated = hcsim(
        "synth", "generate", "--preset", "tiny", "--seed", "1", "--out", root / "world.duckdb"
    )
    assert generated.returncode == 0, generated.stderr
    fitted = hcsim("guidance", "fit", "--world", root / "world.duckdb", "--out", root / "guidance")
    assert fitted.returncode == 0, fitted.stderr
    return root / "world.duckdb", root / "guidance"


def study_args(name: str, world: tuple[Path, Path], out: Path) -> list[str]:
    profile = out.parent / f"{name}.yaml"
    profile.write_text(PROFILES[name])
    args = ["study", "run", "--profile", str(profile), "--world", str(world[0]), "--out", str(out)]
    if "guided_forecast" in PROFILES[name]:
        args += ["--guidance", str(world[1])]
    return args


def check_baseline(name: str, card: pd.DataFrame) -> None:
    path = BASELINES / f"{name}.csv"
    if os.environ.get("HCSIM_UPDATE_BASELINES"):
        baseline.write(card, path)
        return
    assert path.exists(), f"no baseline at {path}; run `make baselines`"
    differences = baseline.compare(card, baseline.load(path))
    assert not differences, "the scorecard moved from its baseline:\n  " + "\n  ".join(differences)


def test_the_skeleton_meets_the_milestone_criteria(
    world: tuple[Path, Path], tmp_path: Path
) -> None:
    out = tmp_path / "subprocess"
    ran = hcsim(*study_args("skeleton", world, out))
    assert ran.returncode == 0, ran.stdout + ran.stderr

    card = pd.read_csv(out / "scorecard.csv").set_index("metric")
    assert card["passed"].all()
    value = card["value"].astype(float).to_dict()
    assert value["quota_violations"] == 0
    assert value["headcount_identity_max_gap"] == 0
    assert value["opening_identity_max_gap"] == 0
    assert value["accounting_identity_max_residual"] <= 0.01
    # Three guided categories, every month of every trajectory.
    assert str(card.at["quota_violations", "detail"]) == "720 guided draws checked"

    # The same seed in another process gives bit-identical results and scorecard.
    again = tmp_path / "in_process"
    assert main(study_args("skeleton", world, again)) == 0
    first = json.loads((out / "run.json").read_text())
    second = json.loads((again / "run.json").read_text())
    assert first["results_digest"] == second["results_digest"]
    assert (out / "scorecard.csv").read_bytes() == (again / "scorecard.csv").read_bytes()

    check_baseline("skeleton", pd.read_csv(out / "scorecard.csv", dtype={"seed": str}))


def test_the_reference_study_matches_its_baseline(world: tuple[Path, Path], tmp_path: Path) -> None:
    out = tmp_path / "reference"
    assert main(study_args("reference_unguided", world, out)) == 0
    card = pd.read_csv(out / "scorecard.csv", dtype={"seed": str})
    assert card["passed"].all()
    check_baseline("reference_unguided", card)


def test_another_seed_moves_the_results(world: tuple[Path, Path], tmp_path: Path) -> None:
    profile = StudyProfile.from_mapping({"name": "x", "seed": 1, "trajectories": 2, "months": 3})
    a = run_study(profile, *world)
    b = run_study(profile.model_copy(update={"seed": 2}), *world)
    assert a.passed
    assert b.passed
    # Guided counts can match across seeds by construction, so compare every raw result.
    assert results_digest(a.result) != results_digest(b.result)


# 60 MiB on the Linux runner, where the memory job runs, at M1, and 99 MiB by M5 (16 and 32 MiB
# on macOS). The growth is the run's SBC. Since M6 seeds the workforce's unvested equity, each
# trajectory's long-form expense rows start at the run's first month, with what came before as
# an opening balance, and are released once summed into its SBC frame: 48 MiB on macOS.
@pytest.mark.limit_memory("128 MB")
def test_the_skeleton_run_stays_within_its_memory(world: tuple[Path, Path]) -> None:
    profile = StudyProfile.from_mapping(
        {"name": "skeleton", "seed": 2026, "trajectories": 20, "months": 12}
    )
    assert run_study(profile, *world).passed
