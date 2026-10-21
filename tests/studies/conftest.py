"""Shared inputs for the study tests: a tiny world and a guidance artifact fitted on it."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from hcsim.guidance.pipeline import PipelineConfig
from hcsim.guidance.pipeline import run as fit_guidance
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world


@pytest.fixture(scope="session")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("study")
    world = root / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=5)), world)
    with duckdb.connect(str(world), read_only=True) as con:
        fit_guidance(con, PipelineConfig()).artifact.save(root / "guidance")
    return world, root / "guidance"
