"""Shared worlds for the synth tests. Generating `small` takes about two seconds, so the
tests that need its statistical power share one."""

from __future__ import annotations

import pytest

from hcsim.synth.config import preset
from hcsim.synth.world import World, generate_world


@pytest.fixture(scope="session")
def small_world() -> World:
    return generate_world(preset("small", seed=3))
